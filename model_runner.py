from datetime import datetime, timedelta
import pytz
import numpy as np
import os
import time
from collections import deque
from multiprocessing import Pool
import io
import boto3
import multiprocessing as mp
import concurrent.futures
import logging

import db
import schedule
from abc import ABC, abstractmethod
import threading
import psycopg2 as pg
import psycopg2.extras
import uuid
import s3_models
import psutil

from dynamic_utils import DynamicRingBuffer, parse_time_string_with_colon_offset, interval_to_buckets, timebucket_files, s3_path_to_datetime, get_npz

logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger("schedule").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)


#Makes postgres UUID[] type return as list of strings
psycopg2.extensions.register_type(
    psycopg2.extensions.new_array_type(
        (2951,), 'UUID[]', psycopg2.STRING))

def db_query(q, args=None, fetch_results=True):
    with pg.connect(dbname=os.environ['db_name'],
                    user=os.environ['db_user'],
                    password=os.environ['db_password'],
                    host=os.environ['db_host'],
                    port=os.environ['db_port']) as conn:

        conn.set_session(autocommit = True)

        with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
            if args:
                cur.execute(q, args)
            else:
                cur.execute(q)

            if fetch_results:
                res = cur.fetchall()
                return res

def interval_to_flat_array_threaded(sensor_id, start, end, sample_rate=57000, multiprocessing=False):
    "Returns all dynamic data for a sensor_id, start, and end time in a ring buffer"
    import concurrent.futures
    assert end > start
    start = start.astimezone(pytz.utc).replace(microsecond=0)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        t = t.replace(microsecond=0)
        return t >= start and t <= end

    timebuckets = interval_to_buckets(start,end)

    num_seconds = (end - start).seconds
    num_samples = sample_rate * num_seconds
    buffer = DynamicRingBuffer(num_samples)

    t0 = time.time()

    if multiprocessing:
        with mp.Pool(64) as pool:
            files = pool.starmap(timebucket_files, [(sensor_id, timebucket) for timebucket in timebuckets])
            files = [val for sublist in files for val in sublist]
            files_within_interval = [f for f in files if within_interval(f)]
            logging.debug("Downloading " + str(len(files_within_interval)) +  " files")
            data = pool.map(get_npz, files_within_interval)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            #args = [(sensor_id, timebucket) for timebucket in timebuckets]
            sensor_ids = [sensor_id for i in range(len(timebuckets))]
            files = executor.map(timebucket_files, sensor_ids, timebuckets)
            files = [val for sublist in files for val in sublist]
            files_within_interval = [f for f in files if within_interval(f)]
            logging.debug("Downloading " + str(len(files_within_interval)) +  " files")
            data = executor.map(get_npz, files_within_interval)

    logging.debug("time to download: " + str(time.time() - t0))

    for d in data:
        buffer.append(d)

    num_files = len(files_within_interval)
    if num_files > 0:
        average_sample_rate = len(buffer) / num_files
        earliest_file = s3_path_to_datetime(files_within_interval[0])
        latest_file = s3_path_to_datetime(files_within_interval[-1])

    else:
        average_sample_rate = 0
        earliest_file = None
        latest_file = None

    return buffer, num_files, average_sample_rate, earliest_file, latest_file


class ModelData():
    """
    Class to fetch sensor data for window and update window dates for live mode
    frequency : how frequently model should pull new data (seconds)
    window_size : (seconds)
    initialization_time : time to initialize end of window at
    """
    def __init__(self,
                 frequency,
                 window_size,
                 dynamic_id=None,
                 static_id=None,
                 max_sample_rate=57000,
                 initialization_time = None,
                 live_data=True):
        self.frequency = frequency
        self.window_size = window_size
        self.dynamic_id = dynamic_id
        self.static_id = static_id
        self.max_sample_rate = max_sample_rate
        self.dynamic_buffer = DynamicRingBuffer(window_size * max_sample_rate)
        self.static_buffer = deque(maxlen=window_size)
        self.end = initialization_time
        self.initialized = False
        self.current_window_size = 0
        self.live_data = live_data

        self.retries = 0
        self.max_retries = 30


        if not self.end:
            self.end = datetime.now().astimezone(pytz.utc).replace(microsecond=0)

        self.start = self.end - timedelta(seconds=self.window_size)


    def increment_interval_historical(self, num_new_files=0):
        assert self.start and self.end
        delta = timedelta(seconds=self.frequency)

        self.start += delta
        self.end += delta

        #in case of gaps in data drop data from buffer which is no longer in window
        seconds_to_drop = self.frequency - num_new_files

        if self.static_id:
            for i in range(seconds_to_drop):
                if len(self.static_buffer) > 0:
                    self.static_buffer.popleft()

        if self.dynamic_id:
            for i in range(seconds_to_drop):
                if len(self.dynamic_buffer) > self.max_sample_rate:
                    self.dynamic_buffer.popleftn(self.max_sample_rate)


    def increment_interval_live(self, num_new_files=0):
        "Updates window start and end."
        assert self.start and self.end

        if num_new_files:

            delta = timedelta(seconds=num_new_files)

            #buffer full
            if self.current_window_size >= self.window_size:
                self.start += delta
                self.end += delta

            #Buffer not full only increment end
            else:
                self.end += delta


        #jump window forward after max retries
        elif self.retries >= self.max_retries:
            logging.warning(f"Unable to find interval data within {self.max_retries} retries: jumping to present. Dynamic: {self.dynamic_id}, Static: {self.static_id}")

            #ensure dont jump into the future
            now = datetime.now().astimezone(pytz.utc).replace(microsecond=0)
            delta = (now - self.end)
            self.end += delta
            self.start += delta
            self.retries = 0

            #drop buffer data outside new window
            if self.static_id:
                for i in range(delta.seconds):
                    if len(self.static_buffer) > 0:
                        self.static_buffer.popleft()

            if self.dynamic_id:
                for i in range(delta.seconds):
                    if len(self.dynamic_buffer) >= self.max_sample_rate:
                        self.dynamic_buffer.popleftn(self.max_sample_rate)

        else:
            self.retries += 1


    def interval_to_fetch(self):
        #data never fetched before
        if not self.initialized:
            start = self.start
            end = self.end

        #cold start: not enough data populated yet
        elif self.current_window_size < self.window_size:
            size_difference = self.window_size - self.current_window_size
            start = self.end - timedelta(seconds=size_difference)
            end = self.end

        #move to next window
        else:
            start = self.end
            end = self.end + timedelta(seconds=self.frequency)

        return start, end


    def update_dynamic_data(self, update_interval=False):
        """
        Fetches full window if not initalized.
        Otherwise fetches 'frequency' sized window starting at current end time
        """
        assert self.dynamic_id and self.start and self.end

        start, end = self.interval_to_fetch()
        new_data, num_files, sample_rate, _, _ = interval_to_flat_array_threaded(self.dynamic_id,
                                                                                 start,
                                                                                 end,
                                                                                 sample_rate=self.max_sample_rate,
                                                                                 multiprocessing=(not self.live_data)
        )

        self.sample_rate = sample_rate
        self.dynamic_buffer.append(new_data)
        return num_files


    def update_static_data(self, update_interval=False):
        assert self.static_id and self.start and self.end

        start, end = self.interval_to_fetch()
        #should we init() every time? do we need to close db conn?
        q = """SELECT *
               FROM monitoring.sensor_data
               WHERE sensor_id = %s
               AND time >= %s
               AND time <= %s
               ORDER BY time asc"""
        results = db_query(q, (self.static_id, start, end))
        for res in results:
            self.static_buffer.append(res)
        return len(results)


    #TODO: num_new_files is bad assumption. Files are sometimes skipped
    def update_data(self):
        num_new_files = 0

        if self.dynamic_id:
            num_new_files = self.update_dynamic_data()

        #prioritize new dynamic files
        if self.static_id:
            num_new_static_files =  self.update_static_data()
            if not num_new_files:
                num_new_files = num_new_static_files


        if self.current_window_size < self.window_size:
            self.current_window_size += num_new_files

        #start,end already set if not initialized
        if self.initialized:
            if self.live_data:
                self.increment_interval_live(num_new_files)
            else:
                self.increment_interval_historical(num_new_files)

        self.initialized = True

        return num_new_files > 0


    def dynamic_data(self):
        return self.dynamic_buffer

    def static_data(self):
        return list(self.static_buffer)


def well_sensors(api_14):
    "Fetches well static and dynamic sensor ids"
    q = """
        SELECT sensors.id, pressure_type
        FROM monitoring.sensors
        JOIN monitoring.wells ON monitoring.wells.id = well_id
        JOIN monitoring.sensor_models ON monitoring.sensor_models.id = sensor_model_id
        WHERE api = %s"""
    res = db_query(q, (api_14,))
    sensors = {}
    for row in res:
        id = row['id']
        pressure_type = row['pressure_type']
        sensors[pressure_type] = id
    return sensors


def update_job_last_updated(job_id):
    db_query("""UPDATE monitoring.model_queue
                    SET last_update=%s
                    WHERE id=%s""",
             (datetime.now().astimezone(pytz.utc), job_id), fetch_results=False)


class ModelRunner():
    """
    Everything needed to run model on data

    Parameters:
    ----------
    model : Class that implements AbstractModel - pass data to this and get events back
    config : Dict defining period and window size to run model on
    """
    def __init__(self, config, model, initialization_time=None, live_data=False):

        self.frequency = config["frequency"]
        self.window_size = config["window_size"]
        self.well_id = config["well_id"]
        self.job_id = config["id"]
        self.hub_id = config['hub_id']
        self.model = model
        self.version = config['model_version']

        self.timeseries = config.get("timeseries", False)

        static_ids = config.get('static_sensor_ids')
        dynamic_ids = config.get('dynamic_sensor_ids')
        self.static_id = None
        self.dynamic_id = None

        if static_ids:
            self.static_id = static_ids[0]

        if dynamic_ids:
            self.dynamic_id = dynamic_ids[0]

        if not initialization_time:
            initialization_time =  datetime.now().astimezone(pytz.utc).replace(microsecond=0)

        self.model_data = ModelData(self.frequency,
                                    self.window_size,
                                    dynamic_id=self.dynamic_id,
                                    static_id=self.static_id,
                                    initialization_time=initialization_time,
                                    live_data=live_data)


    def calculate_result_times(self, model_results):
        """For events with big window calculate start/end time of window based"""
        sample_rate = self.model_data.max_sample_rate
        start_idx = model_results['start_idx']
        end_idx = model_results['end_idx']
        start_elapsed_sec = start_idx / sample_rate
        end_elapsed_sec = end_idx / sample_rate

        event_start = self.model_data.start + timedelta(seconds=start_elapsed_sec)
        event_end = self.model_data.start + timedelta(seconds=end_elapsed_sec)

        return event_start, event_end

    def infer(self):
        dynamic_data = self.model_data.dynamic_data()
        static_data = self.model_data.static_data()
        result = self.model.infer(dynamic_data, static_data)
        """
        if results:
            for result in results:
                start, end = self.calculate_result_times(result)
                result['start_time'] = start
                result['end_time'] = end
            return results
        """
        return result


    def update_data_and_infer(self):
        is_new_data = self.model_data.update_data()
        if is_new_data:
            results = self.infer()
            return results


    def insert_timeseries_event(self, event):
        q = """INSERT INTO monitoring.timeseries_events
               (time, event, well_id, value, model_version)
               VALUES (%s, %s, %s, %s, %s)"""

        args = (self.model_data.end.astimezone(pytz.utc),
                event['event'],
                self.well_id,
                event['value'],
                self.version)

        db_query(q, args, fetch_results=False)


    def insert_event(self, event):
        q = """INSERT INTO monitoring.events
                 (id, well_id, event, start_time, end_time)
                 VALUES (%s, %s, %s, %s, %s)"""

        if event:
            results = self.calculate_result_times(event)
            event_id = str(uuid.uuid4())
            args = (event_id,
                    self.well_id,
                    results.get('event'),
                    results.get('start_time'),
                    results.get('end_time'))
            db_query(q, args)


    def detect_and_insert_events(self):
        is_new_data = self.model_data.update_data()
        if is_new_data:
            result = self.infer()
            if self.timeseries and result:
                self.insert_timeseries_event(result)
            else:
                if result:
                    self.insert_event(result)

        update_job_last_updated(self.job_id)


def partition(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class HistoricalModelRunner(ModelRunner):
    """
    Everything needed to run model on data

    Parameters:
    ----------
    model : Class that implements infer(dynamic_data, static_data) - pass data to this and get events back
    config : Dict defining period and window size to run model on
    """
    def __init__(self, config, model, initialization_time, window_multipler=5):

        self.frequency = config["frequency"]
        self.well_id = config["well_id"]
        self.window_size = config["window_size"]
        self.job_id = config["id"]
        self.hub_id = config['hub_id']
        self.model = model
        self.timeseries = config["timeseries"]
        static_ids = config.get('static_sensor_ids')
        dynamic_ids = config.get('dynamic_sensor_ids')
        self.static_id = None
        self.dynamic_id = None
        self.window_multipler=window_multipler

        if static_ids:
            self.static_id = static_ids[0]

        if dynamic_ids:
            self.dynamic_id = dynamic_ids[0]


        self.model_data = ModelData(self.frequency*window_multipler,
                                    self.window_size*window_multipler,
                                    dynamic_id=self.dynamic_id,
                                    static_id=self.static_id,
                                    initialization_time=initialization_time,
                                    live_data=False)

    def infer(self):
        dynamic_data = self.model_data.dynamic_data()
        static_data = self.model_data.static_data()

        partitioned_dynamic = np.array(dynamic_data).reshape((self.window_multipler, -1))
        partitioned_static = partition(static_data, self.window_multipler)

        results = []
        for dynamic, static in zip(partitioned_dynamic, partitioned_static):
            res = self.model.infer(dynamic, static)
            results.append(res)

        return results


def monitoring_group_status(monitoring_group_id):
    q = """
        SELECT status
        FROM monitoring.monitoring_groups
        WHERE id = %s"""
    result = db_query(q, (monitoring_group_id,))
    if len(result) > 0:
        return result[0]['status']
    else:
        return None


def get_hub_status(hub_id):
    q = """
        SELECT status
        FROM monitoring.hubs
        WHERE id = %s"""
    result = db_query(q, (hub_id,))
    if len(result) > 0:
        return result[0]['status']
    else:
        return None


def run_threaded(job_func):
    """Used to run jobs in separate thread so jobs do not block the main thread enusring jobs are launched on time
       https://schedule.readthedocs.io/en/stable/faq.html#how-to-execute-jobs-in-parallel"""
    job_thread = threading.Thread(target=job_func, daemon=True)
    job_thread.start()


def model_runner_process(job_config):
    """To be ran in a separate process.
       Runs model peridoically until hub marked complete.
       Runs model in seperate thread from process main thread to prevent blocking the scheduler
       https://schedule.readthedocs.io/en/stable/faq.html#how-to-execute-jobs-in-parallel"""
    try:
        model_type = job_config['model_type']
        model_name = job_config['model_name']
        model_version = job_config['model_version']

        #TODO: redo pickled py model class to take dictionary for config like tf model
        if model_type == 'py':
            model_class, _ = s3_models.download_py_model(model_name, model_version)
            model_instance = model_class(job_config['window_size'])

        if model_type =='tf':
            keras_model, metadata = s3_models.download_tf_model(model_name, model_version)
            #merge metadata and job specific config
            model_config = {**metadata, **job_config}
            model_instance = s3_models.TF_Model(keras_model, model_config)

        runnable_model = ModelRunner(job_config, model_instance, live_data=True)
        frequency = runnable_model.frequency

        #kick off a job before the scheduled job to allow model to block while downloading tons of files if necessary to start
        runnable_model.detect_and_insert_events()

        job = schedule.every(frequency).seconds.do(run_threaded, runnable_model.detect_and_insert_events)
        #kick off job immediately instead of waiting 'frequency' seconds
        job.run()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            hub_status = None
            while hub_status != "STOPPED":
                f = executor.submit(get_hub_status, runnable_model.hub_id)
                schedule.run_pending()
                time.sleep(0.25)
                hub_status = f.result()


        db_query("""UPDATE monitoring.model_queue
                    SET status='FINISHED'
                    WHERE id=%s""", (runnable_model.job_id,), fetch_results=False)

        logging.info("Job finished")
        logging.info(job_config)
        schedule.clear()

    except Exception as e:
        logging.error("Error in model_runner_process: " +  str(e))


def pull_next_job(failed=False):
    pending_q = """UPDATE monitoring.model_queue
           SET status = 'RUNNING'
           WHERE id = (
               SELECT id
               FROM monitoring.model_queue
               WHERE status = 'PENDING'
               FOR UPDATE SKIP LOCKED
               LIMIT 1
               )
       RETURNING * """

    pending = db_query(pending_q)
    if len(pending) > 0:
        return pending[0]

    if failed:
        #Jobs for active monitoring groups that havent been updated in 5 periods
        failed_q = """UPDATE monitoring.model_queue
                      SET last_update = now()
                      WHERE id = (
                          SELECT id
                          FROM monitoring.model_queue
                          WHERE status = 'RUNNING'
                          AND last_update + make_interval(secs => 5*frequency) < now()
                          FOR UPDATE SKIP LOCKED
                          LIMIT 1
                         )
                      RETURNING *
                   """

        failed = db_query(failed_q)
        if len(failed) > 0:
            return failed[0]

    return None


def scheduler_queue():
    max_jobs = psutil.cpu_count()
    current_jobs = 0
    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        while True:
            try:
                while current_jobs < max_jobs:
                    job_config = pull_next_job(failed=False)
                    if job_config:
                        logging.info("Starting new job")
                        logging.info(job_config)

                        f = executor.submit(model_runner_process, job_config)
                        futures.append(f)
                        current_jobs += 1
                    time.sleep(0.5)


                done, not_done = concurrent.futures.wait(futures,
                                                         timeout=0.05,
                                                         return_when=concurrent.futures.FIRST_COMPLETED)

                futures = list(not_done)
                current_jobs = len(not_done)
                time.sleep(0.5)

            except KeyboardInterrupt as e:
                for f in futures:
                    f.cancel()
                print(e)
                break


if __name__ == "__main__":
    scheduler_queue()
