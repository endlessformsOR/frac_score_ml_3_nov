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
import csv
import argparse
import dateutil

from dynamic_utils import DynamicRingBuffer, parse_time_string_with_colon_offset, interval_to_buckets, timebucket_files, s3_path_to_datetime, get_npz, interval_to_flat_array

logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger("schedule").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)


#Makes postgres UUID[] type return as list of strings
psycopg2.extensions.register_type(
    psycopg2.extensions.new_array_type(
        (2951,), 'UUID[]', psycopg2.STRING))

def db_query(q, args=None, fetch_results=True, insert_multiple=False):
    with pg.connect(dbname=os.environ['db_name'],
                    user=os.environ['db_user'],
                    password=os.environ['db_password'],
                    host=os.environ['db_host'],
                    port=os.environ['db_port']) as conn:

        conn.set_session(autocommit = True)

        with conn.cursor(cursor_factory=pg.extras.RealDictCursor) as cur:
            if not insert_multiple:
                if args:
                    cur.execute(q, args)
                else:
                    cur.execute(q)

            if insert_multiple:
                psycopg2.extras.execute_values(cur, q, args)

            if fetch_results and not insert_multiple:
                res = cur.fetchall()
                return res


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
                 target_sample_rate=40000,
                 initialization_time = None,
                 live_data=True):
        self.frequency = frequency
        self.window_size = window_size
        self.dynamic_id = dynamic_id
        self.static_id = static_id
        self.target_sample_rate = target_sample_rate
        self.dynamic_buffer = DynamicRingBuffer(window_size * target_sample_rate)
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

    def increment_interval_historical(self, secs_downloaded=0):
        assert self.start and self.end
        delta = timedelta(seconds=self.frequency)

        self.start += delta
        self.end += delta

        #in case of gaps in data drop data from buffer which is no longer in window
        seconds_to_drop = self.frequency - secs_downloaded

        if self.static_id:
            for i in range(seconds_to_drop):
                if len(self.static_buffer) > 0:
                    self.static_buffer.popleft()

        if self.dynamic_id:
            for i in range(seconds_to_drop):
                if len(self.dynamic_buffer) > self.target_sample_rate:
                    self.dynamic_buffer.popleftn(self.target_sample_rate)


    def increment_interval_live(self, secs_downloaded=0):
        "Updates window start and end."
        assert self.start and self.end

        if secs_downloaded:

            delta = timedelta(seconds=secs_downloaded)

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
                    if len(self.dynamic_buffer) >= self.target_sample_rate:
                        self.dynamic_buffer.popleftn(self.target_sample_rate)

        else:
            self.retries += 1


    def interval_to_fetch(self):
        #data never fetched before
        if not self.initialized:
            start = self.start
            end = self.end

        #cold start: not enough data populated yet
        #only use in live mode
        elif self.live_data and self.current_window_size < self.window_size:
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
        new_data, num_files = interval_to_flat_array(self.dynamic_id,
                                                     start,
                                                     end,
                                                     target_sample_rate=self.target_sample_rate,
                                                     multiprocessing=(not self.live_data),
                                                     return_secs_downloaded = True
        )

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


    #TODO: secs_downloaded is bad assumption. Files are sometimes skipped
    def update_data(self):
        secs_downloaded = 0

        if self.dynamic_id:
            secs_downloaded = self.update_dynamic_data()

        #prioritize new dynamic files
        if self.static_id:
            num_new_static_files =  self.update_static_data()
            if not secs_downloaded:
                secs_downloaded = num_new_static_files


        if self.current_window_size < self.window_size:
            self.current_window_size += secs_downloaded

        #start,end already set if not initialized
        if self.initialized:
            if self.live_data:
                self.increment_interval_live(secs_downloaded)
            else:
                self.increment_interval_historical(secs_downloaded)

        self.initialized = True

        return secs_downloaded


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
        sample_rate = self.model_data.target_sample_rate
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
        secs_downloaded = self.model_data.update_data()
        if secs_downloaded:
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
        secs_downloaded = self.model_data.update_data()
        if secs_downloaded:
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

def flatten_list(l):
    """Flattens list of lists
       https://stackoverflow.com/a/952952"""
    return [item for sublist in l for item in sublist]


def get_well_sensors(well_id):
    q = """SELECT sensors.id, pressure_type, archived
           FROM monitoring.wells
           JOIN monitoring.sensors ON sensors.well_id = wells.id
           JOIN monitoring.sensor_models ON sensor_models.id = sensors.sensor_model_id
           WHERE wells.id =%s"""

    return db_query(q, (well_id,))


def sensor_time_bounds(sensor_id):
    "Finds earliest and latest data points for sensor"
    sensor_id = str(sensor_id)
    start_q = """SELECT time
                 FROM monitoring.sensor_data
                 WHERE sensor_id = %s
                 ORDER BY TIME ASC
                 LIMIT 1"""

    end_q = """SELECT time
               FROM monitoring.sensor_data
               WHERE sensor_id = %s
               ORDER BY TIME desc
               LIMIT 1"""

    start = (db_query(start_q, (sensor_id,)))
    end = (db_query(end_q, (sensor_id,)))

    if start:
        return start[0]['time'], end[0]['time']

def add_sensor_time_bounds(sensor):
            start, end = sensor_time_bounds(sensor['id'])
            sensor['start'] = start
            sensor['end'] = end


def date_overlap(a_start, a_end, b_start, b_end):
            latest_start = max(a_start, b_start)
            earliest_end = min(a_end, b_end)
            delta = (earliest_end - latest_start).total_seconds()
            overlap = max(0, delta)
            return overlap


def within_interval(d, start, end):
            return start < d < end


class HistoricalModelRunner2():
    def __init__(self, well_id, pressure_type, start_time, end_time, model, frequency, window_size, window_multipler=5, csv_file=None, db_save=False, model_version=None, target_sample_rate=40000):
        assert start_time < end_time

        self.well_id = well_id
        self.start_time = start_time
        self.end_time = end_time
        self.buffer_start = start_time
        self.buffer_end   = start_time + timedelta(seconds = window_size*window_multipler)
        self.model = model
        self.frequency = frequency
        self.window_size = window_size
        self.window_multipler = window_multipler
        self.pressure_type = pressure_type

        self.csv_file = csv_file
        self.db_save = db_save
        self.is_csv_header_written = False
        self.model_version = model_version
        self.target_sample_rate=40000


        sensors = [sensor for sensor in get_well_sensors(well_id) if sensor['pressure_type'] == self.pressure_type]
        [add_sensor_time_bounds(sensor) for sensor in sensors]
        sensors.sort(key=lambda x: x['start'])
        self.sensors = sensors


        self.model_data = []
        for sensor in self.sensors:
            if self.pressure_type == 'dynamic':
                self.model_data.append(ModelData(frequency*window_multipler,
                                                 window_size*window_multipler,
                                                 dynamic_id=sensor['id'],
                                                 initialization_time=start_time,
                                                 live_data=False,
                                                 target_sample_rate=target_sample_rate))
            if self.pressure_type == 'static':
                self.model_data.append(ModelData(frequency*window_multipler,
                                                 window_size*window_multipler,
                                                 static_id=sensor['id'],
                                                 initialization_time=start_time,
                                                 live_data=False,
                                                 target_sample_rate=target_sample_rate))

        print(self.sensors)


    def get_buffer_interval(self):
        return self.model_data[0].start, self.model_data[0].end


    def calculate_result_timestamp(self, buffer_start, result_idx, model_frequency):
        return buffer_start + timedelta(seconds = (result_idx * model_frequency))

    """
    def add_results_timestamps(self, results):
        buffer_start, buffer_end = self.get_buffer_interval()
        for i, result in enumerate(results):
            result['time'] = self.calculate_result_timestamp(buffer_start, i, self.frequency)
        return results
    """

    def save_results(self,results):
        if self.csv_file:
            with open(self.csv_file, 'a+') as csvfile:
                fields = ['time', 'well_id', 'event', 'value', 'model_version']
                writer = csv.DictWriter(csvfile, fieldnames=fields)

                if not self.is_csv_header_written:
                    writer.writeheader()
                    self.is_csv_header_written = True

                for res in results:
                    writer.writerow(res)
        if self.db_save:
            q = """INSERT INTO monitoring.timeseries_events
               (time, well_id, event, value, model_version)
               VALUES %s"""

            args = []
            for res in results:
                arg = tuple([res[k] for k in ('time', 'well_id', 'event', 'value', 'model_version')])
                args.append(arg)

            db_query(q, args, insert_multiple=True)

        else:
            return results


    def run_model(self):
        buffer_start, buffer_end = self.get_buffer_interval()
        while buffer_end < self.end_time:
            results = self.update_data_and_infer()
            if results:
                self.save_results(results)
            _, buffer_end = self.get_buffer_interval()


    def update_data_and_infer(self):
        buffer_start, buffer_end = self.get_buffer_interval()
        results = []
        for sensor, model_data in zip(self.sensors, self.model_data):
            sensor_start, sensor_end = [sensor[k] for k in ('start', 'end')]

            if date_overlap(sensor_start, sensor_end, self.buffer_start, self.buffer_end) > 0:
                secs_downloaded = model_data.update_data()

                if secs_downloaded:
                    dynamic_data = model_data.dynamic_data()
                    static_data =  model_data.static_data()

                    partitioned_dynamic = partition(dynamic_data, self.target_sample_rate)
                    partitioned_static = partition(static_data, self.window_multipler)

                    if self.pressure_type == 'static':
                        for static in partitioned_static:
                            res = self.model.infer([], static)
                            results.append(res)

                    if self.pressure_type == 'dynamic':
                        for dynamic in partitioned_dynamic:
                            res = self.model.infer(dynamic,[])
                            results.append(res)


            else:
                #model_data.update_data() calls increment_interval() already but if we dont call it
                #we need to increment manually
                model_data.increment_interval_historical()


        for idx, result in enumerate(results):
            result['time'] = self.calculate_result_timestamp(buffer_start, idx, self.frequency)
            result['well_id'] = self.well_id
            result['model_version'] = self.model_version

        return results


def test_window():
    from frac_score import FracScore

    window_multipler = 120
    dev_id = "56ef12c9-8eac-432a-8016-c56911bd38aa"
    dev_start = datetime(2020,7,15,21,21,00, tzinfo=pytz.utc)
    dev_end   = datetime(2020,7,15,21,30,10, tzinfo=pytz.utc)


    #live_start = datetime(2020, 8, 10).astimezone(pytz.utc),
    #live_end   = datetime(2020, 8, 10,1,0).astimezone(pytz.utc)
    #live_id = "04b0ee4f-42a6-47e1-9eb6-7b144255dc81"


    h = HistoricalModelRunner2(dev_id,
                               "dynamic",
                               dev_start,
                               dev_end,
                               FracScore(1),
                               1,
                               1,
                               window_multipler=window_multipler,
                               csv_file='test.csv',
                               db_save=False
    )

    iterations = 5
    #num_inferences = iterations * window_multipler
    num_inferences = (dev_end - dev_start).total_seconds()

    t0 = time.time()
    h.run_model()
    """
    for i in range(iterations):
        res = h.update_data_and_infer()
        print(res)
        #print([r['time'] for r in h.add_results_timestamps(res)])
    """
    t1 = time.time()

    print(num_inferences, "total inferences")
    print(t1 -t0, "seconds")
    print(num_inferences/ (t1 - t0), "inferences per second")



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
        self.current_start_time = initialization_time

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

    def update_data_and_infer(self):
        self.model_data.update_data()
        results = self.infer()
        results_start_time = self.current_start_time
        self.current_start_time += timedelta(seconds=self.window_size + self.window_multipler)
        return results_start_time, results

    def create_db_rows(self, start_time, results):
        rows = []
        for i, result in enumerate(results):
            rows.append(
                (start_time + timedelta(seconds=self.window_size),
                 result['event'],
                 self.well_id,
                 result['value'],
                 self.version)
            )
        return rows


    def insert_timeseries_events(self, events):
        q = """INSERT INTO monitoring.timeseries_events
               (time, event, well_id, value, model_version)
               VALUES %s"""

        args = (self.model_data.end.astimezone(pytz.utc),
                event['event'],
                self.well_id,
                event['value'],
                self.version)

        db_query(q, args, fetch_results=False, insert_multiple=True)

"""
from frac_score import FracScore
hw33_config = {"start": parse_time_string_with_colon_offset("2020-01-02T06:01:00-05:00"),
               "end":   parse_time_string_with_colon_offset("2020-01-12T21:53:00-05:00"),
               "dynamic_sensor_ids": ["4b65d71e-9512-4101-a32d-c6a06a1bfd71"],
               "static_sensor_ids": ["2db7044a-b4cc-4173-843a-3a82bb658df4"],
               "frequency": 1,
               "window_size": 1,
               "well_id": None,
               "id": None,
               "hub_id": None,
               "timeseries": True}

hw33_config = {"start": parse_time_string_with_colon_offset("2020-01-02T06:01:00-05:00"),
               "end":   parse_time_string_with_colon_offset("2020-01-12T21:53:00-05:00"),
               "dynamic_sensor_ids": ["4b65d71e-9512-4101-a32d-c6a06a1bfd71"],
               "static_sensor_ids": ["2db7044a-b4cc-4173-843a-3a82bb658df4"],
               "frequency": 1,
               "window_size": 1,
               "well_id": None,
               "id": None,
               "hub_id": None,
               "timeseries": True}


def tst():
    conf = hw33_config
    start_time = conf['start']
    current_time = start_time
    model = FracScore(window_size=1)
    model_runner = HistoricalModelRunner(conf, model, initialization_time=conf.get('start'), window_multipler=5)
    while current_time <= conf['end']:
        results = model_runner.update_data_and_infer()
        print(results)

def tst2():
    conf = hw33_config
    files = []
    for bucket in interval_to_buckets(conf['start'], conf['end']):
        print(timebucket_files("4b65d71e-9512-4101-a32d-c6a06a1bfd71", bucket))
        #files.append(timebucket_files("4b65d71e-9512-4101-a32d-c6a06a1bfd71", bucket))
    #return files

"""

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
    parser = argparse.ArgumentParser(description='Runs Stratalink models either in historical mode on one well or as a worker on live wells')

    parser.add_argument('--historical',
                        help='True/False. Run model in historical mode. Requires well_id, start,end, and model info',
                        required=False,
                        default=False,
                        type=bool)

    parser.add_argument('--well_id',
                        help='UUID of well to run historical models on',
                        required=False,)

    parser.add_argument('--pressure_type',
                        help="run model on 'dynamic' or 'static' sensor",
                        required=False,
                        default='dynamic')

    parser.add_argument('--start',
                        help='Historical start date. Iso format',
                        required=False,)

    parser.add_argument('--end',
                        help='Historical end date. Iso format',
                        required=False,)

    parser.add_argument('--model_type',
                        help='py for raw python or tf for tensorflow models',
                        required=False,
                        default='py')

    parser.add_argument('--model_name',
                        help='Name of model to run',
                        required=False,)

    parser.add_argument('--model_version',
                        help='version of model to run. Leave blank for latest',
                        required=False,)

    parser.add_argument('--batch_multiplier',
                        help="""How many multiples of the model window size to download at once.
                                A multiplier that results in a download window of 240 seconds seems optimal""",
                        required=False,
                        type=int,
                        default=120,)

    parser.add_argument('--csv',
                        help="Optional CSV file to save model output to",
                        required=False)

    parser.add_argument('--save_db',
                        help="True/False. Save model output to database",
                        required=False,
                        default=False,
                        type=bool,)

    args = vars(parser.parse_args())

    if args['historical']:
        start = dateutil.parser.isoparse(args['start'])
        end = dateutil.parser.isoparse(args['end'])
        well_id = args['well_id']
        pressure_type = args['pressure_type']
        model_type = args['model_type']
        model_name = args['model_name']
        model_version = args['model_version']
        csv_file = args['csv']
        save_db = args['save_db']
        window_multipler = args['batch_multiplier']


        if model_type == 'py':
            model_class, metadata  = s3_models.download_py_model(model_name, model_version)
            model_instance = model_class(metadata['window_size'])
            model_version = metadata['version']
            frequency = metadata['frequency']
            window_size = metadata['window_size']

            runner = HistoricalModelRunner2(well_id,
                                        pressure_type,
                                        start,
                                        end,
                                        model_instance,
                                        frequency,
                                        window_size,
                                        window_multipler,
                                        csv_file=csv_file,
                                        db_save=save_db,
                                        model_version=model_version)

            runner.run_model()

        if model_type =='tf':
            model, metadata = s3_models.download_tf_model(model_name, model_version)
            #TODO refactor s3_models.TF_Model to be compatable with HistoricalModelRunner





    else:
        scheduler_queue()
