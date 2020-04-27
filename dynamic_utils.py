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

import db
import s3fs
import zarr
import schedule
from abc import ABC, abstractmethod
import threading
import psycopg2 as pg
import psycopg2.extras
import uuid


S3_BUCKET = "sensor-data-live"
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

S3_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
NPZ_TIME_FORMAT = S3_TIME_FORMAT + ".npz"

#define these globally so we can parellize fetching data fns
#otherwise get cannot pickle class errors
s3_resource = boto3.resource('s3',
                             aws_access_key_id=ACCESS_KEY,
                             aws_secret_access_key=SECRET_KEY)
s3_client = boto3.client('s3',
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY)
s3_bucket = s3_resource.Bucket(S3_BUCKET)


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


class FlatRingBuffer():
    "Numpy backed buffer that stores data as one flat array"
    def __init__(self, capacity, dtype=np.float32, initial_value=0):
        self.capacity = int(capacity)
        self._idx = 0
        self.full = False
        self._values = np.full(self.capacity, initial_value, dtype=dtype)

    def _add_vector(self, vector):
        #vector cant be directly copied into buffer
        if len(vector) + self._idx > self.capacity:
            self.full = True
            self._values = np.concatenate((self._values, vector))
            self._values = self._values[-self.capacity:]
            self._idx = min(len(vector) + self._idx, self.capacity - 1)

        #copy vector directly into buffer
        else:
            self._values[self._idx: self._idx + len(vector)] = vector
            #ensure index doesnt overflow capacity
            self._idx = min(len(vector) + self._idx, self.capacity - 1)

            if self._idx >= self.capacity -1:
                self.full = True


    def _add_scalar(self, scalar):
        if self.full:
            self._values = np.roll(self._values, -1)
            self._values[self._idx-1] = scalar

        else:
            self._values[self._idx] = scalar
            self._idx += 1
            if (self._idx) == self.capacity:
                self.full = True

    def add(self, v):
        if np.isscalar(v):
            self._add_scalar(v)
        else:
            self._add_vector(v)

    def append(self, v):
        self.add(v)

    def values(self):
        return self._values[0:self._idx]

    def size(self):
        return self._idx + 1


def datetime_utc_string(dt):
    return dt.astimezone(pytz.utc).strftime(S3_TIME_FORMAT)


def datetime_to_s3_time_bucket(dt):
    "Generates the 10-second timebucket we use on s3"
    formatted_str = dt.strftime(S3_TIME_FORMAT)
    bucket = formatted_str[:-9] + "0"
    return bucket


def parse_time_string_with_colon_offset(s):
    "parses timestamp string with colon in UTC offset ex: -6:00"
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    utc = local.astimezone(pytz.utc)
    return utc


def timebucket_files(sensor_id, timebucket):
    "lists files in a sensor's timebucket"
    prefix = f"{sensor_id}/{timebucket}"
    return sorted([obj.key for obj in s3_bucket.objects.filter(Prefix=prefix)])


def interval_to_buckets(start, end):
    "generates timebucket strings from datetime objects"
    delta = end - start

    #ensure all buckets are found when interval overlaps buckets and less than 10 seconds
    if delta.seconds < 10:
        increment = delta.seconds
    else:
        increment = 10

    timebuckets = []

    current_time = start
    while current_time <= end:
        timebucket = datetime_to_s3_time_bucket(current_time)
        timebuckets.append(timebucket)
        current_time += timedelta(seconds=increment)
    return sorted(set(timebuckets))


def get_npz(key):
    "Returns numpy array from npz on s3"
    with io.BytesIO() as f:
        s3_bucket.download_fileobj(key, f)
        f.seek(0)
        data = np.load(f)['arr_0']
        return data


def s3_path_to_datetime(path):
    "Converts s3 npz path to python datetime object"
    filename = path.split('/')[-1]
    t = datetime.strptime(filename, NPZ_TIME_FORMAT)
    t_utc = pytz.utc.localize(t)
    return t_utc


def interval_to_flat_array(sensor_id, start, end, sample_rate=57000):
    "Returns all dynamic data for a sensor_id, start, and end time in a ring buffer"
    assert end > start
    start = start.astimezone(pytz.utc).replace(microsecond=0)
    end = end.astimezone(pytz.utc).replace(microsecond=0)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        t = t.replace(microsecond=0)
        return t >= start and t < end

    timebuckets = interval_to_buckets(start,end)


    num_seconds = (end - start).seconds
    num_samples = sample_rate * num_seconds
    buffer = FlatRingBuffer(num_samples)

    t0 = time.time()

    with mp.Pool(64) as pool:
        files = pool.starmap(timebucket_files, [(sensor_id, timebucket) for timebucket in timebuckets])
        files = [val for sublist in files for val in sublist]
        files_within_interval = [f for f in files if within_interval(f)]
        print("Downloading ", len(files_within_interval), " files")
        data = pool.map(get_npz, files)
    print("Files downloaded in ", time.time() - t0)

    for d in data:
        buffer.add(d)

    return buffer


def interval_to_flat_array_threaded(sensor_id, start, end, sample_rate=57000):
    "Returns all dynamic data for a sensor_id, start, and end time in a ring buffer"
    import concurrent.futures
    assert end > start
    start = start.astimezone(pytz.utc).replace(microsecond=0)
    end = end.astimezone(pytz.utc).replace(microsecond=0)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        t = t.replace(microsecond=0)
        return t >= start and t < end

    timebuckets = interval_to_buckets(start,end)


    num_seconds = (end - start).seconds
    num_samples = sample_rate * num_seconds
    buffer = FlatRingBuffer(num_samples)

    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        #args = [(sensor_id, timebucket) for timebucket in timebuckets]
        sensor_ids = [sensor_id for i in range(len(timebuckets))]
        files = executor.map(timebucket_files, sensor_ids, timebuckets)
        files = [val for sublist in files for val in sublist]
        files_within_interval = [f for f in files if within_interval(f)]
        print("Downloading ", len(files_within_interval), " files")
        data = executor.map(get_npz, files)


    print("Files downloaded in ", time.time() - t0)

    for d in data:
        buffer.add(d)

    return buffer


def interval_to_flat_array_single(sensor_id, start, end, sample_rate=57000):
    "Returns all dynamic data for a sensor_id, start, and end time in a ring buffer"
    assert end > start
    start = start.astimezone(pytz.utc).replace(microsecond=0)
    end = end.astimezone(pytz.utc).replace(microsecond=0)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        t = t.replace(microsecond=0)
        return t >= start and t < end

    timebuckets = interval_to_buckets(start,end)


    num_seconds = (end - start).seconds
    num_samples = sample_rate * num_seconds
    buffer = FlatRingBuffer(num_samples)

    t0 = time.time()

    files = [timebucket_files(sensor_id, timebucket) for timebucket in timebuckets]
    files = [val for sublist in files for val in sublist]
    files_within_interval = [f for f in files if within_interval(f)]
    data = [get_npz(f) for f in files]
    print("Files downloaded in ", time.time() - t0)

    for d in data:
        buffer.add(d)

    return buffer

#not used?
class DataPuller():
    def __init__(self, buffer_capacity=60*57000, delay=0):
        self.delay = delay
        self.latest = (datetime.now() - timedelta(seconds=delay)).astimezone(pytz.utc)
        self.buffer = FlatRingBuffer(buffer_capacity)


    def get_latest_file(self, sensor_id, timebucket):
        "Gets latest file in timebucket"
        prefix = f"{sensor_id}/{timebucket}"
        files = s3_bucket.objects.filter(Prefix=prefix)
        files = [obj for obj in sorted(files, key=lambda x: x.last_modified, reverse=True)]

        if len(files) > 0:
            return files[0].key
        else:
            return None


    #check current timebucket then check prior if no file
    def add_latest_to_buffer(self, sensor_id):
        now = datetime.now().astimezone(pytz.utc)
        current_bucket = datetime_to_s3_time_bucket(now)
        previous_bucket = datetime_to_s3_time_bucket(now - timedelta(seconds=10))

        for bucket in [current_bucket, previous_bucket]:
            latest_file = self.get_latest_file(sensor_id, bucket)
            if latest_file:
                filetime = s3_path_to_datetime(latest_file)
                if filetime > self.latest:
                    #print(latest_file)
                    #data = load_s3_npz(f"{S3_BUCKET}/{latest_file}")
                    data = get_npz(s3_bucket, latest_file)
                    self.buffer.append(data)
                    self.latest = filetime
                    break


    def get_next_file(self, sensor_id):
        current_bucket = datetime_to_s3_time_bucket(self.latest)
        next_bucket = datetime_to_s3_time_bucket(self.latest +  timedelta(seconds=10))
        for bucket in [current_bucket, next_bucket]:
            files = sorted(timebucket_files(sensor_id, bucket))
            for f in files:
                filetime = s3_path_to_datetime(f)
                if filetime > self.latest:
                    data = get_npz(f)
                    self.buffer.append(data)
                    self.latest = filetime
                    break

class ModelData():
    """
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
                 initialization_time = None,):
        self.frequency = frequency
        self.window_size = window_size
        self.dynamic_id = dynamic_id
        self.static_id = static_id
        self.max_sample_rate = max_sample_rate
        self.dynamic_buffer = FlatRingBuffer(window_size * max_sample_rate)
        self.static_buffer = deque(maxlen=window_size)
        self.end = initialization_time
        self.initialized = False


        if not self.end:
            self.end = datetime.now().astimezone(pytz.utc).replace(microsecond=0)

        self.start = self.end - timedelta(seconds=self.window_size)


    def increment_interval(self):
        assert self.start and self.end

        delta = timedelta(seconds=self.frequency)
        #update ends
        self.start += delta
        self.end += delta


    def interval_to_fetch(self):
        if not self.initialized:
            start = self.start
            end = self.end

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
        new_data = interval_to_flat_array_threaded(self.dynamic_id,
                                          start,
                                          end,
                                          sample_rate=self.max_sample_rate).values()
        self.dynamic_buffer.add(new_data)


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


    def update_data(self):
        start, end = self.interval_to_fetch()
        if self.dynamic_id:
            self.update_dynamic_data()

        if self.static_id:
            self.update_static_data()

        #start,end already set if not initialized
        if self.initialized:
            self.increment_interval()

        self.initialized = True


    def dynamic_data(self):
        return self.dynamic_buffer.values()

    def static_data(self):
        return list(self.static_buffer)

"""
md = ModelData(5,
               60,
               dynamic_id="a61b91c7-de10-4f45-9cdc-fb744eece46e",
               static_id="e5ac7795-a213-4488-a67b-36ed40c70d32",
               initialization_time= datetime.now().astimezone(pytz.utc) - timedelta(seconds=3600)
)
"""

class AbstractModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def infer(self, dynamic_data=None, static_data=None):
        """Takes in data and returns a list of detected events
            [{"event": "stage start",
              "start_idx": 57000*2,
              "end_idx": 57000*20,
             "tags": ["tag1", "tag2"]}]"""
        pass


class StageModel(AbstractModel):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def infer(self, dynamic_data=None, static_data=None):
        print("inferring", dynamic_data[-1], static_data[-1]['time'])
        events = [{"event": "stage start",
                 "start_idx": 57000*2,
                 "end_idx": 57000*20,
                 "tags": ["tag1", "tag2"]}]
        return events


class FracScore(AbstractModel):
    def __init__(self, num_seconds):
        self.seconds = num_seconds

    def infer(self, dynamic_data, static_data):
        import fracs
        #result = fracs.fracScore(dynamic_data, static_data, self.seconds)
        #print(np.array(result[1]).sum())
        #for r in result:
            #print(np.array(r).sum())
        result = [{"event": "stage start",
                 "start_idx": 57000*2,
                 "end_idx": 57000*20,
                 "tags": ["tag1", "tag2"]}]
        return result



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


class ModelRunner():
    """
    Everything needed to run model on data

    Parameters:
    ----------
    model : Class that implements AbstractModel - pass data to this and get events back
    config : Dict defining period and window size to run model on
    delay : seconds in past to initialize model on - needed because delay uploading data to s3
    """
    def __init__(self, config, model, initialization_time=None):


        self.frequency = config["frequency"]
        self.api = config["api"]
        self.window_size = config["window_size"]
        self.job_id = config["id"]
        self.monitoring_group_id = config['monitoring_group_id']
        self.model = model
        #self.delay = delay

        sensors = well_sensors(self.api)
        self.static_id = sensors['static']
        self.dynamic_id = sensors['dynamic']

        if not initialization_time:
            initialization_time =  datetime.now().astimezone(pytz.utc).replace(microsecond=0)

        #data_init_time = datetime.now().astimezone(pytz.utc).replace(microsecond=0) - timedelta(seconds=self.delay)
        self.model_data = ModelData(self.frequency,
                                    self.window_size,
                                    dynamic_id=self.dynamic_id,
                                    static_id=self.static_id,
                                    initialization_time=initialization_time)


    def calculate_result_times(self, model_results):
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
        results = self.model.infer(dynamic_data, static_data)
        """
        if results:
            for result in results:
                start, end = self.calculate_result_times(result)
                result['start_time'] = start
                result['end_time'] = end
            return results
        """
        return results

    def update_data_and_infer(self):
        t0 = time.time()
        self.model_data.update_data()
        results = self.infer()
        print("update infer time", time.time() - t0)
        return results

    def detect_and_insert_events(self):
        results = self.update_data_and_infer()
        q = """INSERT INTO monitoring.events
                 (id, api, event, start_time, end_time)
                 VALUES (%s, %s, %s, %s, %s)"""

        sample_args = (str(uuid.uuid4()),
                       self.api,
                       "test event",
                       datetime.now().astimezone(pytz.utc),
                       datetime.now().astimezone(pytz.utc) - timedelta(seconds=60))

        db_query(q, sample_args, fetch_results=False)
        db_query("""UPDATE monitoring.model_queue
                    SET last_update=%s
                    WHERE id=%s""",
                 (datetime.now().astimezone(pytz.utc), self.job_id), fetch_results=False)


        """
        if results:
            event_id = str(uuid.uuid4())
            args = (event_id,
                    self.api,
                    results.get('event'),
                    results.get('start_time'),
                    results.get('end_time'))
            db_query(q, args)

        """
        return True


# init - pull window and infer
# periodically update data and infer

sample_model_config = {'frequency': 8, 'window': 30}


# doesnt persist ModelRunner state
def run_parallel(job_func):
    p = mp.Process(target=job_func)
    p.start()



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


def run_threaded(job_func):
    """Used to run jobs in separate thread so jobs do not block the main thread enusring jobs are launched on time
       https://schedule.readthedocs.io/en/stable/faq.html#how-to-execute-jobs-in-parallel"""
    job_thread = threading.Thread(target=job_func, daemon=True)
    job_thread.start()


def model_runner_process(runnable_model):
    """To be ran in a separate process.
       Runs model peridoically until monitoring_group marked complete.
       Runs model in seperate thread from process main thread to prevent blocking the scheduler
       https://schedule.readthedocs.io/en/stable/faq.html#how-to-execute-jobs-in-parallel"""


    frequency = runnable_model.frequency
    print("frequency", runnable_model.frequency )
    job = schedule.every(frequency).seconds.do(run_threaded, runnable_model.detect_and_insert_events)
    #job.run()


    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        group_status = "ACTIVE"
        while group_status != "COMPLETE":
            f = executor.submit(monitoring_group_status, runnable_model.monitoring_group_id)
            schedule.run_pending()
            time.sleep(0.25)
            group_status = f.result()


    db_query("""UPDATE monitoring.model_queue
                SET status='FINISHED'
                WHERE id=%s""", (runnable_model.job_id,), fetch_results=False)

    print("group finished")
    schedule.clear()


def scheduler_parallel():
    try:
        api = "42479442680000"
        #models_with_configs = [(sample_model_config, StageModel(1,2), api)]
        sample_setup = (sample_model_config, FracScore(30), api)
        models_with_configs = [sample_setup for i in range(3)]
        #models_with_configs = [(sample_model_config, FracScore(600), api)]
        runnable_models = []

        init_time = parse_time_string_with_colon_offset("2020-02-08T06:45:00-05:00")

        for config, model, api in models_with_configs:
            runnable_models.append(ModelRunner(model, config, api, initialization_time=init_time))

        processes =  []
        for runnable_model in runnable_models:
            p = mp.Process(target=launch_scheduler_process, args=(runnable_model,))
            p.start()
            processes.append(p)

    except KeyboardInterrupt as e:
        for p in processes:
            p.kill()


#sample_queue_info = {'frequency': 2, 'window': 30, 'api': "42479442680000"}
#TEST_QUEUE = deque([sample_queue_info for i in range(6)])


def pull_next_job():
    q = """UPDATE monitoring.model_queue
           SET status = 'RUNNING'
           WHERE id = (
               SELECT id
               FROM monitoring.model_queue
               WHERE status = 'PENDING'
               FOR UPDATE SKIP LOCKED
               LIMIT 1
               )
       RETURNING * """

    result = db_query(q)
    if len(result) > 0:
        return result[0]
    else:
        return None

example_job_config = {'model_name': 'frac score',
                      'frequency': 1,
                      'window_size': 30,
                      'api': "42283351490000",
                      'monitoring_group_id': 'bae36828-3f04-4a5f-a5f0-b4db8eb225e9',
                      'id': "some job id"}

def scheduler_queue():
    max_jobs = 2
    current_jobs = 0
    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        while True:
            try:
                while current_jobs < max_jobs:
                    #job_info = TEST_QUEUE.pop()

                    job_config = pull_next_job()
                    if job_config:
                        #job_config = example_job_config
                        model_name = job_config['model_name']

                        #init_time = parse_time_string_with_colon_offset("2020-02-08T06:45:00-05:00")
                        init_time = parse_time_string_with_colon_offset("2020-04-20T06:45:00-05:00")

                        runnable_model = ModelRunner(job_config, FracScore(30), initialization_time=init_time)
                        #f = executor.submit(runnable_model.update_data_and_infer)
                        f = executor.submit(model_runner_process, runnable_model)
                        futures.append(f)
                        current_jobs += 1


                done, not_done = concurrent.futures.wait(futures,
                                                         timeout=0.05,
                                                         return_when=concurrent.futures.FIRST_COMPLETED)

                futures = list(not_done)
                current_jobs = len(not_done)
                [print(f.result) for f in done]
                print(len(done), len(not_done))
                time.sleep(0.5)

            except KeyboardInterrupt as e:
                for f in futures:
                    f.cancel()
                print(e)
                break


def scheduler():
    api = "42479442680000"
    models_with_configs = [(sample_model_config, FracScore(30), api)]
    #models_with_configs = [(sample_model_config, StageModel(1,2), api)]
    init_time = parse_time_string_with_colon_offset("2020-02-08T06:45:00-05:00")
    jobs = []
    for config, model, api in models_with_configs:
        runnable_model = ModelRunner(model, config, api, initialization_time=init_time)
        frequency = config['frequency']
        job = schedule.every(frequency).seconds.do(runnable_model.update_data_and_infer)
        jobs.append(job)

    #dont try to run all jobs at once
    for job in jobs:
        job.run()
        time.sleep(0.25)

    #kick off immediately otherwise it waits for 'frequency' seconds before starting jobs
    schedule.run_all()

    while True:
        try:
            schedule.run_pending()
            time.sleep(0.5)

        except KeyboardInterrupt as e:
            print(e)
            schedule.clear()
            break


#stage_model = StageModel(1,2)
#mi = ModelRunner(stage_model, sample_model_config, dynamic_id="a61b91c7-de10-4f45-9cdc-fb744eece46e", delay=60)


def foo():
    sensor_id = "23df1558-d475-4f78-9c43-9702bf2daefd"
    d = DataPuller(60*57000)
    while True:
        try:
            t0 = time.time()
            d.get_next_file(sensor_id)
            print(d.buffer._idx, time.time() - t0)
            time.sleep(0.2)


        except KeyboardInterrupt as e:
            print(e)
            return d


if __name__ == "__main__":
    scheduler_parallel()
