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

import csv


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

    def values(self):
        return self._values[0:self._idx]

    def size(self):
        return self._idx + 1


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
    seconds = delta.seconds
    timebuckets = []

    current_time = start
    while current_time < end:
        timebucket = datetime_to_s3_time_bucket(current_time)
        timebuckets.append(timebucket)
        current_time += timedelta(seconds=10)
    return timebuckets


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
    start = start.astimezone(pytz.utc)
    end = end.astimezone(pytz.utc)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        return t >= start and t <= end

    timebuckets = interval_to_buckets(start,end)

    num_seconds = (end - start).seconds
    num_samples = sample_rate * num_seconds
    buffer = FlatRingBuffer(num_samples)

    t0 = time.time()
    print("Downloading files...")
    with mp.Pool(64) as pool:
        files = pool.starmap(timebucket_files, [(sensor_id, timebucket) for timebucket in timebuckets])
        files = [val for sublist in files for val in sublist]
        files_within_interval = [f for f in files if within_interval(f)]
        #files_within_interval = list(filter(lambda f: within_interval(f), files))
        data  = pool.map(get_npz, files)
    print("Files downloaded in ", time.time() - t0)

    for d in data:
        buffer.add(d)

    return buffer


def load_csv(fpath):
    with open(fpath) as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def csv_row_to_dynamic_data(csv, rownum):
    row = csv[rownum]
    start_time = parse_time_string_with_colon_offset(row['START_TIME'])
    end_time = parse_time_string_with_colon_offset(row['END_TIME'])
    sensor_id = row['DYNAMIC_SENSOR_ID']
    return interval_to_flat_array(sensor_id, start_time, end_time)


class DataPuller():
    def __init__(self, buffer_capcity=60, delay=0):
        self.delay = delay
        self.latest = datetime.utcnow() - timedelta(seconds=delay)
        self.buffer = deque(maxlen=buffer_capcity)
        self.s3 = boto3.resource('s3',
                                 aws_access_key_id=ACCESS_KEY,
                                 aws_secret_access_key=SECRET_KEY)
        self.bucket = s3.Bucket(S3_BUCKET)


    def get_latest_file(self, sensor_id, timebucket):
        prefix = f"{sensor_id}/{timebucket}"
        files = self.bucket.objects.filter(Prefix=prefix)
        files = [obj for obj in sorted(files, key=lambda x: x.last_modified, reverse=True)]

        if len(files) > 0:
            print('size', files[0].size)
            return files[0].key
        else:
            return None

    def timebucket_files(self, sensor_id, timebucket):
        prefix = f"{sensor_id}/{timebucket}"
        return sorted([obj.key for obj in self.bucket.objects.filter(Prefix=prefix)])

    #check current timebucket then check prior if no file
    def add_latest_to_buffer(self, sensor_id):
        now = datetime.utcnow()
        current_bucket = datetime_to_s3_time_bucket(now)
        previous_bucket = datetime_to_s3_time_bucket(now - timedelta(seconds=10))

        for bucket in [current_bucket, previous_bucket]:
            print(i)
            latest_file = self.get_latest_file(sensor_id, bucket)
            if latest_file:
                filetime = s3_path_to_datetime(latest_file)
                if filetime > self.latest:
                    print(latest_file)
                    #data = load_s3_npz(f"{S3_BUCKET}/{latest_file}")
                    data = get_npz(self.bucket, latest_file)
                    self.buffer.append(data)
                    self.latest = filetime
                    break
