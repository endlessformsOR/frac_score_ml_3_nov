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

import schedule
from abc import ABC, abstractmethod
import threading
import psycopg2 as pg
import psycopg2.extras
import uuid

from numpy_ringbuffer import RingBuffer



ENV = os.environ.get('LEGEND_ENV', "staging")
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
S3_BUCKET = f"sensor-data-{ENV}"

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


class DynamicRingBuffer(RingBuffer):
    """
    Creates a numpy backed ringbuffer which is able to append entire numpy vectors

    Parameters
    ----------
    capacity: int
	The maximum capacity of the ring buffer
    dtype: data-type, optional
	Desired type of buffer elements. Use a type like (float, 2) to
	produce a buffer with shape (N, 2)
    allow_overwrite: bool
	If false, throw an IndexError when trying to append to an already
	full buffer
    """
    def __init__(self, capacity, dtype=np.float32, allow_overwrite=True):
        super().__init__(capacity, dtype, allow_overwrite)


    def append(self, v):
        if np.isscalar(v):
            RingBuffer.append(self,v)
        else:
            self.append_vector(v)

    """
    def append_vector(self, v):
        v = v[:] #be able to append other RingBuffers

        #vector cant be directly copied into end of buffer nicely
        #if len(v) + len(self) > self._capacity:
        if len(v) > self._capacity - self._right_index:
            #insert overlap
            num_overlap = self._capacity - self._right_index
            self._arr[self._right_index : self._capacity] = v[:num_overlap]

            #concat and reshape excess
            self._arr = np.concatenate((self._arr, v[num_overlap:]))
            self._arr = self._arr[-self._capacity:]

            #reset indices
            self._right_index = self._capacity
            self._left_index = 0


        #copy vector directly into buffer
        else:
            print("Len", len(self._arr), len(self), len(v), self._right_index)
            self._arr[self._right_index: self._right_index + len(v)] = v
            self._right_index += len(v)
            self._fix_indices()
    """

    def append_vector(self, v):
        v = v[:] #be able to append other RingBuffers

        num_fits_in_end = min(len(v), self._capacity - self._right_index)
        num_doesnt_fit = len(v) - num_fits_in_end

        self._arr[self._right_index : self._right_index + num_fits_in_end] = v[:num_fits_in_end]
        self._right_index += num_fits_in_end


        if num_doesnt_fit > 0:
            self._arr = np.concatenate((self._arr, v[num_fits_in_end:]))
            self._arr = self._arr[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0



    def popleftn(self, n):
        if len(self) < n:
            raise IndexError("pop from an empty or too small RingBuffer")

        if n == 1:
            res = self.popleft()

        elif n < self._capacity - self._left_index:
            res = self[self._left_index: self._left_index + n]

        else:
            left = self._arr[self._left_index : min(self._right_index, self._capacity)]
            leftover = int(n - len(left))
            right = self._arr[:leftover]
            res = np.concatenate((left, right))

        self._left_index += n
        self._fix_indices()

        return res

#d = DynamicRingBuffer(4)
#d.append(np.array([1,2]))
#d.append(np.array([3,4,5]))


class FlatRingBuffer():
    "Numpy backed buffer that stores data as one flat array"
    def __init__(self, capacity, dtype=np.float32, initial_value=0):
        self.capacity = int(capacity)
        self._idx = 0
        self.full = False
        self._values = np.full(self.capacity, initial_value, dtype=dtype)
        self._tail = 0
        self._head = 0

    def _add_vector(self, vector):
        #vector cant be directly copied into buffer
        if len(vector) + self._tail > self.capacity:
            self.full = True
            self._values = np.concatenate((self._values, vector))
            self._values = self._values[-self.capacity:]
            self._tail = min(len(vector) + self._tail, self.capacity - 1)

        #copy vector directly into buffer
        else:
            self._values[self._tail: self._tail + len(vector)] = vector
            #ensure index doesnt overflow capacity
            self._tail = min(len(vector) + self._tail, self.capacity - 1)

            if self._tail >= self.capacity -1:
                self.full = True


    def _add_scalar(self, scalar):
        if self.full:
            self._values = np.roll(self._values, -1)
            self._values[self._tail-1] = scalar

        else:
            self._values[self._tail] = scalar
            self._tail += 1
            if (self._tail) == self.capacity:
                self.full = True

    def add(self, v):
        if np.isscalar(v):
            self._add_scalar(v)
        else:
            self._add_vector(v)


    def append(self, v):
        self.add(v)

    def values(self):
        return self._values[0:self._tail]

    def size(self):
        return self._tail + 1





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

"""
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
"""
