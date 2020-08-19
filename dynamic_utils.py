from datetime import datetime, timedelta
import pytz
import numpy as np
import os
import time
from collections import deque
from multiprocessing import Pool
import logging

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
from scipy import signal


logging.basicConfig(format='%(levelname)s %(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger("schedule").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)


ENV = os.environ.get('LEGEND_ENV', "dev")
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

if ENV == "dev":
    S3_BUCKET = "sensor-data-development"
else:
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


    def append_vector(self, v):
        v = v[:] #be able to append other RingBuffers

        if self._capacity > self._right_index:
            num_fits_in_end = min(len(v), self._capacity - self._right_index)
        else:
            num_fits_in_end = 0
        num_doesnt_fit = len(v) - num_fits_in_end

        #add data that fits in end of buffer
        if num_fits_in_end > 0:
            self._arr[self._right_index : self._right_index + num_fits_in_end] = v[:num_fits_in_end]
        self._right_index += num_fits_in_end


        #if we try to add more data that cant fit at end of buffer wrap concat it and resize buffer
        if num_doesnt_fit > 0:
            self._arr = np.concatenate((self._arr, v[num_fits_in_end:]))
            self._arr = self._arr[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0


    def popn(self, n):
        if len(self) < n:
            raise IndexError("pop from an empty or too small RingBuffer")

        if n == 1:
            res = self.pop()

        else:
            res = self[self._capacity - n : self._capacity]

        self._right_index -= n
        self._fix_indices()
        return res


    def popleftn(self, n):
        if len(self) < n:
            raise IndexError("pop from an empty or too small RingBuffer")

        if n == 1:
            res = self.popleft()

        else:
            res = self[self._left_index: self._left_index + n]


        self._left_index += n
        self._fix_indices()

        return res

    def values(self):
        return np.array(self)


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


def interval_to_flat_array(sensor_id, start, end, target_sample_rate=40000, multiprocessing=True, return_num_files=False):
    """Returns all dynamic data for a sensor_id, start, and end time in a ring buffer
     Resamples data to target_sample_rate"""

    assert end > start
    file_start = start.astimezone(pytz.utc).replace(microsecond=0)

    def within_interval(s3_path):
        t = s3_path_to_datetime(s3_path)
        t = t.replace(microsecond=0)
        return t >= file_start and t <= end

    timebuckets = interval_to_buckets(file_start, end)

    t0 = time.time()

    if multiprocessing:
        with mp.Pool(64) as pool:
            files = pool.starmap(timebucket_files, [(sensor_id, timebucket) for timebucket in timebuckets])
            files = [val for sublist in files for val in sublist]
            files_within_interval = [f for f in files if within_interval(f)]
            logging.debug("Downloading " + str(len(files_within_interval)) +  " files")
            arrays = pool.map(get_npz, files_within_interval)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            sensor_ids = [sensor_id for i in range(len(timebuckets))]
            files = executor.map(timebucket_files, sensor_ids, timebuckets)
            files = [val for sublist in files for val in sublist]
            files_within_interval = [f for f in files if within_interval(f)]
            logging.debug("Downloading " + str(len(files_within_interval)) +  " files")
            arrays = executor.map(get_npz, files_within_interval)

    logging.debug("time to download: " + str(time.time() - t0))

    #Need to handle when requeted start,end dont align with uploaded file times
    last_file_time = s3_path_to_datetime(files_within_interval[-1]).replace(microsecond=0)
    downloaded_data_end_time = last_file_time + timedelta(seconds=1)
    file_end_delta = downloaded_data_end_time - end
    file_start_delta = (start - file_start)

    buffer_size = int(target_sample_rate * (end - start).total_seconds())
    buffer = DynamicRingBuffer(buffer_size)

    for s3_path, array in zip(files_within_interval, arrays):
        resampled = signal.resample(array, target_sample_rate)

        if s3_path == files_within_interval[0]:
            num_dropped_samples = int(file_start_delta.total_seconds() * target_sample_rate)
            data_to_append = resampled[num_dropped_samples:]

        elif s3_path == files_within_interval[-1]:
            end_delta = (end - last_file_time)
            num_dropped_samples = int(file_end_delta.total_seconds() * target_sample_rate)
            data_to_append = resampled[0: target_sample_rate - num_dropped_samples]

        else:
            data_to_append = resampled

        buffer.append(data_to_append)

    if return_num_files:
        return buffer, len(arrays)
    else:
        return buffer
