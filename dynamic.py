import dask
from datetime import datetime
import s3fs
import numpy as np
import os
import time

S3_BUCKET = "sensor-data-live"
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

fs = s3fs.S3FileSystem(key=ACCESS_KEY ,secret=SECRET_KEY)

def datetime_to_s3_time_bucket(dt):
    "Generates the 10-second timebucket we use on s3"
    formatted_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    bucket = formatted_str[:-9] + "0"
    return bucket


def timebucket_files(sensor_id, timebucket):
    return fs.ls(f"{S3_BUCKET}/{sensor_id}/{timebucket}")


def load_s3_npz(s3_path):
    with fs.open(files[0]) as f:
        a = np.load(f)['arr_0']
    return a


class RingBuffer():
    def __init__(self, capacity, dtype=np.float32, initial_value=0):
        self.capacity = int(capacity)
        self._size = 0
        self._idx = 0
        self.full = False
        self._buffer = np.full(self.capacity, initial_value, dtype=dtype)

    def add_vector(self, vector):
        t1 = time.time()

        if len(vector) + self._idx > self.capacity:
            self.full = True
            self._buffer = np.concatenate((self._buffer, vector))
            self._buffer = self._buffer[-self.capacity:]
            self._idx = min(len(vector) + self._idx, self.capacity - 1)

        else:
            self._buffer[self._idx: self._idx + len(vector)] = vector
            #ensure index doesnt overflow capacity
            self._idx = min(len(vector) + self._idx, self.capacity - 1)


        #print(time.time() - t1)


    def add_scalar(self, scalar):
        if self.full:
            self._buffer = np.roll(self._buffer, -1)
            self._buffer[self._idx-1] = scalar

        else:
            self._buffer[self._idx] = scalar
            self._idx += 1
            if (self._idx) == self.capacity:
                print(self.full, "full")
                self.full = True

    def add(self, v):
        if np.isscalar(v):
            self.add_scalar(v)

        else:
            self.add_vector(v)


files  = fs.ls("sensor-data-live/25a22327-918f-4831-8d55-40be6709ff4f/2020-02-18T06:55:50")

"""
b = RingBuffer(len(files) * 6 * 59)

for f in files*6*60:
    d = load_s3_npz(f)
    b.add(d)


data_size = 33e6
b = RingBuffer(data_size * 0.8)
a = np.arange(data_size)
for v in a.reshape((-1, 55000)):
    t1 = time.time()
    b.add(v)
    print(time.time() - t1)
"""





"""
with fs.open(files[0]) as f:
    a = np.load(f)['arr_0']
"""
#"2020-03-25T21:17:11.0946Z"
#'2020-03-25T21:20:20.080016Z'
