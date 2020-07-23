import boto3
import json
import os
import io
import cloudpickle
import pickle
import tensorflow as tf
import tempfile
import numpy as np

S3_BUCKET = "legend-insight-models"
ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

s3_resource = boto3.resource('s3',
                             aws_access_key_id=ACCESS_KEY,
                             aws_secret_access_key=SECRET_KEY)
s3_client = boto3.client('s3',
                         aws_access_key_id=ACCESS_KEY,
                         aws_secret_access_key=SECRET_KEY)
s3_bucket = s3_resource.Bucket(S3_BUCKET)

class TF_Model():
    def __init__(self, keras_model ,metadata):
        self.model = keras_model
        self.static_input_size = metadata.get('static_input_size') or metadata.get('window_size')
        self.dynamic_input_size = metadata.get('dynamic_input_size')
        self.events = metadata['events']
        self.sensors = metadata['sensors']
        self.detection_threshold = metadata['detection_threshold']


    def detect_event(self, data):
        pred = self.model.predict(data.reshape(1,-1,1))
        if np.max(pred) >= self.detection_threshold:
            event_idx = np.argmax(pred)
            event = self.events[event_idx]
            return event


    def infer(self, dynamic_data, static_data):
        if 'static' in self.sensors and len(static_data) >= self.static_input_size:
            data = static_data[-self.static_input_size:]
            data = np.array([x['average'] for x in data])
            return self.detect_event(data)

        if 'dynamic' in self.sensors and len(dynamic_data) >= self.dynamic_input_size:
            data = dynamic_data[-self.dynamic_input_size:]
            return self.detect_event(data)


def list_models(model_type):
    objects = [obj.key for obj in s3_bucket.objects.filter(Prefix=model_type)]
    models = set([obj.split('/')[1] for obj in objects])
    return models


def list_model_versions(model_type, model_name):
    objects = [obj.key for obj in s3_bucket.objects.filter(Prefix=f"{model_type}/{model_name}")]
    versions = set([obj.split('/')[2] for obj in objects])
    return versions


def download_metadata(model_type, model_name, version=None):
    if not version:
        versions = list_model_versions(model_type, model_name)
        version = sorted(versions)[-1]

    metadata_key = f"{model_type}/{model_name}/{version}/metadata.json"
    with io.BytesIO() as f:
        s3_bucket.download_fileobj(metadata_key, f)
        f.seek(0)
        metadata = json.loads(f.read())

    return metadata


def download_py_model(model_name, version=None):
    if not version:
        versions = list_model_versions('py', model_name)
        version = sorted(versions)[-1]

    prefix = f"py/{model_name}/{version}/"
    model_key = prefix + f"{model_name}-{version}"

    metadata = download_metadata('tf', model_name, version)

    #download model
    with io.BytesIO() as f:
        s3_bucket.download_fileobj(model_key, f)
        f.seek(0)
        model = pickle.loads(f.read())

    return model, metadata


def download_tf_model(model_name, version=None):
    if not version:
        versions = list_model_versions('tf', model_name)
        version = sorted(versions)[-1]

    prefix = f"tf/{model_name}/{version}/"
    model_key = prefix + f"{model_name}-{version}"

    metadata = download_metadata('tf', model_name, version)

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            s3_bucket.download_fileobj(model_key, f)
            f.seek(0)
            keras_model = tf.keras.models.load_model(f.name)

    return keras_model, metadata


def put_py_model(model, name, version, metadata):
    """At minimum metadata needs these keys
       frequency: int
       window_size: int
       sensors: array of 'static' and/or 'dynamic' ex ['static', 'dynamic']
    """
    pickled_model = cloudpickle.dumps(model)
    prefix = f"py/{name}/{version}/"
    model_key = prefix + f"{name}-{version}"
    metadata_key = prefix + "metadata.json"
    s3_bucket.put_object(Key=model_key, Body=pickled_model)
    s3_bucket.put_object(Key=metadata_key, Body = json.dumps(metadata))


def put_tf_model(model_h5, name, version, metadata):
    """Uploads .h5 model to s3
       At minimum metadata needs these keys
       frequency: int
       window_size: int
       sensors: array of 'static' and/or 'dynamic' ex ['static', 'dynamic']
       events: array of event names corresponding to model output
       detection_threshold: float - how high model output must be before we count as actual event detection
       dynamic_input_size: int - if dynamic model is used
    """
    prefix = f"tf/{name}/{version}/"
    model_key = prefix + f"{name}-{version}"
    metadata_key = prefix + "metadata.json"
    s3_client.upload_file(model_h5, S3_BUCKET, model_key)
    s3_bucket.put_object(Key=metadata_key, Body = json.dumps(metadata))
