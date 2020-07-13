import boto3
import json
import os
import io
import cloudpickle
import pickle
import tensorflow as tf

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
    def __init__(self, h5, metadata):
        self.model = tf.keras.models.load_model(h5)
        self.input_size = metadata['input_size']
        self.events = metadata['events']
        self.uses_static_data = metadata.get('static_data')
        self.uses_dynamic_data = metadata.get('dymamic_data')
        self.detection_threshold = metadata['detection_threshold']


    def detect_event(self, data):
        pred = self.model.predict(data.reshape(1,-1,1))
        print(pred)
        if np.max(pred) >= self.detection_threshold:
            event_idx = np.argmax(pred)
            event = self.events[event_idx]
            return event


    def infer(self, dynamic_data, static_data):
        if self.uses_static_data:
            return self.detect_event(static_data)

        if self.uses_dynamic_data:
            return self.detect_event(dynamic_data)


test_metadata = {'input_size': 901,
                 'events': ['PUMPDOWN_PERFS_PLUGS_START' , 'PUMPDOWN_PERFS_PLUGS_STOP',
                            'PERF_GUN_FIRING' , 'FRAC_STAGE_START' , 'FRAC_STAGE_STOP',
                            'FORMATION_BREAKDOWN', 'PRESSURIZATION STEP', 'GEAR_SHIFT'],
                 'static_data': True,
                 'detection_threshold': 0.5}

m = '/home/dan/Downloads/static_varLength_s.h5'



def list_models(model_type):
    objects = [obj.key for obj in s3_bucket.objects.filter(Prefix=model_type)]
    models = set([obj.split('/')[1] for obj in objects])
    return models


def list_model_versions(model_type, model_name):
    objects = [obj.key for obj in s3_bucket.objects.filter(Prefix=f"{model_type}/{model_name}")]
    versions = set([obj.split('/')[2] for obj in objects])
    return versions


def download_model(model_type, model_name, version=None):
    if not version:
        versions = list_model_versions(model_type, model_name)
        version = sorted(versions)[-1]

    prefix = f"{model_type}/{model_name}/{version}/"
    model_key = prefix + f"{model_name}-{version}"
    metadata_key = prefix + "metadata.json"

    #download metadata
    with io.BytesIO() as f:
        s3_bucket.download_fileobj(metadata_key, f)
        f.seek(0)
        metadata = json.loads(f.read())

    #download model
    with io.BytesIO() as f:
        s3_bucket.download_fileobj(model_key, f)
        f.seek(0)
        if model_type == "py":
            model = pickle.loads(f.read())
        if model_type == "tf":
            model = TF_Model(f, metadata)

    return model, metadata

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
