import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg

LEGEND_LIVE_DIR="legend-live"

conn = None

def psql_connection():
    return_conn = pg.connect(
        dbname=os.environ['db_name'],
        user=os.environ['db_user'],
        password=os.environ['db_password'],
        host=os.environ['db_host'],
        port=os.environ['db_port']
    )
    return_conn.set_session(autocommit = True)
    cur=return_conn.cursor()
    cur.execute("SET application_name = 'analytics'")
    return return_conn


def query(q, args=None):
    global conn
    #cur = conn.cursor(cursor_factory=DictCursor)
    cur = conn.cursor(cursor_factory = pg.extras.RealDictCursor)

    try:
        if args:
            cur.execute(q, args)
        else:
            cur.execute(q)
        res = cur.fetchall()
        cur.close()
        return res
    except Exception as e:
        conn.rollback()
        print(e)
        return e


def query_dataframe(q, args=None):
    global conn
    try:
        if args:
            df = pd.read_sql(q, conn, params=args)
        else:
            df = pd.read_sql(q, conn)
        return df
    except Exception as e:
        print(e)
        return e


def init():
    global conn
    # TODO: This isn't ideal, but will work for now...
    conn = psql_connection()


def table_cols(table, schema='public'):
    q = f"""
    SELECT
    pg_attribute.attname AS column_name,
    pg_catalog.format_type(pg_attribute.atttypid, pg_attribute.atttypmod) AS data_type
FROM
    pg_catalog.pg_attribute
INNER JOIN
    pg_catalog.pg_class ON pg_class.oid = pg_attribute.attrelid
INNER JOIN
    pg_catalog.pg_namespace ON pg_namespace.oid = pg_class.relnamespace
WHERE
    pg_attribute.attnum > 0
    AND NOT pg_attribute.attisdropped
    AND pg_namespace.nspname = '{schema}'
    AND pg_class.relname = '{table}'
ORDER BY
    attnum ASC;
    """
    return query(q)


def well_by_name(name, number):
    q = f"""
    SELECT * FROM wells
    WHERE wells.name LIKE '%{name}%' AND number = '{number}'"""
    return query(q)[0]


def well_name_to_api(name, number):
    return well_by_name(name, number)['api']


def well_sensors(api, pressure_type=None):
    '''Get all of the sensors for a well by its API.
    The pressure_type can be either 'dynamic' or 'static'.
    '''
    if pressure_type:
        q = f"""
        SELECT sensors.id, sensors.created_at, wells.api, pad_id, serial, config,
        sensor_type_id, sensor_model_id, pressure_type
        FROM wells as wells
        JOIN sensors as sensors ON sensors.well_id = wells.id
        JOIN sensor_models as sensor_models ON sensors.sensor_model_id = sensor_models.id
        WHERE api = '{api}' AND pressure_type = '{pressure_type}'"""
    else:
        q = f"""
        SELECT sensors.id, sensors.created_at, wells.api, pad_id, serial, config,
        sensor_type_id, sensor_model_id, pressure_type
        FROM wells
        JOIN sensors ON sensors.well_id = wells.id
        JOIN sensor_models ON sensors.sensor_model_id = sensor_models.id
        WHERE api = '{api}'"""
    return query(q)


def sensor_data(sensor_id, start_time=None, end_time=None, val='max', period=None):
    '''Get 1-second sensor data, optionally passing start_time, end_time,
    selecting one of 'min', 'max', 'avg'.  Also supports a period argument in
    seconds, which will then aggregate over windows of that many seconds.'''

    where = f"""WHERE sensor_id = '{sensor_id}'"""

    if start_time and end_time:
        where += f""" AND time BETWEEN '{start_time.isoformat()}' AND '{end_time.isoformat()}'"""
    elif start_time:
        where += f""" AND time >= '{start_time.isoformat()}'"""
    elif end_time:
        where += f""" AND time <= '{end_time.isoformat()}'"""

    if period:
        agg_fn = val.upper()
        if agg_fn == 'AVERAGE':
            agg_fn = 'AVG'

        q = f"""
        SELECT time_bucket('{period} seconds', time) as period,
        {agg_fn}({val}) as value
        FROM sensor_data {where}
        GROUP BY period
        ORDER BY period ASC
        """
        df = query_dataframe(q)
        return df.set_index('period')
    else:
        q = f"""
        SELECT {val} as value
        FROM sensor_data {where}
        ORDER BY time ASC
        """
        df = query_dataframe(q)
        return df


def relative_sensor_data_file(sensor_id, start_time, end_time, path, environment='staging'):
    script_path = os.path.abspath("scripts/relative_sensor_data.sh")
    cmd = [script_path, sensor_id, start_time, end_time, environment, path]
    print("cmd: ", ' '.join(cmd))
    process = subprocess.Popen(cmd,
                               #cwd=LEGEND_LIVE_DIR,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("stdout:", stdout)
    print("stderr:", stderr)


def load_data_file(path):
    npy = np.load(path)
    return npy['arr_0']


def relative_sensor_data(sensor_id, start_time, end_time,
                         environment='staging', tmp_path='tmp_data.npz'):
    relative_sensor_data_file(sensor_id, start_time, end_time, tmp_path, environment)
    return load_data_file(tmp_path)


def download_single_sensor(sensor_id, start, end, path, env):
    cmd = ['lein', 'audio', sensor_id, start, end, path, env, 'single-npz']
    print("cmd: ", ' '.join(cmd))
    process = subprocess.Popen(cmd,
                               text=True,
                               cwd=LEGEND_LIVE_DIR,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("stdout:", stdout)
    print("stderr:", stderr)


def download_monitoring_group(group_id, start, end, path, env):
    cmd = ['lein', 'audio', group_id, start, end, path, env, 'monitoring-group']
    print("cmd: ", ' '.join(cmd))
    process = subprocess.Popen(cmd,
                               text=True,
                               cwd=LEGEND_LIVE_DIR,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("stdout:", stdout)
    print("stderr:", stderr)

