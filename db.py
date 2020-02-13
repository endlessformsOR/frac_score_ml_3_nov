import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg
from psycopg2.extras import DictCursor, DictConnection

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
    cur = conn.cursor(cursor_factory=DictCursor)
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
    try:
        if args:
            df = pd.read_sql(q, conn, params=args)
        else:
            df = pd.read_sql(q, conn)
        return df
    except Exception as e:
        print(e)
        return e


# Initialize environment automatically on boot
def init():
    global conn
    # TODO: This isn't ideal, but will work for now...
    conn = psql_connection()
