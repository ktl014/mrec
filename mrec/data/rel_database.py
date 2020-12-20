"""
Filename: rel_database
Description: CRUD functions to interact with a database
USAGE
-----
$ cd mrec/data
$ python rel_database.py --db_path=../../dataset/external/mrec.db --input_csv=../../dataset/raw/validation.csv


"""

# imports
import logging
import os
import sqlite3
from pathlib import Path
from sqlite3 import Error
import sys
from pathlib import Path
from mrec.data.dataset import load_data
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third party imports
import click
import pandas as pd

# Module Level Constants
import mrec.mrec
from mrec.config import Constants as CONST


def create_db():
    path = '../../dataset/external/mrec_new_test_1.db'
    csv_fnames = {'validation': '../../dataset/raw/validation.csv'}
    df = load_data(csv_fnames)

    Path(path).touch()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    SQL_CreateTable = '''CREATE TABLE IF NOT EXISTS test_sample_1 (
                 _unit_id TEXT,
                 relation TEXT,
                 sentence TEXT,
                 direction TEXT,
                 term1 TEXT,
                 term2 TEXT
                 )'''

    cursor.execute(SQL_CreateTable)
    engine = create_engine('sqlite:///../../dataset/external/mrec_new_test_1.db', echo=False)

    df.validation[['_unit_id', 'relation', 'sentence', 'direction', 'term1', 'term2']].to_sql('test_sample_1', con=engine, if_exists='append', index=False)
    cursor.execute("SELECT * FROM test_sample_1")

    myresult = cursor.fetchall()
    print(len(myresult))
    for x in myresult:

        print(x)

create_db()
