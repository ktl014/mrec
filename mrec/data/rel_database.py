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
from sqlite3 import Error
import sys
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third party imports
import click
import pandas as pd

# Module Level Constants
import mrec.mrec
from mrec.config import Constants as CONST

SQL_CreateTable = '''CREATE TABLE IF NOT EXISTS mrec_table (
                 _unit_id TEXT,
                 relation TEXT,
                 sentence TEXT,
                 direction TEXT,
                 term1 TEXT,
                 term2 TEXT
                 )'''

logger = logging.getLogger(__name__)

def create_db(db_path, validation_csv, test_csv):
    from mrec.data.dataset import load_data

    if not os.path.exists(validation_csv):
        logger.warning(f"File {validation_csv} was not found. Current dir: {os.getcwd()}")
        raise FileNotFoundError("Could not initialize validation set because file not found.")

    if not os.path.exists(test_csv):
        logger.warning(f"File {test_csv} was not found. Current dir: {os.getcwd()}")
        raise FileNotFoundError("Could not initialize test set because file not found.")

    csv_fnames = {'validation': validation_csv, 'test': '{}'.format(test_csv)}
    df = load_data(csv_fnames)

    db = Database(db_path)
    db.cursor.execute(SQL_CreateTable)
    logger.debug('SUCCESS: TABLE CREATED')

    cols = ['_unit_id', 'relation', 'sentence', 'direction', 'term1', 'term2']
    df.validation[cols].to_sql('mrec_table', con=db.engine, if_exists='append', index=False)
    df.test[cols].to_sql('mrec_table', con=db.engine, if_exists='append', index=False)

    logger.debug('SUCCESS: INSERTED DATA')
    db.close_connection()

class Database:
    """ Database instance for CRUD interaction
    """

    def __init__(self, db_path):
        """ construct the database
        :param db_path: path to the database
        """
        #if os.path.exists(db_path):
        #    raise FileExistsError(f"File {db_path} is existed!")

        Path(db_path).touch()
        self.conn = self.create_connection(db_path)
        self.engine = create_engine('sqlite:///' + db_path, echo=False)
        self.cursor = self.conn.cursor()


    def create_connection(self, db_path):
        """ create a db connection to database
        :param db_path: database file path
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(db_path)
            print('SUCCESS: Table Connected')
            return conn
        except Error as e:
            print(e)

        return None

    def close_connection(self):
        """ close the connection
        """
        if self.conn != None:
            self.conn.close()

        if self.engine != None:
            self.engine.dispose()

if __name__ == "__main__":
    db_path = 'dataset/external/mrec.db'
    validation_csv = 'dataset/raw/validation.csv'
    test_csv = 'dataset/raw/test.csv'

    create_db(db_path, validation_csv, test_csv)
