"""
Filename: rel_database
Description: CRUD functions to interact with a database
USAGE
-----
$ python mrec/data/rel_database.py

"""
# imports
import logging
import os
import sqlite3
from sqlite3 import Error
import sys
from pathlib import Path
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third party imports

# Project Level Imports
import mrec.mrec

# Module Level Constants

SQL_CreateTable = '''CREATE TABLE IF NOT EXISTS mrec_table (
                 _unit_id TEXT,
                 relation TEXT,
                 sentence TEXT,
                 direction TEXT,
                 term1 TEXT,
                 term2 TEXT
                 )'''

logger = logging.getLogger(__name__)

def create_db(db_path: str, validation_csv: str, test_csv: str):
    """ Create database

    Database is created by first loading the csv dataset files, then importing the
    data into a created sqllite database.

    Args:
        db_path: Absolute path for output database.
        validation_csv: Absolute path to validation set csv file.
        test_csv: Absoltue path to test set csv file.

    Returns:

    """
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
    logger.info('SUCCESS: TABLE CREATED')

    cols = ['_unit_id', 'relation', 'sentence', 'direction', 'term1', 'term2']
    df.validation[cols].to_sql('mrec_table', con=db.engine, if_exists='append', index=False)
    df.test[cols].to_sql('mrec_table', con=db.engine, if_exists='append', index=False)

    logger.info('SUCCESS: INSERTED DATA')
    db.close_connection()

class Database:
    """ Database instance for CRUD interaction
    """

    def __init__(self, db_path: str):
        """ Initializes a Database

        A sqllite database is created given the database path. Connections and cursors are access via attributes.

        Args:
            db_path:
        """

        Path(db_path).touch()
        self.conn = self._create_connection(db_path)
        self.engine = create_engine('sqlite:///' + db_path, echo=False)
        self.cursor = self.conn.cursor()


    def _create_connection(self, db_path: str):
        """ Create a db connection to database

        Args:
            db_path: Absolute path to database

        Returns:
            conn: Sqlite3 connection. Default is none if error is raised
        """

        try:
            conn = sqlite3.connect(db_path)
            logger.debug('SUCCESS: Table Connected')
            return conn
        except Error as e:
            logger.debug(e)

        return None

    def close_connection(self):
        """ Close the connection and engine

        Returns:

        """
        if self.conn != None:
            self.conn.close()

        if self.engine != None:
            self.engine.dispose()

if __name__ == "__main__":
    assert os.getcwd() == str(Path(__file__).resolve().parents[2]), "Script must be ran from `mrec` project directory"

    db_path = 'dataset/external/mrec.db'
    validation_csv = 'dataset/raw/validation.csv'
    test_csv = 'dataset/raw/test.csv'

    create_db(db_path, validation_csv, test_csv)
