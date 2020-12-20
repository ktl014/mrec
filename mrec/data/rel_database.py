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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/')

# Third party imports
import click
import pandas as pd

# Module Level Constants
import mrec.mrec
from mrec.config import Constants as CONST

DB_DIR = "../dataset/external/mrec.db"
SQL_CMD = {
    "create": f"CREATE TABLE mrec_table ("
              f"{CONST._unit_id} TEXT PRIMARY KEY,"
              f"f{CONST.relation} TEXT,"
              f"f{CONST.direction} TEXT,"
              f"f{CONST.sentence} TEXT,"
              f"f{CONST.term1} TEXT,"
              f"f{CONST.term2} TEXT)"
}
SELECT_CMD = ""

logger = logging.getLogger(__name__)

@click.command()
@click.option('--db_path', default=None, help='DB Path to create.')
@click.option('--input_csv', default=None, help='DB Path to create.')

def create_db(db_path, input_csv):
    db_path = db_path if db_path else DB_DIR
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.debug("REMOVED TABLE")

    db = Database(db_path)
    db.cursor.execute(SQL_CMD['create'])
    logger.debug('SUCCESS: TABLE CREATED')
    logger.debug([description for description in db.cursor.description])

    df = pd.read_csv(input_csv)[['_unit_id', 'relation', 'direction', 'sentence', 'term1', 'term2']]
    df.to_sql(name="mrec_table", con=db.conn, if_exists='append', index=False)

    logger.debug(f'SUCCESS: DATA FROM {input_csv} INSERTED')


class Database:
    """ Database instance for CRUD interaction
    """

    def __init__(self, db_path):
        """ construct the database
        :param db_path: path to the database
        """
        self.conn = self.create_connection(db_path)

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

    @property
    def cursor(self):
        logger.debug("Getting cursor...")
        return self.conn.cursor()

    def new_table(self, name, schema):
        """ create a new table with the given schema
        :param name: name of the new table
        :param schema: the schema as a string
        :return: None
        """
        query = "CREATE TABLE " + str(name) + " (" + str(schema) + ");"
        self.execute("create new table", query)

    #CRUD: C--> CREATE
    def create(self, query, data):
        """ create rows in table from the given data
        :param query: the Insert query as a string
        :param data: a list of row tuples to be inserted
        :return: None
        """
        try:
            cur = self.conn.cursor()
            cur.executemany(query, data)
        except:
            print("error in insert operation")
            self.conn.rollback()

    #CRUD: R--> READ
    def read(self, table_name, cols_needed="*", conditions=None):
        """ get all rows, or all rows specified by the query
        :param table_name: name of the table to select from
        :param cols_needed: string with comma separated list of cols needed, defaults to *
        :param conditions: string with conditions
        :return: result table
        """
        if conditions == None:
            query = "SELECT " + cols_needed + " FROM " + table_name
        else:
            query = "SELECT " + cols_needed + " FROM " + table_name + " " + conditions

        try:
            cur = self.conn.cursor()
            cur.execute(query)
            return cur.fetchall()
        except:
            print("error in select operation")
            self.conn.rollback()

    #CRUD: U--> UPDATE
    def update(self, table_name, new_vals, prim_key_id):
        """ update certain values specified by query
        :param table_name: name of th table to update
        :param new_vals: a dict with attributes as keys, and
                         values as values
        :param prim_key_id: key value pair as list of size 2
                         primary key identifier for row to update
        :return: None
        """
        query = "UPDATE " + table_name + " SET "
        for key in new_vals.keys():
            query += str(key) \
                     + " " \
                     + str(new_vals[key]) \
                     + " , " \
 \
                # remove last comma, and space
        query = query[:len(query) - 3]
        query += " WHERE " \
                 + str(prim_key_id[0]) \
                 + " = " \
                 + str(prim_key_id[1]) \
 \
            # execute the query
        self.execute("update", query)

    #CRUD: D--> DELETE
    def delete(self, table_name, prim_key_id):
        """ delete a row from specified table, and prim key value
        :param table_name: name of the table to delete from
        :param prim_key_id: key value pair as list of size 2
                         primary key identifier for row to update
        :return: None
        """
        query = "DELETE FROM " \
                + table_name \
                + " WHERE " \
                + str(prim_key_id[0]) \
                + " = " \
                + str(prim_key_id[1]) \
 \
            # execute the query
        self.execute("delete", query)

if __name__ == '__main__':
    create_db()
