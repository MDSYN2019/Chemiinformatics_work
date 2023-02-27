"""
Python module which return the queries condition of smiles as a pandas table, and then 
which you can choose to represent as a graph 
"""


import sys
import os
import numpy as np
import pandas as pd

from rdkit.Chem import PandasTools
from rdkit import RDConfig

import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel

# try import psycopg2

# use pydantic to ensure that we have proper inputs
import pydantic

try:
    import psycopg2
except:
    print("Error: You need psycopg2 to run this code")


class PostgresConnect:
    """
    Description here
    """

    def __init__(self, database: str, user: str, host: str, password: str) -> None:
        self.database = str(database)
        self.user = str(user)
        self.host = str(host)
        self.password = str(password)
        self.string = "dbname={} user={} host={} password={}".format(
            self.database, self.user, self.host, self.password
        )
        self.conn = psycopg2.connect(self.string)

    def _return(self):
        """ """

        # Working with the emolecules module.

        try:
            conn = psycopg2.connect(
                dbname="emolecules",
                user="sang",
                host="localhost",
                password="silver!!",
                port=5432,
            )
        except:
            print("Did not connect to database")

    def return_setter(self):
        """
        setter function for _return
        """
        pass
