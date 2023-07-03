from typing import Tuple, Any
import sys
import os
import random
import logging

import numpy as np
import pandas as pd
import deepchem as dc
import pandera as pa
from rdkit import RDConfig
from rdkit.Chem import PandasTools

# from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
# use pydantic to ensure that we have proper inputs
from pydantic import BaseModel

try:
    import psycopg2
    from psycopg2 import Error
except:
    print("Error: You need psycopg2 to run this code")


class PostgresConnect:
    """
    interface for interacting with the postgres database
    """

    def __init__(
        self, database: str, user: str, host: str, password: str, port: str
    ) -> None:
        self.user: str = str(user)
        self.host: str = str(host)
        self.password: str = str(password)
        self.database: str = str(database)
        self.port: str = str(port)
        self.input_vals: dict = {
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
        }

        # sql queries that search for substructures within the
        # rdkit database defined
        self.queries: dict[str, str] = {
            "pyridine": "sql/pyridine.sql",
            "benzene": "sql/benzene.sql",
            "ferrocene" "sql/ferrocene.sql",
        }

    def _read_external_sql(self, molecule_query: str, limit: int = 10) -> str:
        """
        Create the SQL strings for the database to search for. At the moment we have pyridine,
        benzne, and ferrocene.
        """
        assert (
            molecule_query in self.queries.keys()
        ), "molecule is not part of any query!"

        molecular_query = self.queries[molecule_query]
        with open(molecular_query, "r") as f:
            molecular_query_string = f.read()

            limit_string = f"LIMIT {limit}"
            full_query = molecular_query_string + " " + limit_string + ";"

        return full_query

    def make_connection(self) -> None:
        """
        try to make connection to the database using
        the input values as stated from the constructor
        """
        self.conn = psycopg2.connect(**self.input_vals)
        self.cursor = self.conn.cursor()
        # except (Exception, Error) as error:
        #    print("Error while connecting to PostgresSQL", error)

    def _return_query(self, molecule_query: str = "pyridine"):
        """
        setter function for _return
        """
        # get the query necessary to extract like molecules in the rdkit
        query = self._read_external_sql(molecule_query)
        if query:
            self.cursor.execute(f"{query}")
        # if default values for query, then query the following query
        else:
            # what is this query trying to do with the smiles database?
            self.cursor.execute(
                "SELECT id, structure FROM molecules WHERE structure@>'c1cccnc1' LIMIT 100;"
            )
        # TODO - explore a number of queries for getting molecules
        record = self.cursor.fetchall()
        return record

    def return_pandas_table(self) -> pd.DataFrame:
        """
        take the postgres output and return it as a panda dataframe. This
        dataframe will also be checked with pandera before outputting
        """
        query_output = self._return_query()
        # return the smiles from the rdkit cartridge
        output_dataframe = pd.DataFrame(query_output, columns=["index", "smiles"])
        # check output using pandera
        schema = pa.DataFrameSchema(
            {
                "index": pa.Column(int),  # could add more conditionals here
                "smiles": pa.Column(str),  # check for valid smiles string properly here
            }
        )
        # validate output here
        schema.validate(output_dataframe)
        return output_dataframe

    def _split(
        self,
        dataset: dc.data.NumpyDataset,
        frac_train: float = 0.6,
        frac_valid: float = 0.2,
        frac_test: float = 0.2,
    ):
        """
        split the machine learning dataset into training, validation
        and testing datasets

        """
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset=dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
        )
        return (train_dataset, valid_dataset, test_dataset)

    def prepare_deepchem_dataset(
        self, dataframe: pd.DataFrame, size: int = 1024
    ) -> None:
        """
        prepare the deepchem dataset, but we need to ensure we have the
        correct featurizer for the smiles molecules. At the moment we

        https://deepchem.readthedocs.io/en/latest/get_started/tutorials.html#data-handling

        """
        ecfp = None
        properties = None
        weights = None

        featurizer = dc.feat.CircularFingerprint(size=size)
        # convols = featurizer.featurize(dataframe['smiles'])
        ecfp = featurizer.featurize(dataframe["smiles"])
        # generate random properties for now) - needs to be of the same length as
        # the length of the smiles dataset

        # This should be changed later to represent a true property for the properties.
        properties = [random.random() for _ in dataframe["smiles"]]
        weights = [random.random() for _ in dataframe["smiles"]]

        self.engineered_features = dc.data.NumpyDataset(
            X=ecfp,
            y=np.array(properties),
            w=weights,
            ids=dataframe["smiles"].to_numpy(),
        )
        # split the dataset
        training_data = self._split(self.engineered_features)
        return training_data

    def close_connection(self) -> None:
        """
        option to close the connection to postgres
        """
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Postgresql connection is closed")


if __name__ == "__main__":
    # running a simple query to access to emolecules database
    # in the rdkit cartridge please ensure that the docker container
    # for this cartridge is running before running
    # this query
    connection = PostgresConnect(
        user="postgres",
        password="mypassword",
        host="127.0.0.1",
        port="5432",
        database="emolecules",
    )
    connection.make_connection()
    # I can either feed this table directly to make a deepchem
    # input or feed this into a
    record = connection.return_pandas_table()
    deepchem_sample = connection.prepare_deepchem_dataset(record) # currently not being used
    print(record)
    connection.close_connection()
