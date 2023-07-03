"""
What are we trying to do with this? 

Storing the rdkit objects we work with in a neo4j graph database.
"""
from typing import Callable, Union, Iterable  # iterable not used yet

# from collections.abc import Iterator, Iterable
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# udf import
from pyspark.sql.functions import col, udf
from graphframes import GraphFrame
import datamol as dm


# could be useful making a abstract function for this here
class udf_function_suite_mol:
    def __init__(self):
        """ """
        pass

    def convert_case(input_str: str):
        """ """
        res_str = " "
        arr = str.split(" ")
        for x in arr:
            res_str = res_str + x[0:1].upper() + x[1 : len(x)] + " "
        return res_str

    def datamol_clean_func(smiles_str: str) -> str:
        """
        preprocessing the smiles strings as following the instructions here:
        https://doc.datamol.io/stable/tutorials/Preprocessing.html

        return sanitized mol from the smiles input
        """
        assert type(smiles_str) == str, "not a string!"
        mol = dm.to_mol(smiles_str)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        return mol


class chem2qunt_lipo:
    def __init__(self, lipo_text: str, spark_app_name: str):
        self.lipo_text: str = "/home/sang/Desktop/git/Chemiinformatics_work/Chemistry2quant/src/zip/lipophilicity.csv"
        # sparksession start (each sparksession has sparkcontext)
        self.spark = SparkSession.builder.appname("demo").getOrCreate()
        self.udf_function_for_chem: Union[Callable, None] = None

    def _udf_function(self, callable_function: Callable) -> Callable:
        """
        convert function call to a udf function that can transform the underlying
        input funciton
        """
        udf_function = udf(callable_function)
        return udf_function

    def load_into_spark() -> None:
        """ """
        self.cases_lipo = spark.read.load(
            lipo_text, format="csv", sep=",", inferSchema=True, header="true"
        )
        self.cases_lipo.cache()  # cache the results
        self.cases_lipo.count()  # store the cached results like this

    def graph_implementation(my_function: Callable) -> None:
        """ """
        pass


class chem2quant_graph:
    def __init__(self, input_file_node: str, input_file_rels: str, spark_app_name: str):
        # initialize sparksession
        self.spark = (
            SparkSession.builder.master("local").appName(spark_app_name).getOrCreate()
        )

        self.input_file_node = input_file_node
        self.input_file_rels = input_file_rels


# The following creates GraphFrame from the example CSV files
def create_transport_graph(input_file_node: str, input_file_rels: str) -> None:
    """ """
    node_fields = [
        StructField("id", StringType(), nullable=True),
        StructField("latitude", StringType(), nullable=True),
        StructField("longitude", FloatType(), nullable=True),
        StructField("population", IntegerType(), nullable=True),
    ]

    # need to check here the nodes structure versus the relationships structure

    nodes = spark.read.csv(input_file_node, header=True, schema=StructType(node_fields))
    relationships = spark.read.csv(
        input_file_rels, header=True
    )  # without the schema, I think spark infers the schema, which is not what we want.

    reversed_rels = (
        rels.withColumn("newSrc", rels.dst)
        .withColumn("newDst", rels.src)
        .drop("dst", "src")
        .withColumnRenamed("newSrc", "src")
        .withColumnRenamed("newDst", "dst")
        .select("src", "dst", "relationship", "cost")
    )

    relationships = rels.union(reversed_rels)
    return GraphFrame(nodes, relationships)


class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(
                self.__uri, auth=(self.__user, self.__pwd)
            )
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = (
                self.__driver.session(database=db)
                if db is not None
                else self.__driver.session()
            )
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response
