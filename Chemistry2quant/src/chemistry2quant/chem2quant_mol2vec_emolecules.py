try:
    import psycopg2
except:
    print("Error: You need psycopg2 to run this code")   
try:
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
    from gensim.models import word2vec
except:
    print("Could not find packages")

# All part of parsing the emolecules dataset
def emolecule_command(sql_command, user, password):
    sql_string = "dbname='emolecules' user={} host='localhost' password={}".format(user, password)
    try:                                                                                                                                            
        conn = psycopg2.connect(sql_string) 
    except:                                                                                                                                         
        print("Did not connect to database")
    emolecules_cur = conn.cursor()
    emolecules_cur.execute(str(sql_command))                                   
    database = emolecules_cur.fetchall()
    return database

data_100 = emolecule_command("SELECT * FROM raw_data LIMIT 100;", "sang", "silver!!")

