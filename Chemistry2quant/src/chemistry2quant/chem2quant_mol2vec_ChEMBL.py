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
    
aa_smis = ['CC(N)C(=O)O', 'N=C(N)NCCCC(N)C(=O)O', 'NC(=O)CC(N)C(=O)O', 'NC(CC(=O)O)C(=O)O',
          'NC(CS)C(=O)O', 'NC(CCC(=O)O)C(=O)O', 'NC(=O)CCC(N)C(=O)O', 'NCC(=O)O',
          'NC(Cc1cnc[nH]1)C(=O)O', 'CCC(C)C(N)C(=O)O', 'CC(C)CC(N)C(=O)O', 'NCCCCC(N)C(=O)O',
          'CSCCC(N)C(=O)O', 'NC(Cc1ccccc1)C(=O)O', 'O=C(O)C1CCCN1', 'NC(CO)C(=O)O',
          'CC(O)C(N)C(=O)O', 'NC(Cc1c[nH]c2ccccc12)C(=O)O', 'NC(Cc1ccc(O)cc1)C(=O)O',
          'CC(C)C(N)C(=O)O']

aa_codes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# All part of parsing the emolecules dataset
def emolecule_command(sql_command)
    try:                                                                                                                                            
        conn = psycopg2.connect("dbname='emolecules' user='sang' host='localhost' password='Blad1bl@1234'") 
    except:                                                                                                                                         
        print("Did not connect to database")                                                                                                             
    emoleculesCur = conn.cursor()
    emoleculesCur.execute(str(sql_command))                                   
    database = emoleculesCur.fetchall()
    return database

def chembl24_command(sql_command)
    try:
        chembl_24_conn = psycopg2.connect("dbname='chembl_24' user='sang' host='localhost' password='Blad1bl@1234'") 
    except:
        print("Could not connect to chembl_24")    

    chembl_cur = chembl_24_conn.cursor()
    chembl_cur.execute(str(sql_command))
    chembl_database = chembl_cur.fetchall()
    return chembl_database

numpyDatabase = np.asarray(database) # convert to a numpy array to make use of numpy data parsing
smiles = numpyDatabase[:,0] # Isolate the smiles 
mols = [Chem.MolFromSmiles(x) for x in smiles if Chem.MolFromSmiles(x) != None]

