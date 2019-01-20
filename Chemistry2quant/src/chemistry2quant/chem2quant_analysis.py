from __future__ import print_function

"""

Multiprocessing modules


"""
from functools import reduce
from multiprocessing import Pool, cpu_count
import sys, os, argparse
import numpy as np 
import pandas as pd 

#RDKit modules 

import rdkit
from rdkit import DataStructs

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols # Fingerprinting
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS
from rdkit.Chem.Fingerprints import FingerprintMols

# Tensorflow
import tensorflow as tf

# Sklearn modules

from sklearn.decomposition import PCA


def sdfToMol(sdf):
    """
    Returns array of mols from sdf
    """
    suppl_data = Chem.SDMolSupplier(sdf)
    ms = [x for x in suppl_data if x is not None] # Filter data from sdf suppl_data 
    return ms

def substructure_search(substruct, struct_array):
    patt = Chem.MolFromSmarts(str(substruct))
    for x in struct_array:
        print(x.HasSubstructMatch(patt))

# RDkit sdf file
#/home/noh/Desktop/Current_work_in_progress/Chemiinformatics/RDKIT/rdkit/Docs/Book/data

class rdkit_processdf:
    def __init__(self,directory,sdf_file_name):
        self.rdkit_directory = str(directory)
        self.lig_data = self.rdkit_directory + "/" + sdf_file_name
        self.dataMol = sdfToMol(self.lig_data)
    def MoltoSmiles(self):
        ms_smiles = [Chem.MolToSmiles(x) for x in self.dataMol]
        return ms_smiles

class RDKit_data_analysis:           
    """
    Here, we want to translate the smilestoMol file into a psi4 file and run DFT calculations for each.
    Based on the code seen in "https://iwatobipen.wordpress.com/2018/08/24/calculate-homo-and-lumo-with-psi4-rdkit-psi4/"
    """
    def __init__(molfile):
        self.molfile = molfile
    def molToPsi4(self):
        mol = self.molfile
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
        AllChem.UFFOptimizeMolecule(mol)
        atoms = mol.GetAtoms()
        string = string = "\n"
        for i, atom in enumerate(atoms):
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
        string += "units angstrom\n"
        return string, mol
    def forEachSimilarity(ref, array):
        SimSimilarityMaps.GetAPFingerprint(mol, fpType = 'normal')
    def storeMolecule():
        pass

"""

Test running

"""
directory = "/home/noh/Desktop/Current_work_in_progress/Chemiinformatics/RDKIT/rdkit/Docs/Book/data"
sdf_file = 'bzr.sdf'
process = rdkit_processdf(directory, sdf_file)
ms_smiles = process.MoltoSmiles()

# Defining the number of hidden layers and the number of nodes inside them

n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 100

X = tf.placeholder(tf.float32, shape  = (None, n_inputs), name = "X")
is_training = tf.placeholder(tf.bool, shape = (), name = "is_training")        

def neuron_layer(X, n_neurons, name, activation = None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X,W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

class multiprocessing_rdkit:
    def __init__(self,num):
        self.num = num
    def multiprocess_new():
        with Pool(processes = self.num) as pool:
            pass
        
    
for x in ms_smiles:
    ind_map = {}
    m = Chem.MolFromSmiles(x)
    for atom in m.GetAtoms() :
        map_num = atom.GetAtomMapNum()
        if map_num:
            ind_map[map_num-1] = atom.GetIdx()
    print (ind_map,x)
        

    
"""
Check the similarity between the molecules

-- Fingerprinting and Molecular Similarity --

The RDkit has a variety of built-in functionality for generating fingerprints
and using them to calculate molecular similarity.

The RDKit has a variety for generating molecular fingerprints and using them to calculate molecular similarity

"""
fps = [FingerprintMols.FingerprintMol(x) for x in ms]
for index in range(0, len(fps)):
    print(DataStructs.FingerprintSimilarity(fps[0],fps[index]))
    
scaffold_ms = [MurckoScaffold.GetScaffoldForMol(x) for x in suppl_data if x is not None]

# Finding the common substructures between the molecules
res=rdFMCS.FindMCS(ms)
print(res.smartsString)

feature_colummns = tf.contrib.learn_infer_real_valued_columns_from_input(X_train)

#dnn_clf = tf.contrib.DNNClassifier(hidden_units = [300, 100], n_classes =)
# I need to find out how many classes there are for my molecules 

"""
DNN part 
"""

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation = "relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation = "relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")
    
init = tf.global_variables_initalizer()
saver = tf.train.Saver()

