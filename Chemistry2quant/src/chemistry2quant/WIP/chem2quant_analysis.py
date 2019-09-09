"""@package docstring 
Documentation for this module

More details here 

"""

smifile = "/home/oohnohnoh1/Desktop/GIT/Chemiinformatics_work/Chemistry2quant/src/chemistry2quant/smifiles/gdb11_size08.smi"


## Misc modules and numpy/pandas

from functools import reduce
from multiprocessing import Pool, cpu_count
import sys, os, argparse
import numpy as np 
import pandas as pd 

## RDKit Modules -- Chemiinformatics 

import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols # Fingerprinting
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs

## Mol2Vec modules

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from gensim.models import word2vec

## Tensorflow

import tensorflow as tf

## Sklearn modules

from sklearn.decomposition import PCA

## A neural network layer for use later 
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


class rdkitProcessDf:
	"""
	What does this do?
	"""
	## The constructor for the class
	def __init__(self, directory, sdf_file_name):
		self.rdkit_directory = str(directory)
		self.lig_data = self.rdkit_directory + "/" + sdf_file_name
		self.dataMol = sdfToMol(self.lig_data)
	## Return the data of the molecule in the Mol format (more details and links here: PLACEHOLDEr)
	def returnMol(self):
		return self.dataMol

	def MoltoSmiles(self):
		self.ms_smiles = [Chem.MolToSmiles(x) for x in self.dataMol]
		return self.ms_smiles
	
	def MACCSfingerprintList(self):
		self.MACCSlist = [MACCSkeys.GenMACCSKeys(x) for x in self.dataMol]
		return MACCSlist
	def torsionalfingerprintList(self):
		self.Pairslist = [Pairs.GetAtomPairFingerprint(x) for x in self.dataMol]
		return self.Pairslist

class rdkitPsi4DataGenerator:           
	"""
	Here, we want to translate the smilestoMol file into a psi4 file and run DFT calculations for each.
	Based on the code seen in "https://iwatobipen.wordpress.com/2018/08/24/calculate-homo-and-lumo-with-psi4-rdkit-psi4/"
	"""
	def __init__(molfile):
		"""
		What does this function do?
		"""
		self.molfile = molfile
	def molToPsi4(self):
		"""
		What does this function do?
		"""
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
		"""
		What does this function do?
		"""   
		SimSimilarityMaps.GetAPFingerprint(mol, fpType = 'normal')
	def storeMolecule():
		"""
		What does this function do?
		"""    
		pass

"""
Test running
"""
directory = "/home/oohnohnoh1/Desktop/GIT/Chemiinformatics_work/Chemistry2quant/src/chemistry2quant/WIP"
sdf_file = 'bzr.sdf'

process = rdkitProcessDf(directory, sdf_file) # Initialization of the class that reads the sdf file
molList = process.returnMol()
molSmiles = process.MoltoSmiles()
mol2VecList = [mol2alt_sentence(x,1) for x in molList] # Using mol2vec to encode molecules as sentences, meaning that each substructure

# Defining the number of hidden layers and the number of nodes inside them

n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 100

"""
---------------------------------------------
| Fingerprinting and Molecular Similarity   |
---------------------------------------------

The RDkit has a variety of built-in functionality for generating fingerprints
and using them to calculate molecular similarity. The RDKit has a variety for 
generating molecular fingerprints and using them to calculate molecular similarity

"""

