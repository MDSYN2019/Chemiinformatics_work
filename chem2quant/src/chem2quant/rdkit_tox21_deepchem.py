"""

Code taken from as seen from "https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_sklearn_models.py", 

from the deepchem repository

"""

import numpy as np
import deepchem as dc
from deepchem.molnet import load_tox21
from sklearn.ensemble import RandomForestClassifier

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21()
(train_dataset, valid_dataset, test_dataset) = tox21_datasets

