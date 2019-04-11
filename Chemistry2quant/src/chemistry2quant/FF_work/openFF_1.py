
"""

We analyze which parameters are applied to a group of molecules

This example uses get_molecule_parameterIDs, a simple utility fuction similar to label_molecules, but intended for use on 
large datasets. get_molecule_parameterIDs processes a list of molecuels using a 
"""


from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import get_data_filename
from openforcefield.topology import Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils.structure import get_molecule_parameterIDs
