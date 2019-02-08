

"""
"""

import psi4
import numpy as np

class psi4QM:
	def __init__(self, memory, geometry_input, output_file):
		pass
	def output(self):
		psi4.core.set_output_file(str(output_file), False)
		newmol = psi4.geometry("""{}""".format(geometry_input))
		

		
