import psi4
#! Sample UHF/6-31G** CH2 Computation

R = 1.075
A = 133.93

ch2 = psi4.geometry("""
0 3
C
H 1 {0}
H 1 {0} 2 {1}
""".format(R, A)
					)

psi4.set_options({'reference': 'uhf'})
psi4.energy('scf/6-31g**')
