import sys, math

## CONSTANTS ##

# threshold beyond average of covalent radiii to determine bond cutoff
bond_thresh = 1.2

# conversion from radians to degrees and vice versa
rad2deg = 180.0 / math.pi
deg2rad = 1.0 / rad2deg

# covalent (or ionic) radii by atomic element (Angstroms) from
# "Inorganic Chemistry" 3rd ed, Housecroft, Appendix 6, pgs 1013-1014
cov_rads = {  'H' : 0.37, 'C' : 0.77, 'O' : 0.73, 'N' : 0.75, 'F' : 0.71,
  'P' : 1.10, 'S' : 1.03, 'Cl': 0.99, 'Br': 1.14, 'I' : 1.33, 'He': 0.30,
  'Ne': 0.84, 'Ar': 1.00, 'Li': 1.02, 'Be': 0.27, 'B' : 0.88, 'Na': 1.02,
  'Mg': 0.72, 'Al': 1.30, 'Si': 1.18, 'K' : 1.38, 'Ca': 1.00, 'Sc': 0.75,
  'Ti': 0.86, 'V' : 0.79, 'Cr': 0.73, 'Mn': 0.67, 'Fe': 0.61, 'Co': 0.64,
  'Ni': 0.55, 'Cu': 0.46, 'Zn': 0.60, 'Ga': 1.22, 'Ge': 1.22, 'As': 1.22,
  'Se': 1.17, 'Kr': 1.03, 'X' : 0.00}

# relative atomic masses of elements (in atomic mass units [amu]) from
# "CRC Handbook" 84th ed, ed Lide, pgs 1-12 - 1-14
at_masses = {    'H' : 1.00794, 'C' : 12.0107, 'O' : 15.9994, 'N' : 14.0067,
  'F' : 18.9984, 'P' : 30.9738, 'S' : 32.0650, 'Cl': 35.4530, 'Br': 79.9040,
  'I' : 126.904, 'He': 4.00260, 'Ne': 20.1797, 'Ar': 39.9480, 'Li': 6.94100,
  'Be': 9.01218, 'B' : 10.8110, 'Na': 22.9898, 'Mg': 24.3050, 'Al': 26.9815,
  'Si': 28.0855, 'K' : 39.0983, 'Ca': 40.0780, 'Sc': 44.9559, 'Ti': 47.8670,
  'V' : 50.9415, 'Cr': 51.9961, 'Mn': 54.9380, 'Fe': 55.8450, 'Co': 58.9332,
  'Ni': 58.6934, 'Cu': 63.5460, 'Zn': 65.4090, 'Ga': 69.7230, 'Ge': 72.6400,
  'As': 74.9216, 'Se': 78.9600, 'Kr': 83.7980, 'X' : 0.00000}

## IO FUNCTIONS ##

# read file data into a 2-d array
def get_file_string_array(file_name):
    try:
        file = open(file_name, "r")
    except IOError:
        print('Error: file (%s) not found!\n' % (file_name))
        sys.exit()
    lines = file.readlines()
    file.close()
    array = []
    for line in lines:
        array.append(line.split())
    return array

# read in geometry from xyz file
def get_geom(xyz_file_name):
    xyz_array = get_file_string_array(xyz_file_name)
    n_atoms = int(xyz_array[0][0])
    at_types = ['' for i in range(n_atoms)]
    coords = [[0.0 for j in range(3)] for i in range(n_atoms)]
    for i in range(n_atoms):
        at_types[i] = xyz_array[i+2][0]
        for j in range(3):
            coords[i][j] = float(xyz_array[i+2][j+1])
    geom = [at_types, coords]
    return geom

# input syntax and usage warnings
def get_inputs():
    if (not len(sys.argv) == 2):
        print('Usage: center_of_mass.py XYZ_FILE\n')
        print('  XYZ_FILE: coordinates of target molecule\n')
        sys.exit()
    else:
        xyz_file_name = sys.argv[1]
        return xyz_file_name

# print geometry to screen
def print_geom(geom, comment):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    print('%i\n%s\n' % (n_atoms, comment), end='')
    for i in range(n_atoms):
        print('%-2s' % (at_types[i]), end='')
        for j in range(3):
            print(' %12.6f' % (coords[i][j]), end='')
        print('\n', end='')
    print('\n', end='')

# print bond graph to screen
def print_bond_graph(geom, bond_graph, comment):
    at_types = geom[0]
    n_atoms = len(at_types)
    print('%s\n' % (comment), end='')
    for i in range(n_atoms):
        print(' %4i %-2s -' % (i+1, at_types[i]), end='')
        for j in range(len(bond_graph[i])):
            print(' %i' % (bond_graph[i][j] + 1), end='')
        print('\n', end='')
    print('\n', end='')
    
# print list of bond lengths to screen
def print_bonds(geom, bonds):
    at_types = geom[0]
    n_bonds = len(bonds)
    print('%i bond(s) found (Angstrom)' % (n_bonds))
    for q in range(n_bonds):
        n1, n2  = bonds[q][0:2]
        r12 = bonds[q][2]
        nstr = '%i-%i' % (n1+1, n2+1)
        tstr = '(%s-%s) ' % (at_types[n1], at_types[n2])
        print(' %-15s  %-13s    %6.4f\n' % (nstr, tstr, r12), end='')
    print('\n', end='')
    
# print list of bond angles to screen
def print_angles(geom, angles):
    at_types = geom[0]
    n_angles = len(angles)
    print('%i angle(s) found (degrees)' % (n_angles))
    for q in range(n_angles):
        n1, n2, n3 = angles[q][0:3]
        a123 = angles[q][3]
        nstr = '%i-%i-%i' % (n1+1, n2+1, n3+1)
        tstr = '(%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3])
        print(' %-15s  %-13s   %7.3f\n' % (nstr, tstr, a123), end='')
    print('\n', end='')

# print list of torsion angles to screen
def print_torsions(geom, torsions):
    at_types = geom[0]
    n_torsions = len(torsions)
    print('%i torsion(s) found (degrees)' % (n_torsions))
    for q in range(n_torsions):
        n1, n2, n3, n4 = torsions[q][0:4]
        t1234 = torsions[q][4]
        nstr = '%i-%i-%i-%i' % (n1+1, n2+1, n3+1, n4+1)
        tstr = '(%s-%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3], at_types[n4])
        print(' %-15s  %-13s  %8.3f\n' % (nstr, tstr, t1234), end='')
    print('\n', end='')

# print list of out-of-plane angles to screen
def print_outofplanes(geom, outofplanes):
    at_types = geom[0]
    n_outofplanes = len(outofplanes)
    print('%i out-of-plane(s) found (degrees)' % (n_outofplanes))
    for q in range(n_outofplanes):
        n1, n2, n3, n4 = outofplanes[q][0:4]
        o1234 = outofplanes[q][4]
        nstr = '%i-%i-%i-%i' % (n1+1, n2+1, n3+1, n4+1)
        tstr = '(%s-%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3], at_types[n4])
        print(' %-15s  %-13s  %8.3f\n' % (nstr, tstr, o1234), end='')
    print('\n', end='')
    
# print center of mass coordinates
def print_com(com):
    print('molecular center of mass (Angstrom)\n(X,Y,Z):', end='')
    for p in range(3):
        print('%12.6f' % (com[p]), end='')
    print('\n\n', end='')

## MATH FUNCTIONS ##

# calculate distance between two 3-d cartesian coordinates
def get_r12(coords1, coords2):
    r2 = 0.0
    for p in range(3):
        r2 += (coords2[p] - coords1[p])**2
    r = math.sqrt(r2)
    return r

# calculate unit vector between to 3-d cartesian coordinates
def get_u12(coords1, coords2):
    r12 = get_r12(coords1, coords2)
    u12 = [0.0 for p in range(3)]
    for p in range(3):
        u12[p] = (coords2[p] - coords1[p]) / r12
    return u12

# calculate dot product between two unit vectors
def get_udp(uvec1, uvec2):
    udp = 0.0
    for p in range(3):
        udp += uvec1[p] * uvec2[p]
    udp = max(min(udp, 1.0), -1.0)
    return udp

# calculate unit cross product between two unit vectors
def get_ucp(uvec1, uvec2):
    ucp = [0.0 for p in range(3)]
    cos_12 = get_udp(uvec1, uvec2)
    sin_12 = math.sqrt(1 - cos_12**2)
    ucp[0] = (uvec1[1]*uvec2[2] - uvec1[2]*uvec2[1]) / sin_12
    ucp[1] = (uvec1[2]*uvec2[0] - uvec1[0]*uvec2[2]) / sin_12
    ucp[2] = (uvec1[0]*uvec2[1] - uvec1[1]*uvec2[0]) / sin_12
    return ucp

# calculate angle between three 3-d cartesian coordinates
def get_a123(coords1, coords2, coords3):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    dp2123 = get_udp(u21, u23)
    a123 = rad2deg * math.acos(dp2123)
    return a123

# calculate torsion angle between four 3-d cartesian coordinates
def get_t1234(coords1, coords2, coords3, coords4):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    u32 = get_u12(coords3, coords2)
    u34 = get_u12(coords3, coords4)
    u21c23 = get_ucp(u21, u23)
    u32c34 = get_ucp(u32, u34)
    dp = get_udp(u21c23, u32c34)
    sign = 2 * float(get_udp(u21c23, u34) < 0) - 1
    t1234 = rad2deg * sign * math.acos(dp)
    return t1234

# calculate out-of-plane (improper torsion) angle between four 3-d cartesian coordinates
def get_o1234(coords1, coords2, coords3, coords4):
    u42 = get_u12(coords4, coords2)
    u43 = get_u12(coords4, coords3)
    u41 = get_u12(coords4, coords1)
    u42c43 = get_ucp(u42, u43)
    dp = get_udp(u42c43, u41)
    o1234 = rad2deg * math.asin(dp)
    return o1234

# translate coordinates by a defined vector and scale factor
def translate_coords(geom, vector, scale):
    coords = geom[1]
    n_atoms = len(coords)
    for i in range(n_atoms):
        for j in range(3):
            coords[i][j] += scale * vector[j]
    return geom

# calculate center of mass of a set of atoms
def get_com(geom):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    com = [0.0 for p in range(3)]
    mass = 0.0
    for i in range(n_atoms):
        at_mass = at_masses[at_types[i]]
        mass += at_mass
        for j in range(3):
            com[j] += at_mass * coords[i][j]
    for p in range(3):
        com[p] /= mass
    return com

## TOPOLOGY FUNCTIONS ##

# build graph of which atoms are covalently bonded
def get_bond_graph(geom):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    bond_graph = [[] for i in range(n_atoms)]
    for i in range(n_atoms):
        covrad1 = cov_rads[at_types[i]]
        for j in range(i+1, n_atoms):
            covrad2 = cov_rads[at_types[j]]
            thresh = bond_thresh * (covrad1 + covrad2)
            r12 = get_r12(coords[i], coords[j])
            if (r12 < thresh):
                bond_graph[i].append(j)
                bond_graph[j].append(i)
    return bond_graph

# determine atoms which are covalently bonded from bond graph
def get_bonds(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    bonds = []
    for i in range(n_atoms):      
        for a in range(len(bond_graph[i])):
            j = bond_graph[i][a]
            if (i < j):
                r12 = get_r12(coords[i], coords[j])
                bonds.append([i, j, r12])
    return bonds

# determine atoms which form a bond angle from bond graph
def get_angles(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    angles = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            i = bond_graph[j][a]
            for b in range(a+1, n_jbonds):
                k = bond_graph[j][b]
                a123 = get_a123(coords[i], coords[j], coords[k])
                angles.append([i, j, k, a123])
    return angles

# determine atoms which form torsion angles from bond graph
def get_torsions(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    torsions = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            k = bond_graph[j][a]
            if (k < j):
                continue
            n_kbonds = len(bond_graph[k])
            for b in range(n_jbonds):
                i = bond_graph[j][b]
                if (i == k):
                    continue
                for c in range(n_kbonds):
                    l = bond_graph[k][c]
                    if (l == j or l == i):
                        continue
                    t1234 = get_t1234(coords[i], coords[j], coords[k], coords[l])
                    torsions.append([i, j, k, l, t1234])
    return torsions

# determine atoms which form out-of-plane angles from bond graph
def get_outofplanes(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    outofplanes = []
    for l in range(n_atoms):
        n_lbonds = len(bond_graph[l])
        for a in range(n_lbonds):
            i = bond_graph[l][a]
            for b in range(n_lbonds):
                j = bond_graph[l][b]
                if (i == j):
                    continue
                for c in range(b+1, n_lbonds):
                    k = bond_graph[l][c]
                    if (i == k):
                        continue
                    o1234 = get_o1234(coords[i], coords[j], coords[k], coords[l])
                    outofplanes.append([i, j, k, l, o1234])
    return outofplanes

## MAIN BLOCK ##

# read in geometry, determine bonded topology
xyz_file_name = get_inputs()
geom = get_geom(xyz_file_name)
bond_graph = get_bond_graph(geom)

# calculate bond lengths, angles, torsions, outofplanes, and center of mass
bonds = get_bonds(geom, bond_graph)
angles = get_angles(geom, bond_graph)
torsions = get_torsions(geom, bond_graph)
outofplanes = get_outofplanes(geom, bond_graph)
com = get_com(geom)

# print resulting values
print_geom(geom, 'initial geometry')
print_bonds(geom, bonds)
print_angles(geom, angles)
print_torsions(geom, torsions)
print_outofplanes(geom, outofplanes)
print_com(com)

# translate molecule to center of mass and print
com_geom = translate_coords(geom, com, -1.0)
print_geom(com_geom, 'center of mass translated geometry')

# end of program
