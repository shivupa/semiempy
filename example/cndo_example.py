import numpy
import sys
sys.path.append("../semiempy")
from methods.CNDO import CNDO
from utils.molecule import Molecule
# from match_gaussian_basis import MinimalNoCore
from basis.minimal_gaussian_basis_no_core import MinimalNoCore

structure = 'eqh2o.xyz'
m = Molecule(structure)
b = MinimalNoCore(m,num_gaussians=3)

method = CNDO(m, b)
method.run()
