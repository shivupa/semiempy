import numpy as np
import basis
import Molecule

class MinimalNoCore(Basis):
    """
    A minimal basis with no core electrons

    Attributes
    ----------
    exponents : list
        list of basis function exponents
    angular_momentum : list
        list of basis function angular momentum
    pos : list
        list of basis function positions
    type : string
        type of basis functions
    """

    def __init__(self, mol):
        Basis.__init__(self, mol)
