import numpy as np
from basis.function import Function

class Gaussian(Function):
    """
    Class to store information about a single gaussian basis function

    TODO: Add the functions necessary functions to evaluate gaussian to make a cube file

    Attributes
    ----------
    """
    def __init__(self, exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb):
        Function.__init__(self, exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
