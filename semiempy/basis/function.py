import numpy as np

class Function:
    """
    Class to store information about a single gaussian basis function

    Attributes
    ----------
    """
    def __init__(self, exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb):
        self.exponents = exps
        self.contract_coeff = contract_coeff
        self.angular_momentum = ang_mom
        self.pos = pos
        self.center = on_center
        self.center_Z = center_Z
        self.center_symb = center_symb
