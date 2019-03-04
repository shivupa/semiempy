import numpy as np
import basis
from utils.molecule import Molecule
from basis.basis import Basis
from basis.gaussian import Gaussian

# STO-NG fitted parameters from Table I of doi:10.1063/1.1672392
sto_alpha_1s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],[],
    # sto_3g
    [3.207831241e+00, 5.843104649e-01, 1.581372150e-01]
]
sto_alpha_2s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],[],
    # sto_3g
    [5.145620502e+00, 1.195731544e+00, 3.888890096e-01]
]
# These are just the same as the 2s but if we ever decide to do the fit
# by ourselves we could let them become different values
sto_alpha_2p = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],[],
    # sto_3g
    [5.145620502e+00, 1.195731544e+00, 3.888890096e-01]
]
sto_coeff_1s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],[],
    # sto_3g
    [1.543289673e-01, 5.353281423e-01, 4.446345422e-01]
]
sto_coeff_2s = [
    # pad with two empty lists so index corr38esponds to sto_ng
    [],[],[],
    # sto_3g
    [-9.996722919e-02, 3.995128261e-01, 7.001154689e-01]
]
sto_coeff_2p = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],[],
    # sto_3g
    [1.559162750e-01, 6.076837186e-01, 3.919573931e-01]
]

# These functions are fit to Slaters with \zeta == 1. To scale them to a
# paticular atom multiply by \zeta_opt**2
# The following optimized zetas are from Table I of doi: 10.1063/1.1733573
# except for hydrogen which is from doi:10.1063/1.1727227
# Formally Li and Be don't require p functions and weren't provided in the
# paper so we repeat the 2s values. This would be consistent if these were
# calculated using Slater's rules for screening doi:10.1103/PhysRev.36.57
# because the screening only would depend on what the principal quantum
# number is.
opt_zeta_1s = {
    "H" : 1.0
}
opt_zeta_2s = {
    "O" : 1.0
}
opt_zeta_2p = {
    "O" : 1.0
}

class MinimalNoCore(Basis):
    """
    A minimal STO-NG basis with no core electrons

    Attributes
    ----------
    n_func : int
        number of atomic basis functions
    funcs : list
        list of basis functions
    """



    def __init__(self, mol, num_gaussians):
        Basis.__init__(self, mol)
        self.num_gaussians = num_gaussians
        self.n_func = 0
        self.funcs = []
        self.populate_basis_from_mol()

    def populate_basis_from_mol(self):
        for i in range(self.mol.n_atom):
            # hydrogen and helium only have s functions
            if(self.mol.at_num[i] < 3):
                self.n_func += 1
                exps = [j*opt_zeta_1s[self.mol.symb[i]]**2 for j in sto_alpha_1s[self.num_gaussians]]
                contract_coeff = sto_coeff_1s[self.num_gaussians]
                ang_mom = (0,0,0)
                pos = self.mol.xyz[i]
                on_center = i
                center_Z = self.mol.at_num[i]
                center_symb = self.mol.symb[i]
                new_func = Gaussian(exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
                self.funcs.append(new_func)
            elif (self.mol.at_num[i] > 2 and self.mol.at_num[i] < 11):
                self.n_func += 4
                exps = [j*opt_zeta_2s[self.mol.symb[i]]**2 for j in sto_alpha_2s[self.num_gaussians]]
                contract_coeff = sto_coeff_1s[self.num_gaussians]
                ang_mom = (0,0,0)
                pos = self.mol.xyz[i]
                on_center = i
                center_Z = self.mol.at_num[i]
                center_symb = self.mol.symb[i]
                new_func = Gaussian(exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
                self.funcs.append(new_func)

                exps = [j*opt_zeta_2p[self.mol.symb[i]]**2 for j in sto_alpha_2p[self.num_gaussians]]
                contract_coeff = sto_coeff_1s[self.num_gaussians]
                ang_mom = (1,0,0)
                pos = self.mol.xyz[i]
                on_center = i
                center_Z = self.mol.at_num[i]
                center_symb = self.mol.symb[i]
                new_func = Gaussian(exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
                self.funcs.append(new_func)

                exps = [j*opt_zeta_2p[self.mol.symb[i]]**2 for j in sto_alpha_2p[self.num_gaussians]]
                contract_coeff = sto_coeff_1s[self.num_gaussians]
                ang_mom = (0,1,0)
                pos = self.mol.xyz[i]
                on_center = i
                center_Z = self.mol.at_num[i]
                center_symb = self.mol.symb[i]
                new_func = Gaussian(exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
                self.funcs.append(new_func)

                exps = [j*opt_zeta_2p[self.mol.symb[i]]**2 for j in sto_alpha_2p[self.num_gaussians]]
                contract_coeff = sto_coeff_1s[self.num_gaussians]
                ang_mom = (0,0,1)
                pos = self.mol.xyz[i]
                on_center = i
                center_Z = self.mol.at_num[i]
                center_symb = self.mol.symb[i]
                new_func = Gaussian(exps, contract_coeff, ang_mom, pos, on_center, center_Z, center_symb)
                self.funcs.append(new_func)
            else:
                raise NotImplementedError("Functions above 2p are not implemented.")
