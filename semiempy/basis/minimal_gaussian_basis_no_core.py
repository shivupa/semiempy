import numpy as np
import basis
from utils.molecule import Molecule
from basis.basis import Basis
from basis.gaussian import Gaussian

# STO-NG fitted parameters from Table I of doi:10.1063/1.1672392
sto_alpha_1s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [1.51623e-1, 8.51819e-1],
    # sto_3g
    [1.09818e-1, 4.05771e-1, 2.22766],
    # sto_4g
    [8.80187e-2, 2.65204e-1, 9.54620e-1, 5.21686],
    # sto_5g
    [7.44527e-2, 1.97572e-1, 5.78648e-1, 2.07173, 1.13056e1],
    # sto_6g
    [6.51095e-2, 1.58088e-1, 4.07099e-1, 1.18506, 4.23592, 2.31030e1]
]
sto_alpha_2s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [9.74545e-2, 3.84244e-1],
    # sto_3g
    [7.51386e-2, 2.31031e-1, 9.94203e-1],
    # sto_4g
    [6.28104e-2, 1.6351-1, 5.02989e-1, 2.32350],
    # sto_5g
    [5.44949e-2, 1.27920e-1, 3.29060e-1, 1.03250, 5.03629],
    # sto_6g
    [4.85690e-2, 1.05960e-1, 2.43977e-1, 6.34142e-1, 2.04036, 1.03087e1]
]
# These are just the same as the 2s but if we ever decide to do the fit
# by ourselves we could let them become different values
sto_alpha_2p = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [9.74545e-2, 3.84244e-1],
    # sto_3g
    [7.51386e-2, 2.31031e-1, 9.94203e-1],
    # sto_4g
    [6.28104e-2, 1.6351-1, 5.02989e-1, 2.32350],
    # sto_5g
    [5.44949e-2, 1.27920e-1, 3.29060e-1, 1.03250, 5.03629],
    # sto_6g
    [4.85690e-2, 1.05960e-1, 2.43977e-1, 6.34142e-1, 2.04036, 1.03087e1]
]
sto_coeff_1s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [6.78914e-1, 4.30129e-1],
    # sto_3g
    [4.44635e-1, 5.35328e-1, 1.54329e-1],
    # sto_4g
    [2.91626e-1, 5.32846e-1, 2.60141e-1, 5.67523e-2],
    # sto_5g
    [1.93572e-1, 4.82570e-1, 3.31816e-1, 1.13541e-1, 2.21406e-2],
    # sto_6g
    [1.30334e-1, 4.16492e-1, 3.70563e-1, 1.68538e-1, 4.93615e-2, 9.16360e-3]
]
sto_coeff_2s = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [9.63782e-1, 4.94718e-2],
    # sto_3g
    [7.00115e-1, 3.99513e-1, -9.99672e-2],
    # sto_4g
    [4.97767e-1, 5.58855e-1, 2.97680e-5, -6.22071e-2],
    # sto_5g
    [3.46121e-1, 6.12290e-1, 1.28997e-1, -6.53275e-2, -2.94086e-2],
    # sto_6g
    [2.40706e-1, 5.95117e-1, 2.50242e-1, -3.37854e-2, -4.69917e-2, -1.32528e-2]
]
sto_coeff_2p = [
    # pad with two empty lists so index corresponds to sto_ng
    [],[],
    # sto_2g
    [6.12820e-1, 5.11541e-1],
    # sto_3g
    [3.91957e-1, 6.07684e-1, 1.55916e-1],
    # sto_4g
    [2.46313e-1, 5.83575e-1, 2.86379e-1, 4.36843e-2],
    # sto_5g
    [1.56828e-1, 5.10240e-1, 3.73598e-1, 1.07558e-1, 1.25561e-2],
    # sto_6g
    [1.01708e-1, 4.25860e-1, 4.18036e-1, 1.73897e-1, 3.76794e-2, 3.75970e-3]
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
    "H" : 1.2,
    "Li" : 2.6906,
    "Be" : 3.6848,
    "B" : 4.6795,
    "C" : 5.6727,
    "N" : 6.6651,
    "O" : 7.6579,
    "F" : 8.6501
}
opt_zeta_2s = {
    "Li" : 0.6396,
    "Be" : 0.9560,
    "B" : 1.2881,
    "C" : 1.6083,
    "N" : 1.9237,
    "O" : 2.2458,
    "F" : 2.5638
}
opt_zeta_2p = {
    "Li" : 0.6396,
    "Be" : 0.9560,
    "B" : 1.2107,
    "C" : 1.5679,
    "N" : 1.9170,
    "O" : 2.2266,
    "F" : 2.5500
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
