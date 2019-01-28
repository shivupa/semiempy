import numpy as np
from integrals import gaussian_integrals as gi
from methods.method import Method
import scipy.linalg as spla
# MATRIX ELEMENTS FROM Table I of doi:10.1063/1.1727227 in eV
# avg_IP_EA_s = {
#     "H" : 7.176,
#     "Li" : 3.106,
#     "Be" : 5.946,
#     "B" : 9.594,
#     "C" : 14.051,
#     "N" : 19.316,
#     "O" : 25.390,
#     "F" : 32.272
# }
# avg_IP_EA_p = {
#     "Li" : 1.258,
#     "Be" : 2.563,
#     "B" : 4.001,
#     "C" : 5.572,
#     "N" : 7.275,
#     "O" : 9.111,
#     "F" : 11.080
# }
# Matrix elements from Table I of doi:10.1063/1.1727227 in Ha
avg_IP_EA_s = {
    "H": 2.637131E-01,
    "Li": 1.141434E-01,
    "Be": 2.185115E-01,
    "B": 3.525730E-01,
    "C": 5.163647E-01,
    "N": 7.098499E-01,
    "O": 9.330653E-01,
    "F": 1.185974E+00
}
avg_IP_EA_p = {
    "Li": 4.623065E-02,
    "Be": 9.418851E-02,
    "B": 1.470340E-01,
    "C": 2.047672E-01,
    "N": 2.673513E-01,
    "O": 3.348231E-01,
    "F": 4.071825E-01
}
# Beta variable entering off-diagonal terms for Fock matrix
# in eV from Table II of doi: 10.1063/1.1701476
# beta = {
#     "H" : 9,
#     "Li" : 9,
#     "Be" : 13,
#     "B" : 17,
#     "C" : 21,
#     "N" : 25,
#     "O" : 31,
#     "F" : 39
# }
# in Hartree from Table II of doi: 10.1063/1.1701476
beta = {
    "H": 3.307439E-01,
    "Li": 3.307439E-01,
    "Be": 4.777412E-01,
    "B": 6.247385E-01,
    "C": 7.717358E-01,
    "N": 9.187331E-01,
    "O": 1.139229E+00,
    "F": 1.433224E+00
}

class CNDO(Method):
    """
    Class for CNDO

    Attributes
    ----------
    """

    def __init__(self, mol, bas):
        Method.__init__(self, mol, bas)
        self.name = "CNDO/2"

    def overlap(self):
        return None

    def H_core(self):
        self.H = np.zeros((self.bas.n_func, self.bas.n_func))
        if self.iteration_num == 1:
            for i in range(self.bas.n_func):
                if self.bas.funcs[i].center_symb in ['H']:
                    self.H[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
                else:
                    ang_mom = self.bas.funcs[i].angular_momentum[0]
                    ang_mom += self.bas.funcs[i].angular_momentum[1]
                    ang_mom += self.bas.funcs[i].angular_momentum[2]
                    if (ang_mom == 0):
                        self.H[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
                    else:
                        self.H[i, i] = -avg_IP_EA_p[self.bas.funcs[i].center_symb]
                for j in range(i+1, self.bas.n_func):
                    beta_avg = 0.5*(beta[self.bas.funcs[i].center_symb] + beta[self.bas.funcs[j].center_symb])
                    self.H[i, j] = beta_avg*gi.overlap(self.bas.funcs[i], self.bas.funcs[j])
                    self.H[j, i] = self.H[i, j]

    def kinetic(self):
        return None

    def nuclear_attraction(self):
        return None

    def two_electron(self):
        return None

    def form_DM(self):
        self.D = np.zeros((self.bas.n_func, self.bas.n_func))
        for i in range(self.bas.n_func):
            for j in range(self.bas.n_func):
                for k in range((self.mol.num_val_elec)//2):
                    self.D[i, j] += self.C[i, k] * self.C[j, k]

    def form_DM_on_centers(self):
        self.D_centers = np.zeros((self.mol.n_atom))
        for i in range(self.bas.n_func):
            atom = self.bas.funcs[i].center
            for k in range((self.mol.num_val_elec)//2):
                self.D_centers[atom] += self.C[i, k] * self.C[i, k]

    def diag_fock(self):
        self.E_orbitals, self.C = spla.eigh(self.F)

    def form_fock(self):
        self.F = np.zeros((self.bas.n_func, self.bas.n_func))
        if self.iteration_num == 1:
            # for i in range(self.bas.n_func):
            #     if self.bas.funcs[i].center_symb in ['H']:
            #         self.F[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
            #     else:
            #         ang_mom = self.bas.funcs[i].angular_momentum[0]
            #         ang_mom += self.bas.funcs[i].angular_momentum[1]
            #         ang_mom += self.bas.funcs[i].angular_momentum[2]
            #         if (ang_mom == 0):
            #             self.F[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
            #         else:
            #             self.F[i, i] = -avg_IP_EA_p[self.bas.funcs[i].center_symb]
            #     for j in range(i+1, self.bas.n_func):
            #         beta_avg = 0.5*(beta[self.bas.funcs[i].center_symb] + beta[self.bas.funcs[j].center_symb])
            #         self.F[i, j] = beta_avg*gi.overlap(self.bas.funcs[i], self.bas.funcs[j])
            #         self.F[j, i] = self.F[i, j]
            pass
        else:
            self.F += self.H
            self.form_DM_on_centers()
            for i in range(self.bas.n_func):
                if self.bas.funcs[i].center_symb in ['H']:
                    self.F[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
                    gamma_AA = gi.twoelec(self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[i])
                    center_i = self.bas.funcs[i].center
                    self.F[i, i] += ((self.D_centers[center_i] - self.mol.at_num[center_i]) - (0.5*(self.D[i,i] - 1))) * gamma_AA
                    for j in range(self.bas.n_func):
                        center_j = self.bas.funcs[j].center
                        if center_i != center_j:
                            gamma_AB = gi.twoelec(self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[j], self.bas.funcs[j])
                            self.F[i, j] += (self.D_centers[center_j] - self.mol.at_num[center_j]) * gamma_AA
                            self.F[j, i] = self.F[i, j]
                else:
                    ang_mom = self.bas.funcs[i].angular_momentum[0]
                    ang_mom += self.bas.funcs[i].angular_momentum[1]
                    ang_mom += self.bas.funcs[i].angular_momentum[2]
                    if (ang_mom == 0):
                        self.F[i, i] = -avg_IP_EA_s[self.bas.funcs[i].center_symb]
                    else:
                        self.F[i, i] = -avg_IP_EA_p[self.bas.funcs[i].center_symb]
                    gamma_AA = gi.twoelec(self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[i])
                    center_i = self.bas.funcs[i].center
                    self.F[i, i] += ((self.D_centers[center_i] - self.mol.at_num[center_i]) - (0.5*(self.D[i,i] - 1))) * gamma_AA
                    for j in range(self.bas.n_func):
                        center_j = self.bas.funcs[j].center
                        if center_i != center_j:
                            gamma_AB = gi.twoelec(self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[j], self.bas.funcs[j])
                            self.F[i, j] += (self.D_centers[center_j] - self.mol.at_num[center_j]) * gamma_AA
                for j in range(i+1,self.bas.n_func):
                    S_munu = gi.overlap(self.bas.funcs[i], self.bas.funcs[j])
                    gamma_AB = gi.twoelec(self.bas.funcs[i], self.bas.funcs[i], self.bas.funcs[j], self.bas.funcs[j])
                    beta_avg = 0.5*(beta[self.bas.funcs[i].center_symb] + beta[self.bas.funcs[j].center_symb])
                    self.F[i, j] = (beta_avg*S_munu) - (0.5 * self.D[i,j] * gamma_AB)
                    self.F[j, i] = self.F[i, j]

    def generate_basis(self):
        self.bas = None
