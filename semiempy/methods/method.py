import numpy as np
import time
from utils.molecule_utils import distance
from utils.general_io import print_header

class Method:
    """
    Class for semiempirical methods

    Attributes
    ----------
    """

    def __init__(self, mol, bas):
        self.mol = mol
        self.bas = bas
        self.iteration_max = 100
        self.convergence_E = 1e-9
        self.convergence_DM = 1e-5
        # loop variables
        self.iteration_start_time = 0
        self.iteration_num = 0
        self.E_total = 0
        self.E_elec = 0.0
        self.iteration_E_diff = 0.0
        self.iteration_rmsc_dm = 0.0
        self.stop = False
        self.converged = False
        self.exceeded_iterations = False

    def print_start_iterations(self):
        print_header()
        print("{:^79}".format("Starting {}!".format(self.name)))
        print("{:^79}".format("{:>4}  {:>11}  {:>11}  {:>11}  {:>11}".format(
            "Iter", "Time(s)", "RMSC DM", "delta E", "E_elec")))
        print("{:^79}".format("{:>4}  {:>11}  {:>11}  {:>11}  {:>11}".format(
            "****", "*******", "*******", "*******", "******")))

    def print_iteration(self):
        print("{:^79}".format("{:>4d}  {:>11f}  {:>.5E}  {:>.5E}  {:>11f}".format(self.iteration_num,self.iteration_end_time - self.iteration_start_time, self.iteration_rmsc_dm, self.iteration_E_diff, self.E_elec)))

    def print_error(self):
        print("{:^79}".format("SOMETHING HAS GONE HORRIBLY WRONG!"))

    def print_success(self):
        print("{:^79}".format("{} Converged!".format(self.name)))
        print("{:^79}".format("{:>20}  {:>11f}".format("ELECTRONIC ENERGY",self.E_elec)))
        print("{:^79}".format("{:>20}  {:>11f}".format("NUCLEAR REPULSION ENERGY",self.mol.E_nuc)))
        print("{:^79}".format("{:>20}  {:>11f}".format("TOTAL ENERGY",self.E_total)))
        print("{:^79}".format("{:>20}  {:>11f}".format("RUNTIME (s)",self.end_time - self.start_time)))

    def print_exceeded_iterations(self):
        print("{:^79}".format("Did not converge after {:>5d} iterations!".format(self.iteration_max)))
        print("{:^79}".format("{:>20}  {:>11f}".format("RUNTIME (s)",self.end_time - self.start_time)))

    def print_error(self):
        print("{:^79}".format("SOMETHING HAS GONE HORRIBLY WRONG!"))
        print("{:^79}".format("{:>20}  {:>11f}".format("RUNTIME (s)",self.end_time - self.start_time)))

    def overlap(self):
        raise NotImplementedError('Need to implement method')

    def H_core(self):
        raise NotImplementedError('Need to implement method')

    def kinetic(self):
        raise NotImplementedError('Need to implement method')

    def nuclear_attraction(self):
        raise NotImplementedError('Need to implement method')

    def two_electron(self):
        raise NotImplementedError('Need to implement method')

    def form_fock(self):
        raise NotImplementedError('Need to implement method')

    def diag_fock(self):
        raise NotImplementedError('Need to implement method')

    def form_DM(self):
        raise NotImplementedError('Need to implement method')

    def calculate_E_elec(self):
        self.E_elec = np.sum(np.multiply(self.D, (self.H + self.F)))

    def calculate_E_total(self):
        self.E_total = self.E_elec + self.mol.E_nuc

    def check_stop(self):
        # calculate energy change of iteration
        self.iteration_E_diff = np.abs(self.E_elec - self.E_elec_last)
        # rms change of density matrix
        self.iteration_rmsc_dm = np.sqrt(np.sum((self.D - self.D_last)**2))
        # check stopping criteria
        if(np.abs(self.iteration_E_diff) < self.convergence_E and self.iteration_rmsc_dm < self.convergence_DM):
            self.converged = True
            self.stop = True
        elif(self.iteration_num == self.iteration_max):
            self.exceeded_iterations = True
            self.stop = True

    def run_iteration(self):
        # store last iteration and increment counters
        self.iteration_start_time = time.time()
        self.iteration_num += 1
        self.E_elec_last = self.E_elec
        self.D_last = np.copy(self.D)
        # build fock matrix
        self.form_fock()
        # solve the generalized eigenvalue problem
        self.diag_fock()
        # compute new density matrix
        self.form_DM()
        # calculate electronic energy
        self.calculate_E_elec()
        self.iteration_end_time = time.time()
        self.print_iteration()

    def guess_DM(self):
        # for now just initialize with zeros
        self.D = np.zeros((self.bas.n_func, self.bas.n_func))

    def run(self):
        self.start_time = time.time()
        self.print_start_iterations()
        self.guess_DM()
        self.H_core()
        while (not self.stop):
            self.run_iteration()
            self.check_stop()
        self.calculate_E_total()
        self.end_time = time.time()
        if (self.stop and self.converged):
            self.print_success()
        elif (self.stop and self.exceeded_iterations):
            self.print_exceeded_iterations()
        else:
            self.print_error()
