import numpy as np
import cclib
from utils.atom_info import nuc
from utils.atom_info import core_electrons
from utils.molecule_utils import distance
import scipy.constants as sc


class Molecule:
    """
    Class to store molecule information

    Attributes
    ----------
    n_atom : int
        number of atoms
    n_elec : int
        number of electrons
    num_elec_core : int
        List of number of core electrons. Size: (n_atom,1)
    charge : int
        charge on molecule
    multiplicity : int
        multiplicity (2S+1) of molecule
    xyz : float
        xyz coordinates. Size: (n_atom,3)
    symb :
        List of atomic symbol. Size: (n_atom,1)
    at_num :
        List of atomic numbers. Size: (n_atom,1)
    """
    __accepted_file_formats = ['xyz', 'sdf', 'mol']

    def __init__(self, fname=None, charge=0, multiplicity=1):
        if fname is not None:
            self.import_file(fname)
        self.charge = charge
        self.multplicity = multiplicity
        self.calculate_E_nuc()
        return None

    def calculate_E_nuc(self):
        self.E_nuc = 0.0
        for i in range(self.n_atom):
            for j in range(i+1,self.n_atom):
                Z1 = self.at_num[i] - self.num_elec_core[i]
                Z2 = self.at_num[j] - self.num_elec_core[j]
                distance_in_angstroms = distance(self,i,j) * sc.nano * 0.1 / sc.physical_constants["atomic unit of length"][0]
                self.E_nuc += Z1*Z2/distance_in_angstroms
    def symb2num(self, symb):
        """
        Given a chemical symbol, returns the atomic number defined within the class

        Parameters
        -----------
        symb : string
            chemical symbol

        Returns
        --------
        at_num : int
            atomic number for symbol argument
        """
        try:
            atomic_number = nuc['{}'.format(symb)]
            return atomic_number
        except:
            print('{} is not defined.'.format(symb))

    def symb2numelec(self, symb):
        """
        Given a chemical symbol, returns the number of electrons and core electrons

        Parameters
        -----------
        symb : string
            chemical symbol

        Returns
        --------
        num_elec : int
            number of electrons
        num_core_elec : int
            number of core electrons
        """
        try:
            num_elec = nuc['{}'.format(symb)]
            num_core_elec = core_electrons['{}'.format(symb)]
            return num_elec, num_core_elec
        except:
            print('{} is not defined.'.format(symb))

    def import_file(self, fname):
        filetype = fname.split('.')[1]
        if filetype not in Molecule.__accepted_file_formats:
            parsed_properly = self.import_cclib(fname)
            if not parsed_properly:
                formatted_aff = str(
                    Molecule.__accepted_file_formats).strip('[]')
                raise NotImplementedError(
                    'file type \'{}\'  is unsupported. Accepted formats: {} or any cclib suported format.'.format(filetype, formatted_aff))
        if filetype == 'xyz':
            self.import_xyz(fname)
        elif filetype == 'sdf' or filetype == 'mol':
            self.import_sdf(fname)

    def import_xyz(self, fname):
        """
        Imports xyz file as a Molecule class instance

        Parameters
        ----------
        fname : string
            xyz filename
        """
        self.ftype = 'xyz'
        with open(fname) as f:
            lines = f.readlines()
        self.n_atom = int(lines[0].split()[0])

        # reading lines to build up class data
        self.symb = []
        self.at_num = []
        self.num_elec = 0
        self.num_val_elec = 0
        self.num_elec_core = []
        self.xyz = np.zeros((self.n_atom, 3))
        for i, line in enumerate(lines[2:]):
            tmp = line.split()
            self.symb.append(tmp[0])
            self.at_num.append(self.symb2num(tmp[0]))
            num_elec_on_atom, num_core_elec_on_atom = self.symb2numelec(tmp[0])
            self.num_elec += num_elec_on_atom
            self.num_val_elec += num_elec_on_atom - num_core_elec_on_atom
            self.num_elec_core.append(num_core_elec_on_atom)
            self.xyz[i, 0] = float(tmp[1])
            self.xyz[i, 1] = float(tmp[2])
            self.xyz[i, 2] = float(tmp[3])

    def import_sdf(self, fname):
        """
        Imports xyz file as a Molecule class instance

        Parameters
        ----------
        fname : string
            sdf or mol file name
        """
        self.ftype = 'sdf'
        with open(fname) as f:
            lines = f.readlines()
        self.n_atom = int(lines[3].split()[0])
        self.n_connect = int(lines[3].split()[1])
        self.symb = []
        self.at_num = []
        self.n_place = []
        self.num_elec = 0
        self.num_elec_core = []
        self.xyz = np.zeros((self.n_atom, 3))
        for i, line in enumerate(lines[4:4+self.n_atom]):
            tmp = line.split()
            self.symb.append(tmp[3])
            self.at_num.append(self.symb2num(tmp[3]))
            num_elec_on_atom, num_core_elec_on_atom = self.symb2numelec(tmp[3])
            self.num_elec += num_elec_on_atom
            self.num_elec_core.append(num_core_elec_on_atom)
            self.xyz[i, 0] = float(tmp[0])
            self.xyz[i, 1] = float(tmp[1])
            self.xyz[i, 2] = float(tmp[2])
            self.n_place.append(i)
        self.connect = np.zeros((self.n_connect, 2))
        for i, line in enumerate(lines[4+self.n_atom:4+self.n_atom+self.n_connect]):
            tmp = line.split()
            self.connect[i, 0] = tmp[0]
            self.connect[i, 1] = tmp[1]

    def import_cclib(self, fname):
        """
        Imports any cclib parsable file as a Molecule class instance

        Parameters
        -----------
        fname : string
            cclib parsable output file name
        """
        try:
            self.ftype = 'cclib'
            data = cclib.io.ccread(fname)
            self.n_atom = data.natom
            self.at_num = data.atomnos
            # This gets the atomic symbols by looking up the keys of the
            # atomic numbers. It looks somewhat crazy but it is looking
            # through a list of the values stored in the dictionary,
            # matching the value to the atomic number and returning
            # the key that corresponds to that atomic number. It works
            # with this dictionary because the keys to values are 1 to 1.
            self.symb = []
            for i in data.atomnos:
                self.symb.append(list(nuc.keys())[
                                list(nuc.values()).index(i)])
            # cclib stores the atomic coordinates in a array of shape
            # [molecule, num atoms, 3 for xyz] because I think they might
            # have many "molecules" from each step of an optimization or
            # something. Here we are taking just the last one.
            self.xyz = data.atomcoords[-1]
            return True
        except:
            return False
