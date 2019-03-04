"""
Various math utils for functions
"""
import numpy as np



def distance(molecule, atomi, atomj):
    """
    Returns the length between two atoms

    Parameters
    -----------
    molecule : object
        molecule object
    atomi, atomj : int
        atoms

    Returns
    --------
    rij : float
        length between the two
    """
    x = molecule.xyz[atomi][0] - molecule.xyz[atomj][0]
    y = molecule.xyz[atomi][1] - molecule.xyz[atomj][1]
    z = molecule.xyz[atomi][2] - molecule.xyz[atomj][2]
    rij = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    return rij
