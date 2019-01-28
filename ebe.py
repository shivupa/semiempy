import numpy as np
import scipy.constants as sc
import argparse


parser = argparse.ArgumentParser()

# ARGUMENTS TO PARSE

har_to_ev= sc.physical_constants["Hartree energy in eV"][0]

beta = {
    "H" : 9,
    "Li" : 9,
    "Be" : 13,
    "B" : 17,
    "C" : 21,
    "N" : 25,
    "O" : 31,
    "F" : 39
}

avg_IP_EA_s = {
    "H" : 7.176,
    "Li" : 3.106,
    "Be" : 5.946,
    "B" : 9.594,
    "C" : 14.051,
    "N" : 19.316,
    "O" : 25.390,
    "F" : 32.272
}
avg_IP_EA_p = {
    "Li" : 1.258,
    "Be" : 2.563,
    "B" : 4.001,
    "C" : 5.572,
    "N" : 7.275,
    "O" : 9.111,
    "F" : 11.080
}

for i in beta.keys():
    print('"{}" : {:6E},'.format(i,beta[i]/har_to_ev))
