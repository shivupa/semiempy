import pyscf
from pyscf import gto


def overlap(GaussianA, GaussianB):
    ang = ['S', 'P']
    ang_num_funcs = [1, 3]

    mol = gto.Mole()
    atoms = "ghost1 {} {} {}\n".format(GaussianA.pos[0], GaussianA.pos[1], GaussianA.pos[2])
    atoms += "ghost2 {} {} {}".format(GaussianB.pos[0], GaussianB.pos[1], GaussianB.pos[2])

    mol.atom = atoms
    ghost1basis = "X   {}\n".format(ang[GaussianA.angular_momentum[0]+GaussianA.angular_momentum[1]+GaussianA.angular_momentum[2]])
    for i in range(len(GaussianA.exponents)):
        ghost1basis += "\t{}\t{}\n".format(GaussianA.exponents[i], GaussianA.contract_coeff[i])
    ghost2basis = "X   {}\n".format(ang[GaussianB.angular_momentum[0]+GaussianB.angular_momentum[1]+GaussianB.angular_momentum[2]])
    for i in range(len(GaussianB.exponents)):
        ghost2basis += "\t{}\t{}\n".format(GaussianB.exponents[i], GaussianB.contract_coeff[i])

    basis = {}
    basis["ghost1"] = gto.basis.parse(ghost1basis)
    basis["ghost2"] = gto.basis.parse(ghost2basis)

    mol.basis = basis

    mol.build()
    num_funcs_on_A = ang_num_funcs[GaussianA.angular_momentum[0]+GaussianA.angular_momentum[1]+GaussianA.angular_momentum[2]]
    num_funcs_on_B = ang_num_funcs[GaussianB.angular_momentum[0]+GaussianB.angular_momentum[1]+GaussianB.angular_momentum[2]]
    idxA = num_funcs_on_A - 1
    idxB = num_funcs_on_A + num_funcs_on_B - 1
    return mol.intor('cint1e_ovlp_cart')[idxA][idxB]

__idx2_cache = {}
def idx2(i, j):
    if (i, j) in __idx2_cache:
        return __idx2_cache[i, j]
    elif i >= j:
        __idx2_cache[i, j] = int(i*(i+1)/2+j)
    else:
        __idx2_cache[i, j] = int(j*(j+1)/2+i)
    return __idx2_cache[i, j]


def idx4(i, j, k, l):
    return idx2(idx2(i, j), idx2(k, l))

def twoelec(GaussianA, GaussianB, GaussianC, GaussianD):

    ang = ['S', 'P']
    ang_num_funcs = [1, 3]

    mol = gto.Mole()
    atoms = "ghost1 {} {} {}\n".format(GaussianA.pos[0], GaussianA.pos[1], GaussianA.pos[2])
    atoms += "ghost2 {} {} {}\n".format(GaussianB.pos[0], GaussianB.pos[1], GaussianB.pos[2])
    atoms += "ghost3 {} {} {}\n".format(GaussianC.pos[0], GaussianC.pos[1], GaussianC.pos[2])
    atoms += "ghost4 {} {} {}".format(GaussianD.pos[0], GaussianD.pos[1], GaussianD.pos[2])

    mol.atom = atoms
    ghost1basis = "X   {}\n".format(ang[GaussianA.angular_momentum[0]+GaussianA.angular_momentum[1]+GaussianA.angular_momentum[2]])
    for i in range(len(GaussianA.exponents)):
        ghost1basis += "\t{}\t{}\n".format(GaussianA.exponents[i], GaussianA.contract_coeff[i])
    ghost2basis = "X   {}\n".format(ang[GaussianB.angular_momentum[0]+GaussianB.angular_momentum[1]+GaussianB.angular_momentum[2]])
    for i in range(len(GaussianB.exponents)):
        ghost2basis += "\t{}\t{}\n".format(GaussianB.exponents[i], GaussianB.contract_coeff[i])
    ghost3basis = "X   {}\n".format(ang[GaussianC.angular_momentum[0]+GaussianC.angular_momentum[1]+GaussianC.angular_momentum[2]])
    for i in range(len(GaussianC.exponents)):
        ghost3basis += "\t{}\t{}\n".format(GaussianC.exponents[i], GaussianC.contract_coeff[i])
    ghost4basis = "X   {}\n".format(ang[GaussianD.angular_momentum[0]+GaussianD.angular_momentum[1]+GaussianD.angular_momentum[2]])
    for i in range(len(GaussianD.exponents)):
        ghost4basis += "\t{}\t{}\n".format(GaussianD.exponents[i], GaussianD.contract_coeff[i])

    basis = {}
    basis["ghost1"] = gto.basis.parse(ghost1basis)
    basis["ghost2"] = gto.basis.parse(ghost2basis)
    basis["ghost3"] = gto.basis.parse(ghost3basis)
    basis["ghost4"] = gto.basis.parse(ghost4basis)

    mol.basis = basis

    mol.build()

    num_funcs_on_A = ang_num_funcs[GaussianA.angular_momentum[0]+GaussianA.angular_momentum[1]+GaussianA.angular_momentum[2]]
    num_funcs_on_B = ang_num_funcs[GaussianB.angular_momentum[0]+GaussianB.angular_momentum[1]+GaussianB.angular_momentum[2]]
    num_funcs_on_C = ang_num_funcs[GaussianC.angular_momentum[0]+GaussianC.angular_momentum[1]+GaussianC.angular_momentum[2]]
    num_funcs_on_D = ang_num_funcs[GaussianD.angular_momentum[0]+GaussianD.angular_momentum[1]+GaussianD.angular_momentum[2]]
    idxA = num_funcs_on_A - 1
    idxB = num_funcs_on_A + num_funcs_on_B - 1
    idxC = num_funcs_on_A + num_funcs_on_B + num_funcs_on_C - 1
    idxD = num_funcs_on_A + num_funcs_on_B + num_funcs_on_C + num_funcs_on_D - 1
    return mol.intor('cint2e_sph', aosym='s8')[idx4(idxA,idxB,idxC,idxD)]
