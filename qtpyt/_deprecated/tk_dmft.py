import numpy as np


# with open('selfenergy','w') as fp:
#     m, n = Sigret.shape[1:]
#     for energy,selfenergy in zip(energies,Sigret):
#         fp.write('{:.9}'.format(energy))
#         fp.write('\n')
#         for i in range(3,4):
#             for j in range(3,4):
#                 fp.write('{:.9} {:.9} '.format(selfenergy[i,j].real,selfenergy[i,j].imag))
#             fp.write('\n')
#
# with open('selfenergy','w') as fp:
#     m, n = Sigret.shape[1:]
#     for energy,selfenergy in zip(energies,Sigret):
#         fp.write('{:.9}'.format(energy))
#         fp.write('\n')
#         fp.write('{:.9} {:.9} '.format(selfenergy[3,3].imag, selfenergy[3,3].real))
#         fp.write('\n')
#
# with open('selfenergy-mats-{}'.format(beta),'w') as fp:
#     m, n = Sigret.shape[1:]
#     for n,selfenergy in zip(ns,Sig_mats):
#         w_n = np.pi/beta * (2*n + 1)
#         fp.write('{:.9}'.format(w_n))
#         fp.write('\n')
#         fp.write('{:.9} {:.9} '.format(selfenergy[3,3].imag, selfenergy[3,3].real))
#         fp.write('\n')


def load_LSigma(filename):

    Sigmas = np.loadtxt(filename)
    Sigmas.T[[1, 2]] = Sigmas.T[[2, 1]]
    Sigmas = Sigmas.view("f8,c16")
    Sigmas = np.split(Sigmas, 2)

    return Sigmas


def dump_variables(prefix, *variables, dir="."):

    import os
    import itertools

    for var in variables:

        def variablename(var):
            return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]

        name = variablename(var)
        if len(name) > 1:
            raise Warning("Multiple names for variable", name)
        name = name[0]
        file = os.path.join(dir, prefix) + "_" + name + ".pickle"

        pickle.dump(var, open(file, "wb"), 2)


def get_matsubara(w, Sigma, N_mats, beta, S):
    #     Nw = len(w)
    Nw, n, m = Sigma.shape
    w_mats = np.pi / beta * (2 * np.arange(N_mats) + 1)
    Re_mats = np.zeros((N_mats, n, m), float)
    Im_mats = np.zeros((N_mats, n, m), float)
    A_wmm = Sigma.imag / np.pi
    w_wmm = S[None, :] * w[:, None, None]
    w2_wmm = S[None, :] * (w ** 2)[:, None, None]
    w_mats_wmm = S[None, None, :] * w_mats[:, None, None, None]
    w2_mats_wmm = S[None, None, :] * (w_mats ** 2)[:, None, None, None]
    print(A_wmm.shape, w_wmm.shape, w_mats_wmm.shape)
    for i in range(N_mats):
        Re_mats[i] = np.trapz(A_wmm * w_wmm / (w2_wmm + w2_mats_wmm[i]), w, axis=0)
        Im_mats[i] = np.trapz(
            A_wmm * w_mats_wmm[i] / (w2_wmm + w2_mats_wmm[i]), w, axis=0
        )
    return Re_mats + 1j * Im_mats
