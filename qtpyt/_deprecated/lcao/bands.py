import itertools
import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset


def reduce_vecs(R):
    """This function reduces symmetries along the last
    two directions"""
    R_rc = []
    for r in R:
        try:
            if r[np.nonzero(r)[0][0]] > 0:
                R_rc.append(r)
            else:
                continue
        # zero case
        except IndexError:
            R_rc.append(r)
    return R_rc


def get_realspace_R(kpts_grid, reduce_cc=True):
    """This function returns the real space hopping vectors,
    in crystal coordinates, for the input monkhorst pack size."""
    ndim = len(kpts_grid)  # number of dimensions
    R_rc = list(
        itertools.product(
            *[range(-kpts_grid[i] + 1, kpts_grid[i]) for i in range(len(kpts_grid))]
        )
    )
    # if reduce_cc is True, reduce symmetry in xy-directions.
    if reduce_cc:
        return reduce_vecs(R_rc)
    return R_rc


# def get_realspace_mat(mat_kmm, k_kc, R_rc, w_k):
#     '''This function computes the fourier transform of the matrix
#     evaluated in k-space, at point(s) R_rc.'''
#     R_rc = np.array(R_rc)
#     if R_rc.ndim < 2:
#         R_rc = R_rc[None, :]
#     mat_rmm = []
#     for i, R_c in enumerate(R_rc):
#         mat_R = np.zeros_like(mat_kmm[0])
#         for j, k_c in enumerate(k_kc):
#             c_k = np.exp(2.j * np.pi * np.dot(k_c, R_c)) * w_k[j]
#             try:
#                 if k_c[np.nonzero(k_c)[0][0]] > 0:
#                     pass
#                 else:
#                     continue
#             # zero case
#             except IndexError:
#                 c_k *= 0.5
#             mat_R += mat_kmm[j] * c_k
#         mat_R += mat_R.T.conj()
#         mat_rmm.append(mat_R)
#     return mat_rmm


def get_realspace_mat(mat_kmm, k_kc, R_rc, w_k, flag_hermitian=False):
    """This function computes the fourier transform of the matrix
    evaluated in k-space, at point(s) R_rc."""
    R_rc = np.array(R_rc)
    if R_rc.ndim < 2:
        R_rc = R_rc[None, :]
    mat_rmm = []
    for i, R_c in enumerate(R_rc):
        mat_R = np.zeros_like(mat_kmm[0])
        for j, k_c in enumerate(k_kc):
            c_k = np.exp(2.0j * np.pi * np.dot(k_c, R_c)) * w_k[j]
            if flag_hermitian:
                try:
                    if k_c[np.nonzero(k_c)[0][0]] > 0:
                        pass
                    else:
                        continue
                # zero case
                except IndexError:
                    c_k *= 0.5
            mat_R += mat_kmm[j] * c_k
        if flag_hermitian:
            mat_R += mat_R.T.conj()
        mat_rmm.append(mat_R)
    return mat_rmm


def get_realspace_hs(h_skmm, s_kmm, bzk_kc, weight_k):
    kpts_grid, kpts_shift = get_monkhorst_pack_size_and_offset(bzk_kc)
    R_rc = get_realspace_R(kpts_grid // 2)
    ibzk_kc = reduce_vecs(bzk_kc.tolist())

    h_srii = np.zeros_like(h_skmm)
    s_rii = np.zeros_like(s_kmm)
    for r, R_c in enumerate(R_rc):
        for k, ibzk_c in enumerate(ibzk_kc):
            c_k = np.exp(2.0j * np.pi * np.dot(ibzk_c, R_c)) * weight_k[k]
            h_srii[:, r] += c_k * h_skmm[:, k]
            s_rii[r] += c_k * s_kmm[k]

    return h_srii, s_rii, R_rc


def get_transport_kpts(bzk_kc, dir="x"):

    from ase.dft.kpoints import get_monkhorst_pack_size_and_offset, monkhorst_pack

    dir = "xyz".index(direction)
    transverse_dirs = np.delete([0, 1, 2], [dir])
    dtype = float
    if len(bzk_kc) > 1 or np.any(bzk_kc[0] != [0, 0, 0]):
        dtype = complex

    kpts_grid, kpts_shift = get_monkhorst_pack_size_and_offset(bzk_kc)

    # kpts in the transport direction
    nkpts_p = kpts_grid[dir]
    bzk_p_kc = monkhorst_pack((nkpts_p, 1, 1))[:, 0] + kpts_shift[dir]
    weight_p_k = 1.0 / nkpts_p

    # kpts in the transverse directions
    offset = np.zeros((3,))
    offset[: len(transverse_dirs)] = kpts_shift[transverse_dirs]
    bzk_t_kc = monkhorst_pack(tuple(kpts_grid[transverse_dirs]) + (1,)) + offset

    return (bzk_p_kc, weight_p_k), (bzk_t_kc,)


def get_realspace_hs(
    h_skmm,
    s_kmm,
    bzk_kc,
    weight_k,
    R_c=(0, 0, 0),
    direction="x",
    symmetry={"enabled": False},
):

    from gpaw.symmetry import Symmetry
    from ase.dft.kpoints import get_monkhorst_pack_size_and_offset, monkhorst_pack

    if symmetry["point_group"]:
        raise NotImplementedError("Point group symmetry not implemented")

    nspins, nk, nbf = h_skmm.shape[:3]
    dir = "xyz".index(direction)
    transverse_dirs = np.delete([0, 1, 2], [dir])
    dtype = float
    if len(bzk_kc) > 1 or np.any(bzk_kc[0] != [0, 0, 0]):
        dtype = complex

    kpts_grid, kpts_shift = get_monkhorst_pack_size_and_offset(bzk_kc)

    # kpts in the transport direction
    nkpts_p = kpts_grid[dir]
    bzk_p_kc = monkhorst_pack((nkpts_p, 1, 1))[:, 0] + kpts_shift[dir]
    weight_p_k = 1.0 / nkpts_p

    # kpts in the transverse directions
    offset = np.zeros((3,))
    offset[: len(transverse_dirs)] = kpts_shift[transverse_dirs]
    bzk_t_kc = monkhorst_pack(tuple(kpts_grid[transverse_dirs]) + (1,)) + offset
    if "time_reversal" not in symmetry:
        symmetry["time_reversal"] = True
    if symmetry["time_reversal"]:
        # XXX a somewhat ugly hack:
        # By default GPAW reduces inversion sym in the z direction. The steps
        # below assure reduction in the transverse dirs.
        # For now this part seems to do the job, but it may be written
        # in a smarter way in the future.
        symmetry = Symmetry([1], np.eye(3))
        symmetry.prune_symmetries_atoms(np.zeros((1, 3)))
        ibzk_kc, ibzweight_k = symmetry.reduce(bzk_kc)[:2]
        ibzk_t_kc, weights_t_k = symmetry.reduce(bzk_t_kc)[:2]
        ibzk_t_kc = ibzk_t_kc[:, :2]
        nkpts_t = len(ibzk_t_kc)
    else:
        ibzk_kc = bzk_kc.copy()
        ibzk_t_kc = bzk_t_kc
        nkpts_t = len(bzk_t_kc)
        weights_t_k = [1.0 / nkpts_t for k in range(nkpts_t)]

    h_skii = np.zeros((nspins, nkpts_t, nbf, nbf), dtype)
    if s_kmm is not None:
        s_kii = np.zeros((nkpts_t, nbf, nbf), dtype)

    tol = 7
    for j, k_t in enumerate(ibzk_t_kc):
        for k_p in bzk_p_kc:
            k = np.zeros((3,))
            k[dir] = k_p
            k[transverse_dirs] = k_t
            kpoint_list = [list(np.round(k_kc, tol)) for k_kc in ibzk_kc]
            if list(np.round(k, tol)) not in kpoint_list:
                k = -k  # inversion
                index = kpoint_list.index(list(np.round(k, tol)))
                h = h_skmm[:, index].conjugate()
                if s_kmm is not None:
                    s = s_kmm[index].conjugate()
                k = -k
            else:  # kpoint in the ibz
                index = kpoint_list.index(list(np.round(k, tol)))
                h = h_skmm[:, index]
                if s_kmm is not None:
                    s = s_kmm[index]

            c_k = np.exp(2.0j * np.pi * np.dot(k, R_c)) * weight_p_k
            h_skii[:, j] += c_k * h
            if s_kmm is not None:
                s_kii[j] += c_k * s

    if s_kmm is None:
        return ibzk_t_kc, weights_t_k, h_skii
    else:
        return ibzk_t_kc, weights_t_k, h_skii, s_kii
