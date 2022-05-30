import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg


def sort_eigs(alpha, betha, Z, h_l, h_r, flag):
    """
    The function provides classification and sorting of the eigenvalues
    and eigenvectors. The sorting procedure is described in
    [M. Wimmer, Quantum transport in nanostructures: From computational concepts to
    spintronics in graphene and magnetic tunnel junctions, 2009, ISBN-9783868450255]
    The algorithm puts left-propagating solutions into upper-left block and right-propagating
    solutions - into the lower-right block.
    The sort is performed by absolute value of the eigenvalues. For those whose
    absolute value equals one, classification is performed computing their group velocity
    using eigenvectors. The function also orthogonalizes the eigenvectors contained in the matrix Z
    that correspond to degenerate eigenvalues.
    :param alpha:
    :param betha:
    :param Z:
    :param h_l:
    :param h_r:
    :return:
    """

    margin = 1e-3
    angle_margin = 1e-4
    ans = []

    eigenvals = alpha / betha

    inds = np.where(
        (np.abs(eigenvals) < 1.0 + margin) * (np.abs(eigenvals) > 1.0 - margin)
    )[0]
    inds_deg = []

    for j, j1 in enumerate(inds[:-1]):
        inds_deg.append([j1])
        for j2 in inds[j + 1 :]:
            if (
                np.abs(np.real(eigenvals[j1] - eigenvals[j2])) < angle_margin
                and np.abs(np.imag(eigenvals[j1] - eigenvals[j2])) < angle_margin
            ):
                inds_deg[j].append(j2)

    inds_deg = [i for i in inds_deg if len(i) > 1]

    for item in inds_deg:
        phi = np.matrix(Z[h_r.shape[0] :, item])
        oper = 1j * np.matrix(
            h_r * eigenvals[item[0]] - h_l * np.conj(eigenvals[item[0]])
        )
        _, vects = np.linalg.eig(phi.H * oper * phi)
        print(vects)
        Z[h_r.shape[0] :, item] = phi * np.matrix(vects)

    phi = np.matrix(Z)

    for j, item in enumerate(zip(alpha, betha)):

        if np.abs(item[1]) != 0.0:
            eigenval = item[0] / item[1]
        else:
            eigenval = 1e21

        if np.abs(eigenval) > 1.0 + margin:
            ans.append(False)
        elif np.abs(eigenval) < 1.0 - margin:
            ans.append(True)
        else:
            gv = np.imag(
                2 * phi[h_r.shape[0] :, j].H * np.matrix(h_r) * phi[h_r.shape[0] :, j]
            )
            # gv = group_velocity(phi[h_r.shape[0]:, j], eigenval, h_r)

            if flag:
                gv = group_velocity(phi[h_r.shape[0] :, j], np.conj(eigenval), h_r)

            print("Group velocity is ", gv, np.angle(eigenval))
            if gv > 0:
                ans.append(False)
            else:
                ans.append(True)

    return ans


def iterate_gf(E, h_0, h_l, h_r, gf, num_iter):
    for j in range(num_iter):
        gf = h_r * np.linalg.pinv(E * np.identity(h_0.shape[0]) - h_0 - gf) * h_l

    return gf


def surface_greens_function_poles(E, h_l, h_0, h_r):
    """
    Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here, the energy E is a parameter, and the eigenvalues correspond to wave vectors as `exp(ik)`.
    :param E:     energy
    :type E:      float
    :param h_l:   left block of three-block-diagonal Hamiltonian
    :param h_0:   central block of three-block-diagonal Hamiltonian
    :param h_r:   right block of three-block-diagonal Hamiltonian
    :return:      eigenvalues, k, and eigenvectors, U,
    :rtype:       numpy.matrix, numpy.matrix
    """

    main_matrix = np.block(
        [
            [np.zeros(h_0.shape), np.identity(h_0.shape[0])],
            [-h_l, E * np.identity(h_0.shape[0]) - h_0],
        ]
    )

    overlap_matrix = np.block(
        [[np.identity(h_0.shape[0]), np.zeros(h_0.shape)], [np.zeros(h_0.shape), h_r]]
    )

    alpha, betha, _, eigenvects, _, _ = linalg.lapack.cggev(main_matrix, overlap_matrix)

    eigenvals = np.zeros(alpha.shape, dtype=np.complex128)

    for j, item in enumerate(zip(alpha, betha)):

        if np.abs(item[1]) != 0.0:
            eigenvals[j] = item[0] / item[1]
        else:
            eigenvals[j] = 1e10

    # sort absolute values
    ind = np.argsort(np.abs(eigenvals))
    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]

    vals = np.copy(eigenvals)
    roll = np.where((0.999 < np.abs(vals)) * (np.abs(vals) < 1.001))

    if np.size(roll) != 0:
        roll = h_0.shape[0] - int(round(np.mean(roll)))
        if roll != 0:
            print(roll)
            eigenvals = np.roll(eigenvals, roll)
            eigenvects = np.roll(eigenvects, roll, axis=1)
            eigenvals[: abs(roll)] = 0.0
            eigenvals[-abs(roll) :] = 1e10
            vals = np.copy(eigenvals)

    mask1 = np.abs(vals) < 0.999
    mask2 = np.abs(vals) > 1.001
    vals = np.angle(vals)

    vals[mask1] = -5
    vals[mask2] = 5
    ind = np.argsort(vals, kind="mergesort")

    eigenvals = eigenvals[ind]
    eigenvects = eigenvects[:, ind]

    eigenvects = eigenvects[h_0.shape[0] :, :]
    eigenvals = np.matrix(np.diag(eigenvals))
    eigenvects = np.matrix(eigenvects)

    norms = linalg.norm(eigenvects, axis=0)
    norms = np.array([1e30 if np.abs(norm) < 0.000001 else norm for norm in norms])
    eigenvects = eigenvects / norms[np.newaxis, :]

    return eigenvals, eigenvects


def surface_greens_function_poles_Shur(E, h_l, h_0, h_r):
    """
    Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here, the energy E is a parameter, and the eigenvalues correspond to wave vectors as `exp(ik)`.
    :param E:     energy
    :type E:      float
    :param h_l:   left block of three-block-diagonal Hamiltonian
    :param h_0:   central block of three-block-diagonal Hamiltonian
    :param h_r:   right block of three-block-diagonal Hamiltonian
    :return:      eigenvalues, k, and eigenvectors, U,
    :rtype:       numpy.matrix, numpy.matrix
    """

    main_matrix = np.block(
        [
            [np.zeros(h_0.shape), np.identity(h_0.shape[0])],
            [-h_l, E * np.identity(h_0.shape[0]) - h_0],
        ]
    ).astype(np.complex64)

    overlap_matrix = np.block(
        [[np.identity(h_0.shape[0]), np.zeros(h_0.shape)], [np.zeros(h_0.shape), h_r]]
    ).astype(np.complex64)

    sort = lambda a, b, z: sort_eigs(a, b, z, h_l, h_r, False)

    AA, BB, alpha, betha, eigv_right, eigv_left = linalg.ordqz(
        main_matrix, overlap_matrix, output="complex", sort=sort
    )

    main_matrix1 = np.block(
        [
            [np.zeros(h_0.shape), np.identity(h_0.shape[0])],
            [-h_r, E * np.identity(h_0.shape[0]) - h_0],
        ]
    ).astype(np.complex64)

    overlap_matrix1 = np.block(
        [[np.identity(h_0.shape[0]), np.zeros(h_0.shape)], [np.zeros(h_0.shape), h_l]]
    ).astype(np.complex64)

    sort1 = lambda a, b, z: sort_eigs(a, b, z, h_r, h_l, True)

    AA1, BB1, alpha1, betha1, eigv_right1, eigv_left1 = linalg.ordqz(
        main_matrix1, overlap_matrix1, output="complex", sort=sort1
    )

    eigv_left = np.matrix(eigv_left)
    eigv_left1 = np.matrix(eigv_left1)

    return (
        h_r
        * eigv_left[h_0.shape[0] :, : h_0.shape[0]]
        * np.linalg.pinv(eigv_left[: h_0.shape[0], : h_0.shape[0]]),
        h_l
        * eigv_left1[h_0.shape[0] :, : h_0.shape[0]]
        * np.linalg.pinv(eigv_left1[: h_0.shape[0], : h_0.shape[0]]),
    )


def group_velocity(eigenvector, eigenvalue, h_r):
    """
    Computes the group velocity of wave packets from their wave vectors
    :param eigenvector:
    :param eigenvalue:
    :param h_r:
    :return:
    """

    return np.imag(eigenvector.H * h_r * eigenvalue * eigenvector)


def surface_greens_function(E, h_l, h_0, h_r):
    """
    Computes eigenvalues and eigenvectors for the complex band structure problem.
    Here energy E is a parameter, adn the eigenvalues are wave vectors.
    :param E:     energy
    :type E:      float
    :param h_l:   left block of three-block-diagonal Hamiltonian
    :param h_0:   central block of three-block-diagonal Hamiltonian
    :param h_r:   right block of three-block-diagonal Hamiltonian
    :return:      surface Green's function, for left and right sides
    """

    vals, vects = surface_greens_function_poles(E, h_l, h_0, h_r)
    vals = np.diag(vals)

    u_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    u_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_right = np.matrix(np.zeros(h_0.shape, dtype=np.complex))
    lambda_left = np.matrix(np.zeros(h_0.shape, dtype=np.complex))

    alpha = 0.001

    for j in range(h_0.shape[0]):
        if np.abs(vals[j]) > 1.0 + alpha:

            lambda_left[j, j] = vals[j]
            u_left[:, j] = vects[:, j]

            lambda_right[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
            u_right[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

        elif np.abs(vals[j]) < 1.0 - alpha:
            lambda_right[j, j] = vals[j]
            u_right[:, j] = vects[:, j]

            lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
            u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

        else:

            gv = group_velocity(vects[:, j], vals[j], h_r)
            # ind = np.argmin(np.abs(np.angle(vals[h_0.shape[0]:]) + np.angle(vals[j])))
            print("Group velocity is ", gv, np.angle(vals[j]))

            if gv > 0:

                lambda_left[j, j] = vals[j]
                u_left[:, j] = vects[:, j]

                lambda_right[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                u_right[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

            else:
                lambda_right[j, j] = vals[j]
                u_right[:, j] = vects[:, j]

                lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
                u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

            # lambda_right[j, j] = vals[j]
            # u_right[:, j] = vects[:, j]
            #
            # lambda_left[j, j] = vals[-j + 2 * h_0.shape[0] - 1]
            # u_left[:, j] = vects[:, -j + 2 * h_0.shape[0] - 1]

    sgf_l = h_r * u_right * lambda_right * np.linalg.pinv(u_right)
    sgf_r = h_l * u_left * lambda_right * np.linalg.pinv(u_left)

    # sgf_l = u_right[h_0.shape[0]:, :] * np.linalg.pinv(u_right[:h_0.shape[0], :])
    # sgf_r = h_l * u_left * lambda_right * np.linalg.pinv(u_left)

    return (
        iterate_gf(E, h_0, h_l, h_r, sgf_l, 2),
        iterate_gf(E, h_0, h_r, h_l, sgf_r, 2),
        lambda_right,
        lambda_left,
        vals,
    )

    # return h_r * u_right * lambda_right * np.linalg.pinv(u_right), \
    #        h_l * u_left * lambda_right * np.linalg.pinv(u_left), \
    #        lambda_right, lambda_left, vals
