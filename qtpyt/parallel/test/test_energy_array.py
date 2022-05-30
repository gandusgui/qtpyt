import numpy as np
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import DistGrid, GridDesc
from qtpyt.parallel.pencil import _blockdist


def test_energy_array():

    # np.random.seed(2)
    neners, norbs = 20, 1  # np.random.randint(0, 50, 2)

    diste = DistGrid(neners, norbs, dtype=float)

    a = diste.empty_aligned_orbs()
    b = diste.empty_aligned_eners()
    c = diste.empty_aligned_orbs()
    a[:] = comm.rank
    diste.collect_energies(a, b)
    diste.collect_orbitals(b, c)

    np.testing.assert_allclose(a, c)


def test_energy_grid():
    energies = np.arange(30)

    gd = GridDesc(energies, 3, dtype=int)
    # print(gd.energies)


def test_gather_energies():
    energies = np.arange(30)

    gd = GridDesc(energies, 3, dtype=complex)
    a = np.empty_like(gd.energies)
    a[:] = comm.rank
    b = gd.gather_energies(a)
    if comm.rank == 0:
        desired = np.empty_like(b)
        for r in range(comm.size):
            n, s = _blockdist(energies.size, comm.size, r)
            desired[s : s + n] = r
        np.testing.assert_allclose(b, desired)


def test_sum_energies():
    energies = np.arange(30)

    gd = GridDesc(energies, 3, dtype=complex)
    a = gd.empty_aligned_orbs()
    a[:] = 1 + 1.0j
    b = gd.sum_energies(a)
    c = np.ones((30, 3, 3)) + 1.0j * np.ones((30, 3, 3))
    desired = np.trapz(c, dx=energies[1] - energies[0], axis=0)
    np.testing.assert_allclose(b, desired)


def test_energy_grid_write():
    energies = np.arange(30)

    gd = GridDesc(energies, 3, dtype=int)
    a = gd.empty_aligned_orbs()
    n, s = _blockdist(30, comm.size, comm.rank)
    for i in range(n):
        a[i] = i + s
    gd.write(a, "tmp.bin", (6 < energies) & (energies < 13))
    if comm.rank == 0:
        actual = np.fromfile("tmp.bin", dtype=int).reshape(6, 3, 3)
        desired = np.array(
            [
                [[7, 7, 7], [7, 7, 7], [7, 7, 7]],
                [[8, 8, 8], [8, 8, 8], [8, 8, 8]],
                [[9, 9, 9], [9, 9, 9], [9, 9, 9]],
                [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
                [[11, 11, 11], [11, 11, 11], [11, 11, 11]],
                [[12, 12, 12], [12, 12, 12], [12, 12, 12]],
            ]
        )
        np.testing.assert_allclose(actual, desired)


if __name__ == "__main__":
    test_energy_array()
    test_energy_grid()
    test_gather_energies()
    test_sum_energies()
    test_energy_grid_write()
