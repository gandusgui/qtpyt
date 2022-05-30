import numbers

import ase.neighborlist
import numpy as np
#### NeighborList
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def get_neighbors(atoms):
    """Get neighbors list for each atom.
    
    """
    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    nlists = []
    for a in atoms:
        nlists.append(nl.get_neighbors(a.index)[0])

    return nlists


def argsort(positions, order="xyz"):
    i, j, k = ["xyz".index(i) for i in order]
    idx = np.lexsort((positions[:, k], positions[:, j], positions[:, i]))
    return idx


class Basis:
    """Basis function descriptor.
    
    Example :
    In [1]: from ase import Atoms
    In [2]: from qtpyt.surface.basis import Basis
    In [2]: 
    In [2]: atoms = Atoms('OH',positions=[[0,0,0],[0.5,0.5,0.5]],cell=[1,1,1])
    In [3]: basis = Basis.from_dictionary(atoms, {'O':2,'H':1})
        
    """

    def __init__(self, atoms, nao_a, M_a=None):
        """Build descriptor from atoms and # of atomic orbitals
        per atom nao.

        """
        self.atoms = atoms
        self.nao_a = np.array(nao_a, ndmin=1)
        if M_a is None:
            M_a = np.cumsum(np.insert(self.nao_a[:-1], 0, 0))
        self.M_a = np.array(M_a, ndmin=1)
        self.nao = np.sum(self.nao_a)  # Total # of atomic orbitals

    @classmethod
    def from_dictionary(cls, atoms, basis):
        """Build descriptor from atoms and dictionary of key value
        paris {symbol : nao}.

        """
        nao_a = [basis[symbol] for symbol in atoms.symbols]
        return cls(atoms, nao_a)

    @property
    def centers(self):
        """Expand atomic centers to basis functions.
        
        Example:
        In [1]: basis.atoms.positions
        Out[1]: 
        array([[0. , 0. , 0. ],
            [0.5, 0.5, 0.5]])

        In [2]:
        In [2]: basis.centers
        Out[2]: 
        array([[0. , 0. , 0. ],
            [0. , 0. , 0. ],
            [0.5, 0.5, 0.5]])


        """
        return self._expand(self.atoms.positions)

    def repeat(self, N):
        """Create a new repeated basis object.

        Args:
            N : sequence of three positives.

        Example:
        In [1]: basis.nao_a
        Out[1]: array([2, 1])

        In [2]: basis.repeat((1,2,1)).nao_a
        Out[2]: array([2, 1, 2, 1])

        """
        nao_a = np.tile(self.nao_a, np.prod(N))
        atoms = self.atoms.repeat(N)
        return Basis(atoms, nao_a)

    def argsort(self, order="xyz"):
        """Sort basis indices using corresponding atomic positions.
        
        The first character is used as primary coordinate and so on.
        NOTE : Opposite to np.lexsort

        Example:
        In [2]: basis.nao_a
        Out[2]: array([2, 1, 2, 1])

        In [38]: basis.atoms.positions
        Out[38]: 
        array([[0. , 0. , 0. ],
            [0.5, 0.5, 0.5],
            [0. , 1. , 0. ],
            [0.5, 1.5, 0.5]])

        In [39]: basis.atoms.positions[basis.argsort()]
        Out[39]: 
        array([[0. , 0. , 0. ],
            [0. , 1. , 0. ],
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5]])

        In [41]: basis.nao_a[basis.argsort()]
        Out[41]: array([2, 2, 1, 1])

        """
        return argsort(self.atoms.positions, order)

    def get_indices(self, indices=None):
        """Get basis function indices.
        
        Args:
            indices : (optional) take subset/sorted atomic indices.
        """
        if indices is None:
            M_a, nao_a, nao = self.M_a, self.nao_a, self.nao
        elif isinstance(indices, numbers.Integral):
            return np.arange(self.nao_a[indices]) + self.M_a[indices]
        else:
            M_a = self.M_a[indices]
            nao_a = self.nao_a[indices]
            nao = np.sum(nao_a)

        idx = self._expand(M_a, nao_a)
        i0 = M_a[0]
        for i in range(1, nao):
            if idx[i] == i0:
                idx[i] = idx[i - 1] + 1
            else:
                i0 = idx[i]
        return idx

    def take(self, basis_dict):
        """Take basis function indices for subset of elements' kind.
        
        Args:
            basis_dict : dictionary of element keys and indices value(s). 
                { 'O' : [3, .. ], ..  'C' : 2, .. }    
        """
        idxlist = []
        for kind, idx in basis_dict.items():
            if isinstance(idx, numbers.Integral):
                idx = [idx]
            for a in self.atoms.symbols.search(kind):
                for i in idx:
                    idxlist.append(self.M_a[a] + i)
        idxlist = np.asarray(idxlist)
        idxlist.sort()
        return idxlist

    def extract(self):
        """Extract a subset of the atoms starting from zero index."""
        return self.__class__(self.atoms, self.nao_a)

    def __getitem__(self, i):
        """Return a subset of the atoms."""
        return self.__class__(self.atoms[i], self.nao_a[i], self.M_a[i])

    def __len__(self):
        return self.nao

    def _expand(self, array, repeats=None):
        """Helper function. Expand atoms array to basis functions."""
        if repeats is None:
            repeats = self.nao_a
        return np.repeat(array, repeats, axis=0)
