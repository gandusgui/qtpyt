import numpy as np


class Plane:
    """Class for a simple plane in 3D.
    
    Args:
        p : A point in the plane.
        n : Vector normal to the plane
    """

    def __init__(self, p, n):
        self.p = np.asarray(p)
        self.n = np.asarray(n) / np.linalg.norm(n)
        self.d = -np.inner(p, n)
        self.norm2 = np.inner(n, n)

    def mirror(self, positions):
        """Mirror 3D coordinates.
        
        Example:
            In [1]: plane = Plane((0,0,0),(1,0,0))
            In [2]: positions = np.squeeze(np.indices((3,1,1)).T)
            In [3]: positions
            Out[3]: 
            array([[0, 0, 0],
                   [1, 0, 0],
                   [2, 0, 0]])
            In [4]: plane.mirror(positions)
            Out[4]: 
            array([[ 0.,  0.,  0.],
                   [-1.,  0.,  0.],
                   [-2.,  0.,  0.]])
        """
        p1 = np.atleast_2d(positions)
        k = -(np.dot(p1, self.n) + self.d) / self.norm2
        p2 = (self.n[None, :] * k[:, None]) + p1
        return np.squeeze(2 * p2 - p1)
