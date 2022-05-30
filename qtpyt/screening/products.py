import numpy as np
from qtpyt import xp


def polarization_product(G, Gd, pre=1.0, out=None, rot=None, work1=None, work2=None):
    """Perform kronecker matrix product (G otimes Gd).

    If rot is specified, it is used to rotate the output, such that
    rot^dag . out . rot is returned instead of out

    Args:

        G: ndarray
            input matrix of shape Ni x Ni
        Gd: ndarray
            input matrix of shape Ni x Ni

    Optional:
        
        pre: float or complex (default: 1.0)
            the output is multiplied by this prefactor
        rot: ndarray (default: None)
            if specified, this is used to rotate output.
            shape is Ni^2 x Nq
        rotd: ndarray (default: None)
            dagger(rot) if rot is specified.
            shape is Nq x Ni^2
        out: ndarray (default None)
            if specified, this is used for the output array, otherwise it is
            created. A reference to out is always returned
        work: ndarray (default None)
            used when rot is given.
            shape is Ni^2 x Nq

        NOTE : Nq is 
                i) Ni^2 when rot is None
                ii) # of pair orbitals when rot is given.
                
    If rot is None, the out array generally has the elements::

      out_ij,kl = pre * G_ik Gd_jl

    Else::
    
      out_qq' = pre * sum_ijkl Ud_q,ij G_ik Gd_jl U_kl,q'
      
    NOTE: Matrices can be on GPU.
    """
    #
    #                  -->--
    #              i  |     | k
    #  U       (t,r)   xxxxx   (t',r')   U
    #   q,ij       j  |     | l           kl,q'
    #                  --<--
    #
    #                +
    #          G   G    =  G   Gd
    #           ik  lj      ik   jl
    #
    #
    Ni = len(G)
    if rot is None:
        if out is None:
            out = xp.zeros((Ni ** 2, Ni ** 2), dtype=G.dtype)
        # out[:] = pre * xp.kron(G, Gd)
        out_ijkl = out.reshape(Ni, Ni, Ni, Ni)
        for i, k in np.ndindex(Ni, Ni):
            out_ijkl[i, :, k, :] = G[i, k] * Gd
    else:
        Nq = rot.shape[1]

        if out is None:
            out = xp.zeros((Nq, Nq), dtype=G.dtype)

        # Temporary work arrays
        if work1 is None:
            work1 = xp.zeros_like(rot)
        if work2 is None:
            work2 = work1.copy()

        # Contiguous views with packed format unfolded, p -> ii
        work1 = work1.reshape(Ni, Ni, Nq)
        work2 = work2.reshape(Ni, Ni * Nq)
        rot = rot.reshape(Ni, Ni, Nq)

        # work1_kj,q' = sum_l B_jl rot_kl,q'
        for work1_iq, U_iq in zip(work1, rot):
            Gd.dot(U_iq, out=work1_iq)

        # work2_ij,q' = sum_k A_ik work1_kj,q'
        G.dot(work1.reshape(Ni, -1), out=work2)
        # work2 = xp.tensordot(G,work1,axes=([1],[0]))

        # out_qq' = sum_ij rot^dag_q,ij work2_ij,q'
        # rotd.dot(work2.reshape(-1, Nq), out=out)
        rot.reshape(-1, Nq).T.dot(work2.reshape(-1, Nq), out=out)

    out *= pre
    return out


def exchange_product(G, V, pre=1.0, out=None, rot=None, work1=None, work2=None):
    """Perform matrix product sum_kl G_kl V_ik,jl

    If rot is specified, it is used to rotate V, such that
    rot . V . rot^dag is used instead of V.

    Arguments:

    G: ndarray
       input matrix of shape Ni x Ni
    V: ndarray
       input matrix of shape Nq x Nq
       
    NOTE : see definition Nq above.

    Optional parameters:
    
    pre: float or complex (default: 1.0)
         the output is multiplied by this prefactor
    rot: ndarray, len(G)**2 x len(V) (default: None)
         if specified, V is replaced by rot . V . rot^dag
    out: ndarray (default None)
         if specified, this is used for the output array, otherwise it is
         created. A reference to out is always returned
    work: ndarray (default None)
            used when rot is given.
            shape is Ni^2 x Nq

    If rot is None, the out array generally has the elements::

      out_ij = pre * sum_kl G_kl V_ik,jl

    Else::
    
      out_ij = pre * sum_klqq' G_kl rot_ik,q V_qq' (rot^dag)_q',jl
    """
    assert G.dtype == V.dtype
    if out is None:
        out = xp.zeros_like(G)

    Ni = len(G)
    if rot is None:
        V_ikjl = V.reshape(Ni, Ni, Ni, Ni)
        # out = np.tensordot(G, V, axes=([1,2],[0,1]))
        for i, k in xp.ndindex(Ni, Ni):
            out[i] += V_ikjl[i, k].dot(G[k])
    else:
        Nq = len(V)
        if work1 is None:
            work1 = xp.zeros(Ni ** 2 * Nq, dtype=G.dtype)
        if work2 is None:
            work2 = xp.zeros(Ni ** 2 * Nq, dtype=G.dtype)

        # work2_q,jl = sum_q' V_qq' rot^dag_q',jl
        V.dot(rot.T, out=work2.reshape(Nq, -1))

        # work1_k,qj = sum_l A_kl work2_qj,l
        G.dot(work2.reshape(-1, Ni).T, out=work1.reshape(Ni, -1))

        # out_ij = sum_kq rot_i,kq work1_kq,j
        rot.reshape(Ni, -1).dot(work1.reshape(-1, Ni), out=out)

    out *= pre
    return out


def hartree_product(G, V, pre=1.0, out=None, rot=None, work1=None, work2=None):
    """Perform matrix product sum_kl G_kl V_ij,kl
    
    If rot is specified, it is used to rotate V, such that
    rot . V . rot^dag is used instead of V.

    Arguments:

    G: ndarray
       input matrix of shape Ni x Ni
    V: ndarray
       input matrix of shape Ni**2 x Ni**2

    Optional parameters:
    
    pre: float or complex (default: 1.0)
         the output is multiplied by this prefactor
    rot: ndarray (default: None)
         if specified, V is replaced by rot . V . rot^dag
         shape is Ni^2 x Nq
    out: ndarray (default None)
         if specified, this is used for the output array, otherwise it is
         created. A reference to out is always returned
    work: ndarray (default None)
         used when rot is given.
         shape is Ni^2 x Nq

    If rot is None, the out array generally has the elements::

      out_ij = pre * sum_kl G_kl V_ij,kl

    Else::
    
      out_ij = pre * sum_klqq' G_kl rot_ij,q V_qq' (rot^dag)_q',kl
    """
    assert G.dtype == V.dtype
    if out is None:
        out = np.zeros_like(G)

    Ni = len(G)
    if rot is None:
        V.dot(G.reshape(-1), out=out.reshape(-1))
    else:
        Nq = len(V)
        if work1 is None:
            work1 = np.zeros(Nq, dtype=G.dtype)
        if work2 is None:
            work2 = np.zeros(Nq, dtype=G.dtype)

        # work1_q = sum_kl rot^conj_q,kl G_kl
        rot.T.dot(G.reshape(-1), out=work1)

        # work2_q = sum_q' V_qq' work1_q'
        V.dot(work1, out=work2)

        # out_ij = sum_q rot_ij,q work2_q
        rot.dot(work2, out=out.reshape(-1))

    out *= pre
    return out
