#!/usr/bin/env python

import numpy as np
from pathlib import Path

from scipy.sparse import csr_matrix, coo_matrix
from ase.units import Hartree

def read_bincsr_matrix(filename, dtype=np.float64):
    dt = np.dtype([
        ('o1', np.uint32),
        ('x', np.uint32),
        ('y', np.uint32),
        ('real', dtype),
        ('imag', dtype),
        ('o2', np.uint32)])
    
    coo = np.fromfile(filename, dt, offset=0)
    data = coo['real'] + 1.j * coo['imag']
    mat = coo_matrix((data,(coo['x']-1,coo['y']-1))).toarray()
    tri2full(mat,UL='U')
    return mat


cc_path = Path('/usr/scratch/mont-fort18/ggandus/k-point/sc')


get_ipython().system('egrep BRILLOUIN /usr/scratch/mont-fort18/ggandus/k-point/sc/energy.out')


# ## PREPROCESS SCATTERING
# 
#     i) read in reversed k-points (convention uses +$k_x$)
#     ii) transpose matrix A[k]=A[-k].T

h_cc_k_files = [cc_path/f'sc-KS_SPIN_1_K_{i}-1_0.csr' for i in range(8,0,-1)]
s_cc_k_files = [cc_path/f'sc-S_SPIN_1_K_{i}-1_0.csr' for i in range(8,0,-1)]


read_bincsr = lambda file: read_bincsr_matrix(file).T.copy('C')
h_cc_k = np.stack(list(map(read_bincsr, h_cc_k_files)))
s_cc_k = np.stack(list(map(read_bincsr, s_cc_k_files)))

plt.spy(h_cc_k[-1])

atoms_cc = read(cc_path/'sc.xyz')
basis_cc = basis.Basis.from_dictionary(atoms_cc, {'C':13,'Si':13,'O':13})

get_ipython().system('egrep -i fermi /usr/scratch/mont-fort18/ggandus/k-point/sc/energy.out')

fermi_cc = 0.26936888972782 * Hartree
fermi_cc

h_cc_k *= Hartree
h_cc_k -= fermi_cc * s_cc_k

kpts_t, h_cc_kii, s_cc_kii, h_cc_kij, s_cc_kij = tools.prepare_leads_matrices(basis_cc, h_cc_k, s_cc_k, (4,4,1))

del h_cc_k
del s_cc_k


# In[24]:


kpts_t


# ### SAVE

# In[ ]:


np.save('hs_cc_ii_ij',(h_cc_kii, s_cc_kii, h_cc_kij, s_cc_kij))


# In[22]:


plt.spy(h_cc_kij[0])


# In[23]:


pl_path = cc_path.parent/'uc'#Path('./uc/')


# In[24]:


ls ./uc/


# In[25]:


get_ipython().system('egrep BRILLOUIN ./uc/energy.out')


# ## PREPROCESS LEADS
# 
#     i) read in reversed k-points (convention uses +$k_x$)
#     ii) transpose matrix A[k]=A[-k].T

# In[26]:


pl_path


# In[27]:


h_pl_k_files = [pl_path/f'sc-KS_SPIN_1_K_{i}-1_0.csr' for i in range(48,0,-1)]
s_pl_k_files = [pl_path/f'sc-S_SPIN_1_K_{i}-1_0.csr' for i in range(48,0,-1)]


# In[28]:


read_bincsr = lambda file: read_bincsr_matrix(file).T.copy('C')
h_pl_k = np.stack(list(map(read_bincsr, h_pl_k_files)))
s_pl_k = np.stack(list(map(read_bincsr, s_pl_k_files)))


# In[29]:


plt.spy(h_pl_k[-1])


# In[30]:


atoms_pl = read(pl_path/'pl.xyz')
basis_pl = basis.Basis.from_dictionary(atoms_pl, {'C':13,'Si':13,'O':13})


# In[31]:


pl_path/'energy.out'


# In[32]:


get_ipython().system('egrep -i fermi /usr/scratch/mont-fort18/ggandus/k-point/uc/energy.out')


# In[33]:


fermi_pl = 0.33495567601659 * Hartree
fermi_pl


# In[34]:


h_pl_k *= Hartree
h_pl_k -= fermi_pl * s_pl_k


# In[35]:


kpts_t, h_pl_kii, s_pl_kii, h_pl_kij, s_pl_kij = tools.prepare_leads_matrices(basis_pl, h_pl_k, s_pl_k, (6,8,2))


# In[25]:


del h_pl_k
del s_pl_k


# In[26]:


kpts_t


# ### SAVE

# In[27]:


np.save('hs_pl_ii_ij',(h_pl_kii, s_pl_kii, h_pl_kij, s_pl_kij))


# In[36]:


plt.spy(h_pl_kij[0])


# # END OF PREPROCESSING

# In[67]:


get_ipython().system('hostname')


# In[68]:


get_ipython().system('pwd')


# In[ ]:


# %load trans.py
#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt


from qtpyt.abc import greenfunction, selfenergy
from qtpyt.surface import basis, principallayer, kpts


from ase.io import read


# In[ ]:


atoms_cc = read('sc.xyz')
basis_cc = basis.Basis.from_dictionary(atoms_cc, {'C':13,'Si':13,'O':13})

atoms_pl = read('pl.xyz')
basis_pl = basis.Basis.from_dictionary(atoms_pl, {'C':13,'Si':13,'O':13})

h_cc_kii, s_cc_kii, h_cc_kij, s_cc_kij = np.load('hs_cc_ii_ij.npy')
h_pl_kii, s_pl_kii, h_pl_kij, s_pl_kij = np.load('hs_pl_ii_ij.npy')


# In[50]:


kpts_t = kpts.monkhorst_pack((1,8,2))
Kpts = kpts.monkhorst_pack((1,4,2), offset=(0,0,0))
k_groups = kpts.unfold_kpts(Kpts, kpts_t)


# In[51]:


kpts_t


# In[52]:


Kpts


# In[54]:


k_groups = np.array(k_groups).reshape(-1,4)


# In[55]:


k_groups


# In[58]:


Kpts = kpts.monkhorst_pack((1,4,1), offset=(0,0,0))
Kpts


# In[49]:


Nr = (1,2,2) # # of realspace PLs in SC
Nk = (1,4,1) # # of SC transverse k-points


# In[ ]:


energies = np.linspace(-5,5,500,endpoint=True)
T_e = np.zeros_like(energies)


# In[ ]:


for i in range(np.prod(Nk)):

    selfenergies = [None, None]
    selfenergies[0] = principallayer.PrincipalSelfEnergy(kpts_t[k_groups[i]], (h_pl_kii[k_groups[i],...], s_pl_kii[k_groups[i],...]), (h_pl_kij[k_groups[i],...], s_pl_kij[k_groups[i],...]), Nr=Nr)
    selfenergies[1] = principallayer.PrincipalSelfEnergy(kpts_t[k_groups[i]], (h_pl_kii[k_groups[i],...], s_pl_kii[k_groups[i],...]), (h_pl_kij[k_groups[i],...], s_pl_kij[k_groups[i],...]), id='right', Nr=Nr)

    align_ao = basis_pl.nao
    diff = (h_cc_kii[i][align_ao, align_ao] - selfenergies[0].h_ii[align_ao, align_ao]) / s_cc_kii[i][align_ao, align_ao]
    h_cc_kii[i] -= diff * s_cc_kii[i]
    h_cc_kij[i] -= diff * s_cc_kij[i]

    internalselfenergies = [None, None]
    indices = np.ix_(range(selfenergies[0].nbf_i),range(selfenergies[0].nbf_i))
    internalselfenergies[0] = selfenergy.SelfEnergy((h_cc_kii[i], s_cc_kii[i]), (h_cc_kij[i], s_cc_kij[i]), [(indices, selfenergies[0])])
    internalselfenergies[1] = selfenergy.SelfEnergy((h_cc_kii[i], s_cc_kii[i]), (h_cc_kij[i], s_cc_kij[i]), [(indices, selfenergies[1])], id='right')

    indices = slice(None)
    gf = greenfunction.GreenFunction(internalselfenergies[0].H, internalselfenergies[0].S, [(slice(None), internalselfenergies[0]), (slice(None), internalselfenergies[1])])

    for e, energy in enumerate(energies):
        T_e[e] += gf.get_transmission(energy)
#         T_e[e] += gf.get_pdos(energy).sum(...some_indices...)

T_e /= np.prod(Nk)

np.save('ET', (energies, T_e))
plt.plot(energies, T_e)
plt.xlabel('E-E$_f$')
plt.ylabel('T(E)')
plt.savefig('ET.png',dpi=300)

