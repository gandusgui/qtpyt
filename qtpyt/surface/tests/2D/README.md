####
PL == Principal Layer
SC == Supercell
####

Benchmark of single graphene PL.

Subdir structure:
	NAME : # of transverse PLs. Can be EVEN or ODD)
	CONTAINS : results and data of both SC and PL.
	FILES :
		- hs_{pl,sc}_k.npy :: numpy arrays
				Type: Input
				Read: h_kmm, s_kmm = np.load(file)
				Descr: PL Hamilton and Overlap matrices from DFT for each k-point.

		- kpts_{pl,sc}.txt :: list
				Type: Input
                Read: kpts = np.loadtxt(file)
                Descr: kpts used in DFT calculation.

		- h_{pl,sc}_00.npy :: numpy array
				Type: Output
				Read: h_00 = np.load(file)
				Descr: SC Hamilton matrix obtained with my algorithm.
		
