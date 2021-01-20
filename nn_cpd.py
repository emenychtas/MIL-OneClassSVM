import numpy as np
import tensorly as tl
from scipy.io import loadmat, savemat
from tensorly.decomposition import parafac, non_negative_parafac

# Load MATLAB file.
my_tensor = tl.tensor(loadmat('BKH_200_PROC_033.mat')['T'], dtype='float32')

# Print Tensor Dimensions
print(my_tensor.shape)

# Perform Non-Neg CPD.
factors = non_negative_parafac(my_tensor,rank=40,n_iter_max=2000,init='random',svd='truncated_svd',tol=10e-6,random_state=0,verbose=10)

# Reconstruct tensor from kruskal matrices.
# my_tensor_recon = tl.kruskal_to_tensor(factors)

# Save kruskal matrices.
savemat('NONNEG_R40_BKH_200_PROC_033.mat',{'I':factors[0],'J':factors[1],'K':factors[2]})