import numpy as np

def get_num_of_skew_matrices(n_dim):
    return int((n_dim * (n_dim - 1)) / 2)

def get_list_of_skew_matrices(n_dim):
    num_skew_mat = get_num_of_skew_matrices(n_dim)
    H_mat_lst = []

    l = 0  # set initial row index
    k = 1  # set initial column index

    for i in range(num_skew_mat):
        H_mat = np.zeros((n_dim, n_dim))
        # Create Basis Skew Matrix
        H_mat[l,k] = 1
        H_mat[k,l] = -1
        H_mat_lst.append(H_mat)

        k += 1
        if k >= n_dim:
            k = 2 + l
            l += 1
    return H_mat_lst