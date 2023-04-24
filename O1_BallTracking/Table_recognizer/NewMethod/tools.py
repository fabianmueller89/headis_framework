import numpy as np
import scipy as sp

def calculate_normalisation_matrix_T(X_mat):
    num_pts, dim = X_mat.shape

    x_mean = np.mean(X_mat, axis=0)
    X_diff = X_mat - x_mean
    X_square = np.square(X_diff)
    x_var = np.mean(X_square, axis=0)

    x_sigma = np.sqrt(2.0 / x_var)
    T = np.block([[np.diag(x_sigma), np.array([- x_sigma * x_mean]).T], [np.zeros((1, dim)), np.eye(1)]])
    return T


def singular_value_decomposition(M_mat):
    num_row, num_dim = M_mat.shape
    if num_row < num_dim-1:
        return None
    elif num_row == num_dim-1:
        u, s, vh = np.linalg.svd(M_mat)
        h_vec_init = vh[-1, :]
    else:
        u, s, vh = np.linalg.svd(M_mat)
        h_vec_init = vh[np.argmin(s), :]

        #w, v = np.linalg.eig(np.dot(M_mat.T, M_mat))
        #h_vec_init = v[:, np.argmin(w)]

    return h_vec_init

def get_eigvec_from_min_eigval(M_squared):
    w, v = np.linalg.eig(M_squared)
    b_vec = v[:, np.argmin(w)]
    return b_vec