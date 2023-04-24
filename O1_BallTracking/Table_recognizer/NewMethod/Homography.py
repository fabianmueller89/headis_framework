import numpy as np
import O1_BallTracking.Table_recognizer.NewMethod.tools as tls
from scipy.optimize import least_squares, minimize

class cHomographyCalculator():
    def get_homography_from_correspondences(self, X_world, X_img):
        X_world_scaled, X_img_scaled, T_mat_world, T_mat_img = self.__transform_coordinates(X_world, X_img)

        ### 1. Direct Linear Transformation for calculating Homography matrix
        H_mat_lin_scaled = self.__estimate_homography(X_world_scaled, X_img_scaled)

        H_mat_lin = self.__backtransform_H_mat(T_mat_world, T_mat_img, H_mat_lin_scaled)

        ### 2. Non-Linear Refinement of homography matrix
        #H_mat = self.__refine_homography(H_mat_lin, X_world, X_img)
        H_mat = H_mat_lin

        return H_mat / H_mat[2, 2]

    def __transform_coordinates(self, X_world, X_img):
        X_world[:, -1] = 1
        X_img = np.concatenate((X_img, np.ones((X_img.shape[0], 1))), axis=1)

        ### Transformation of input and output data for numerical stability
        T_mat_img = tls.calculate_normalisation_matrix_T(X_img[:, :-1])
        T_mat_world = tls.calculate_normalisation_matrix_T(X_world[:, :-1])

        X_world_scaled = np.einsum('ij, kj -> ki', T_mat_world, X_world)
        X_img_scaled = np.einsum('ij, kj -> ki', T_mat_img, X_img)

        return X_world_scaled, X_img_scaled, T_mat_world, T_mat_img


    def __backtransform_H_mat(self, T_mat_world, T_mat_img, H_mat_scaled ):
        H_mat = np.linalg.inv(T_mat_img).dot(H_mat_scaled).dot(T_mat_world)
        return H_mat

    def __estimate_homography(self, X_world, X_img):
        num_pts = X_world.shape[0]
        num_dim = 9
        M_mat = np.zeros((2*num_pts, num_dim))
        for idx in range(len(X_world)):
            x, y, z = X_world[idx]
            u, v, w = X_img[idx]
            M_mat[2*idx, :] = np.array([x, y, 1.0, 0.0, 0.0, 0.0, -u*x, -u*y, -u])
            M_mat[2*idx + 1, :] = np.array([0.0, 0.0, 0.0, x, y, 1.0, -v*x, -v*y, -v])

        ### M h = 0
        h_vec = tls.singular_value_decomposition(M_mat)

        ### Backtransformation to original space
        H_mat = np.reshape(h_vec, (3,3))
        return H_mat


    def __refine_homography(self, H_mat, X_world, X_img):
        # X_world_dupl = np.repeat(X_world,2, axis = 0)
        h_vec = H_mat.flatten()
        h_vec = h_vec / h_vec[-1]
        X_img_flatten = X_img.flatten()

        res = least_squares(fun=lambda x: self.__calculate_value_function(X_world, X_img_flatten, x),
                                        jac=lambda x: self.__calculate_jac_mat(X_world, x),
                                        x0=h_vec[:-1], method='lm', verbose=0)

        # res = minimize(fun=lambda x: self.calculate_value_function(X_world, X_img_flatten, x),
        #                                x0=h_vec, method='BFGS')
        h_vec = np.append(res.x, 1)
        H_mat_opt = np.reshape(h_vec, (3, 3))  # / res.x[-8]
        return H_mat_opt

    def __calculate_value_function(self, X_world, X_img, h_vec):
        N = len(X_world)
        y_pred = np.zeros(2 * N)
        for idx in range(N):
            x, y, z = X_world[idx]
            w = h_vec[6] * x + h_vec[7] * y + 1  # h_vec[8]
            u = h_vec[0] * x + h_vec[1] * y + h_vec[2]
            v = h_vec[3] * x + h_vec[4] * y + h_vec[5]
            y_pred[2 * idx] = u / w
            y_pred[2 * idx + 1] = v / w
        return y_pred - X_img
        # return y_pred - X_img

    def __calculate_jac_mat(self, X_mat, h_vec):
        N = len(X_mat)
        j_mat = np.zeros((2 * N, 8))
        for idx in range(N):
            x, y, z = X_mat[idx]
            w = h_vec[6] * x + h_vec[7] * y + 1  # h_vec[8]
            u = h_vec[0] * x + h_vec[1] * y + h_vec[2]
            v = h_vec[3] * x + h_vec[4] * y + h_vec[5]
            j_mat[2 * idx, :] = np.array([x, y, 1.0, 0, 0, 0, -u * x / w, -u * y / w]) / w  # , -u / w])/w
            j_mat[2 * idx + 1, :] = np.array([0, 0, 0, x, y, 1.0, -v * x / w, -v * y / w]) / w  # , -v / w]) / w
        return j_mat

if __name__== '__main__':
    H_mat = np.array([[1,2,3], [4,5,6], [7,3,1]])
    X_w = np.array([[1, 1, 1],[1, -1, 1],[-1,-1, 1], [-1,1,1]])
    X_img = H_mat @ X_w.T
    X_img_n = X_img / X_img[2 ,:]

    print(H_mat)
    print(cHomographyCalculator().get_homography_from_correspondes(X_w, X_img_n[:2, :].T))
