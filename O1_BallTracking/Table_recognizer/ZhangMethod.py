from O9_HelpingFunctions.SkewMatrices import get_list_of_skew_matrices
import numpy as np
import itertools as it
import numpy.linalg as la
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.optimize import least_squares
import scipy as sp
from copy import deepcopy
import cv2
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt

# Positions of Markers on Table
#                L
#       |------------------|
#        __________________          __
#      /         ^ y        \         \
#     /          | -> x      \         \  B
# LL /_LT________|_________RT_\ RR     _\_
#   LC        LM   RM         RC

#               b
#           |-------|
#       _________________   ___
#      |     _______     |  _|_dh ___
#      |    |# ### #|    |         |
#      |    | # ##  |    |         | h
#      |    |###_#_#|    |        _|_
#      |                 |
#      |_________________|

DICT_MARKER = {'d_h': 0.03, 'b': 0.096, 'h': 0.096} # Distance from edge to center


class cArucoMarkerDetector():
    def __init__(self):
        markerSize = 5
        totalMarkers = 250
        self.marker_dictionary = f'DICT_{markerSize}X{markerSize}_{totalMarkers}'

    def get_marker_corners_from_frame(self, frame):
        """
        Calculate corner point in image coordinates of aruco markers and determine marker ids
        :param frame: frame of image
        :return:
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key = getattr(cv2.aruco, self.marker_dictionary)
        arucoDict = cv2.aruco.Dictionary_get(key)
        arucoParam = cv2.aruco.DetectorParameters_create()
        marker_corners_img, marker_ids, marker_rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

        return marker_corners_img, marker_ids

class cMatTransformation():
    def calculate_normalisation_matrix_T(self, X_mat):
        num_pts, dim = X_mat.shape

        x_mean = np.mean(X_mat, axis = 0)
        X_diff = X_mat - x_mean
        X_square = np.square(X_diff)
        x_var = np.mean(X_square, axis = 0)

        x_sigma = np.sqrt(2.0/ x_var)
        T = np.block([[np.diag(x_sigma), np.array([- x_sigma * x_mean]).T], [np.zeros((1, dim)), np.eye(1)]])
        return T

class cZhangMethod():
    def __init__(self, marker_id = 3):
        self.aruco_marker_detector = cArucoMarkerDetector()
        self.marker_id = marker_id
        self.mat_transformator = cMatTransformation()
        self.V_mat = None
        self.P_mat = None
        self.K_mat = None
        self.H1_vec_lst = []
        self.H2_vec_lst = []


    def get_arucomarker_point_correspondences(self, marker_corners_img_lst, marker_id_lst):
        marker_corner_world = None
        marker_corners_img = None
        # Aruco Marker starts with left top corner and then clockwise
        for marker_corners_img_cand, marker_id in zip(marker_corners_img_lst, marker_id_lst):
            if marker_id == self.marker_id:
                b, h = DICT_MARKER['b'] / 2, DICT_MARKER['h'] / 2
                marker_corner_world = np.array([[-b, -h, 0], [-b, h, 0], [b, h, 0], [b, -h, 0]])
                marker_corners_img = deepcopy(marker_corners_img_cand[0,:,:])
        return marker_corner_world, marker_corners_img


    def singular_value_decomposition(self, M_mat):
        num_row, num_dim = M_mat.shape
        if num_row < num_dim-1:
            return None
        elif num_row == num_dim-1:
            u, s, vh = sp.linalg.svd(M_mat)
            h_vec_init = vh[-1, :]
        else:
            u, s, vh = sp.linalg.svd(M_mat)
            h_vec_init = vh[np.argmin(s), :]
            #w, v = np.linalg.eig(np.dot(M_mat.T, M_mat))
            #h_vec_init = v[:, np.argmin(w)]
            #def objective(x_vec):
            #    return np.linalg.norm(M_mat.dot(x_vec))**2
            #res = minimize(objective, x0=h_vec_init, method='Nelder-Mead')#, options={'gtol': 1.0e-8, 'disp': False})
            #h_vec = res.x
        return h_vec_init


    def transform_coordinates(self, X_world, X_img):
        X_world[:, -1] = 1
        X_img = np.concatenate((X_img, np.ones((X_img.shape[0], 1))), axis=1)

        ### Transformation of input and output data for numerical stability
        T_mat_img = self.mat_transformator.calculate_normalisation_matrix_T(X_img[:, :-1])
        T_mat_world = self.mat_transformator.calculate_normalisation_matrix_T(X_world[:, :-1])

        X_world_scaled = np.einsum('ij, kj -> ki', T_mat_world, X_world)
        X_img_scaled = np.einsum('ij, kj -> ki', T_mat_img, X_img)

        return X_world_scaled, X_img_scaled, T_mat_world, T_mat_img

    def backtransform_H_mat(self, T_mat_world, T_mat_img, H_mat_scaled ):
        H_mat = np.linalg.inv(T_mat_img).dot(H_mat_scaled).dot(T_mat_world)
        return H_mat

    def estimate_homography(self, X_world, X_img):
        num_pts = X_world.shape[0]
        num_dim = 9
        M_mat = np.zeros((2*num_pts, num_dim))
        for idx in range(len(X_world)):
            x, y, z = X_world[idx]
            u, v, w = X_img[idx]
            M_mat[2*idx, :] = np.array([x, y, 1.0, 0.0, 0.0, 0.0, -u*x, -u*y, -u])
            M_mat[2*idx + 1, :] = np.array([0.0, 0.0, 0.0, x, y, 1.0, -v*x, -v*y, -v])

        ### M h = 0
        h_vec = self.singular_value_decomposition(M_mat)
        print(h_vec)
        ### Backtransformation to original space
        H_mat = np.reshape(h_vec, (3,3))
        return H_mat

    def calculate_value_function(self, X_world, X_img, h_vec):
        N = len(X_world)
        y_pred = np.zeros(2 * N)
        for idx in range(N):
            x, y, z = X_world[idx]
            w = h_vec[6] * x + h_vec[7] * y + 1 #h_vec[8]
            u = h_vec[0] * x + h_vec[1] * y + h_vec[2]
            v = h_vec[3] * x + h_vec[4] * y + h_vec[5]
            y_pred[2 * idx] = u/ w
            y_pred[2 * idx + 1] = v/ w
        return y_pred-X_img
        #return y_pred - X_img

    def calculate_jac_mat(self, X_mat, h_vec):
        N = len(X_mat)
        j_mat = np.zeros((2 * N, 8))
        for idx in range(N):
            x, y, z = X_mat[idx]
            w = h_vec[6] * x + h_vec[7] * y + 1 #h_vec[8]
            u = h_vec[0] * x + h_vec[1] * y + h_vec[2]
            v = h_vec[3] * x + h_vec[4] * y + h_vec[5]
            j_mat[2 * idx, :] = np.array([x, y, 1.0, 0, 0, 0, -u * x/ w, -u * y/ w])/w#, -u / w])/w
            j_mat[2 * idx + 1, :] = np.array([0, 0, 0, x, y, 1.0, -v * x / w, -v * y / w])/w #, -v / w]) / w
        return j_mat

    def refine_homography(self, H_mat, X_world, X_img):
        #X_world_dupl = np.repeat(X_world,2, axis = 0)
        h_vec = H_mat.flatten()
        h_vec = h_vec/ h_vec[-1]
        X_img_flatten = X_img.flatten()

        res = sp.optimize.least_squares(fun=lambda x: self.calculate_value_function(X_world, X_img_flatten, x),
                                        jac=lambda x: self.calculate_jac_mat(X_world, x),
                                        x0=h_vec[:-1], method='lm', verbose = 1)

        #res = sp.optimize.minimize(fun=lambda x: self.calculate_value_function(X_world, X_img_flatten, x),
        #                                x0=h_vec, method='BFGS')
        h_vec = np.append(res.x, 1)
        H_mat_opt = np.reshape(h_vec, (3,3))#/ res.x[-8]
        return H_mat_opt


    def get_homography_from_pts(self, X_world, X_img):
        X_world_scaled, X_img_scaled, T_mat_world, T_mat_img = self.transform_coordinates(X_world, X_img)

        ### 2. Direct Linear Transformation for calculating Homography matrix
        H_mat_lin = self.estimate_homography(X_world_scaled, X_img_scaled)

        H_mat_lin = self.backtransform_H_mat(T_mat_world, T_mat_img, H_mat_lin)

        ### 3. Non-Linear Refinement of homography matrix
        H_mat = self.refine_homography(H_mat_lin, X_world, X_img) #-> Doesn't work: the minimum number of points is four

        return H_mat/ H_mat[2, 2]

    def get_homography_from_pts_own(self, X_world, X_img):
        N= X_world.shape[0]

        x_coords = X_world[:, 0]
        y_coords = X_world[:, 1]
        u_coords = X_img[:, 0]
        v_coords = X_img[:, 1]

        xx = np.sum(np.square(x_coords))
        xy = np.sum(y_coords * x_coords)
        yy = np.sum(np.square(y_coords))
        xsum = np.sum(x_coords)
        ysum = np.sum(y_coords)

        ux = np.sum(u_coords * x_coords)
        vx = np.sum(v_coords * x_coords)
        uy = np.sum(u_coords * y_coords)
        vy = np.sum(v_coords * y_coords)

        usum = np.sum(u_coords)
        vsum = np.sum(v_coords)

        A_mat = np.array([[xx, xy, xsum],[xy, yy, ysum],[xsum, ysum, N]])

        print(np.linalg.eigvals(A_mat))

        b1 = np.array([ux, uy, usum])
        b2 = np.array([vx, vy, vsum])
        b3 = np.array([xsum, ysum, N])

        h1_vec = np.linalg.solve(A_mat, b1)
        h2_vec = np.linalg.solve(A_mat, b2)
        h3_vec = np.linalg.solve(A_mat, b3)

        return np.array([h1_vec, h2_vec, h3_vec])


    def get_v_vec(self,p, q, H):
        v1 = H[0,p]*H[0,q]
        v2 = H[0,p]*H[1,q] + H[1,p]*H[0,q]
        v3 = H[1,p]*H[1,q]
        v4 = H[2,p]*H[0,q] + H[0,p]*H[2,q]
        v5 = H[2,p]*H[1,q] + H[1,p]*H[2,q]
        v6 = H[2,p]*H[2,q]
        v_vec = np.array([v1, v2, v3, v4, v5, v6])
        return v_vec

    def collect_H_mat_vectors(self, H_mat_frame):
        h1_vec = H_mat_frame[:, 0]
        h2_vec = H_mat_frame[:, 1]

        if len(self.H1_vec_lst) > 100:
            self.H1_vec_lst.pop(0)
            self.H2_vec_lst.pop(0)

        self.H1_vec_lst = self.H1_vec_lst + [h1_vec]
        self.H2_vec_lst = self.H2_vec_lst + [h2_vec]




    def collect_homographies(self, H_mat):
        ### Condition to add to Homographies
        v1_vec = self.get_v_vec(0, 1, H_mat)
        v2_vec = self.get_v_vec(0, 0, H_mat) - self.get_v_vec(1, 1, H_mat)
        V_mat_idx = np.stack((v1_vec, v2_vec))
        #V_mat_idx = np.stack((v1_vec/np.linalg.norm(v1_vec), v2_vec/np.linalg.norm(v2_vec)))

        if isinstance(self.V_mat, np.ndarray):
            if self.V_mat.shape[0] > 100:
                self.V_mat = np.delete(self.V_mat, (0, 1), axis = 0)
            return np.append(self.V_mat, V_mat_idx, axis = 0)
        else:
            ### First picture added
            return deepcopy(V_mat_idx)

    def calc_b_vec_from_K_mat_paras(self, alpha, beta, gamma, u_c, v_c):
        ### B_mat = scal * A^(-T) A^(-1) without scaling factor 1/(alpha*beta)**2
        B_0 = beta**2
        B_1 = -gamma*beta
        B_2 = gamma**2 + alpha**2
        B_3 = gamma*beta*v_c - beta**2*u_c
        B_4 = -gamma**2*v_c -gamma*beta*u_c - alpha**2 * v_c
        B_5 = gamma**2 * v_c**2 + beta**2 * u_c**2 - 2.0*gamma*v_c*beta*u_c + alpha**2 * v_c**2 + alpha**2 * beta**2
        return np.array([B_0, B_1, B_2, B_3, B_4, B_5])

    def calc_jac_db_dk_paras(self, alpha, beta, gamma, u_c, v_c):
        db_dalpha = np.array([0, 0, 2 * alpha, 0.0, -2*alpha * v_c, 2.0 * alpha * (v_c**2 + beta**2)])
        db_dbeta = np.array([2 * beta, -gamma, 0.0, gamma*v_c - 2*beta*u_c, - gamma*u_c, 2.0 * beta * (u_c**2 + alpha**2)- 2.0*gamma*v_c*u_c])
        db_dgamma = np.array([0.0, -beta, 2.0* gamma, beta*v_c, -2*gamma*v_c - beta*u_c, 2*gamma*v_c**2 - 2.0*v_c*beta*u_c])
        db_duc = np.array([0.0, 0.0, 0.0, -beta**2, - gamma*beta, 2*beta**2* u_c - 2.0*gamma*v_c*beta])
        db_dvc = np.array([0.0, 0.0, 0.0, gamma*beta, -gamma**2 - alpha**2, 2*v_c * (gamma**2 * alpha**2) - 2.0*gamma*beta*u_c])

        Jac_db_dk = np.array([db_dalpha, db_dbeta, db_dgamma, db_duc, db_dvc]).T
        return Jac_db_dk

    def calc_jac_dloss_db(self, b_vec, Q_mat):
        return b_vec.dot(Q_mat + Q_mat.T)

    def calculate_K_paras_from_b_vec(self, b):
        w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2.0 * b[1] * b[3] * b[4] - b[2] * b[3]**2
        d = b[0] * b[2] - b[1]**2

        alpha = np.sqrt(w/(d * b[0]))
        beta = np.sqrt(w/d**2 * b[0])
        gamma = np.sqrt(w/(d**2 * b[0])) * b[1]
        u_c = (b[1] * b[4] - b[2] * b[3])/d
        v_c = (b[1] * b[3] - b[0] * b[4])/d
        return np.array([alpha, beta, gamma, u_c, v_c])

    def calculate_K_mat_cholesky(self, B_mat):
        #print(np.linalg.eigvals(B_mat))
        if np.all(np.linalg.eigvals(B_mat) > 0):
            L_mat = sp.linalg.cholesky(B_mat, lower=True)
            K_mat = np.transpose(np.linalg.inv(L_mat)) * L_mat[2,2]
            return K_mat
        else:
            return None

    def calc_b_vec_with_pos_def_constraints(self, V_mat, b_vec_init):

        Q_mat = np.einsum('ij, ik -> jk', V_mat, V_mat)
        #A_mat = np.zeros((3,6))
        #A_mat[0,0] = A_mat[1, 2] = A_mat[2,5] = 1.0

        def loss(x, sign=1.):
            return sign * np.dot(x.T, np.dot(Q_mat, x))
        def jac(x, sign=1.):
            return sign * 2 * np.dot(x.T, Q_mat)

        epsilon = -1.0e-3
        def constr(x, pos_def = True):
            #determinantes of leading main minors of matrix

            H1 = x[0]
            H2 = x[0] * x[2] - x[1] ** 2
            H3 = x[5] * (x[0] * x[2] - x[1] ** 2) - x[4] * (x[0] *x[4] - x[1] * x[3]) + x[3] * (x[1] *x[4] - x[2] * x[3])
            if pos_def:
                return np.array([H1, H2, H3]) - epsilon
            else:
                return np.array([H1-epsilon, -H2+epsilon, H3-epsilon])

        def constr_jac(x, pos_def = True):
            dconstr1_dx = np.array([1.0, 0, 0, 0, 0, 0])
            dconstr2_dx = np.array([x[2], -2*x[1], x[0], 0, 0, 0])
            dconstr3_dx0 = x[2] * x[5] - x[4] ** 2
            dconstr3_dx1 = -2*x[1]*x[5]+ 2*x[3]*x[4]
            dconstr3_dx2 = x[0]*x[5] - x[3]**2
            dconstr3_dx3 = 2*x[1]*x[4] - 2*x[2]*x[3]
            dconstr3_dx4 = -2*x[0]*x[4] + 2*x[1]*x[3]
            dconstr3_dx5 = x[0]*x[2] - x[1]**2
            dconstr3_dx = np.array([dconstr3_dx0, dconstr3_dx1, dconstr3_dx2, dconstr3_dx3, dconstr3_dx4, dconstr3_dx5])
            if pos_def:
                return np.array([dconstr1_dx, dconstr2_dx, dconstr3_dx])
            else:
                return np.array([dconstr1_dx, -dconstr2_dx, dconstr3_dx])

        cons = {'type': 'ineq',
                'fun': constr,
                'jac': constr_jac}
        opt = {'disp': False, 'ftol': 1.0e-8}
        res_cons = minimize(loss, b_vec_init, jac=jac, constraints=cons, method='COBYLA', options=opt)
        print('constr', constr(res_cons.x)+epsilon)

        #print(B_mat[0:1, 0:1], B_mat[0:2, 0:2], B_mat[0:3, 0:3])
        #print(np.linalg.det(B_mat[0:1,0:1]), np.linalg.det(B_mat[0:2,0:2]), np.linalg.det(B_mat[0:3,0:3]))
        return res_cons.x

    def calc_B_mat_from_b_vec(self, b_vec):
        B_mat = np.array(
            [[b_vec[0], b_vec[1], b_vec[3]], [b_vec[1], b_vec[2], b_vec[4]], [b_vec[3], b_vec[4], b_vec[5]]])
        return B_mat

    def loss_nonlinear(self, x, Q_mat, sign=1.):
        b_vec = self.calc_b_vec_from_K_mat_paras(x[0], x[1], x[2], x[3], x[4])
        return sign * np.dot(b_vec.T, np.dot(Q_mat, b_vec))

    def get_k_vec_from_K_mat(self, K_mat):
        alpha = K_mat[0,0]
        beta = K_mat[1,1]
        gamma = K_mat[0,1]
        u_c = K_mat[0, 2]
        v_c = K_mat[1, 2]
        return np.array([alpha, beta, gamma, u_c, v_c])

    def get_K_mat_from_k_vec(self, alpha, beta, gamma, u_c, v_c):
        return np.array([[alpha, gamma, u_c],[0.0, beta, v_c],[0.0, 0.0, 1.0]])


    def calculate_intrinsic_parameters_least_squares(self, V_mat):

        if isinstance(self.K_mat, np.ndarray):
            k_vec_init = self.get_k_vec_from_K_mat(self.K_mat)
        else:
            k_vec_init = np.random.uniform(low=0, high=100, size=5)

        def fun(x):
            return V_mat.dot(self.calc_b_vec_from_K_mat_paras(*x))/(x[0]*x[1])**2
        def jac(x):
            return V_mat.dot(self.calc_jac_db_dk_paras(*x))/(x[0]*x[1])**2 + np.einsum('i,j-> ij', fun(x), np.array([-2/(x[0]**3 * x[1]**2),-2/(x[0]**2 * x[1]**3), 0, 0, 0]))

        #print('funval_start', fun(k_vec_init))
        #print('jac_start', jac(k_vec_init))
        res = sp.optimize.least_squares(fun=fun, x0=k_vec_init, jac=jac, verbose=2, xtol= 1.0e-7, method='lm')
        #print('funval_end', fun(res.x))
        #print('jac_end', jac(k_vec_init))
        return self.get_K_mat_from_k_vec(*res.x)

    def calculate_intrinsic_parameters_least_squares_direct(self, H1_vec_lst, H2_vec_lst):
        def fun_vec(H1_lst, H2_lst, x):
            alpha, beta, gamma, u, v = x
            N = len(H1_vec_lst)
            func_vec = np.zeros(2 * N)
            for idx in range(N):
                h1, h2, h3 = H1_lst[idx]
                g1, g2, g3 = H2_lst[idx]
                func_vec[2 * idx] = g1*(beta**2*h1 - beta*gamma*h2 + beta*h3*(-beta*u + gamma*v)) + g2*(-beta*gamma*h1 + h2*(alpha**2 + gamma**2) + h3*(-alpha**2*v + gamma*(beta*u - gamma*v))) + g3*(beta*h1*(-beta*u + gamma*v) + h2*(-alpha**2*v + gamma*(beta*u - gamma*v)) + h3*(alpha**2*beta**2 + alpha**2*v**2 + (beta*u - gamma*v)**2))
                func_vec[2 * idx + 1] = -g1*(beta**2*g1 - beta*g2*gamma + beta*g3*(-beta*u + gamma*v)) - g2*(-beta*g1*gamma + g2*(alpha**2 + gamma**2) + g3*(-alpha**2*v + gamma*(beta*u - gamma*v))) - g3*(beta*g1*(-beta*u + gamma*v) + g2*(-alpha**2*v + gamma*(beta*u - gamma*v)) + g3*(alpha**2*beta**2 + alpha**2*v**2 + (beta*u - gamma*v)**2)) + h1*(beta**2*h1 - beta*gamma*h2 + beta*h3*(-beta*u + gamma*v)) + h2*(-beta*gamma*h1 + h2*(alpha**2 + gamma**2) + h3*(-alpha**2*v + gamma*(beta*u - gamma*v))) + h3*(beta*h1*(-beta*u + gamma*v) + h2*(-alpha**2*v + gamma*(beta*u - gamma*v)) + h3*(alpha**2*beta**2 + alpha**2*v**2 + (beta*u - gamma*v)**2))
            return func_vec

        def jac_mat_simple(H1_lst, H2_lst, x):
            alpha, beta, gamma, u, v = x
            N = len(H1_vec_lst)
            j_mat = np.zeros((2 * N, 5))
            for idx in range(N):
                h1, h2, h3 = H1_lst[idx]
                g1, g2, g3 = H2_lst[idx]
                j_mat[2 * idx, :] = np.array([2*alpha*(g2*(h2 - h3*v) - g3*(h2*v - h3*(beta**2 + v**2))), beta*g1*(h1 - h3*u) - g1*(-beta*h1 + gamma*h2 + h3*(beta*u - gamma*v)) - g2*gamma*(h1 - h3*u) - g3*(beta*h1*u - gamma*h2*u + h1*(beta*u - gamma*v) - 2*h3*(alpha**2*beta + u*(beta*u - gamma*v))), -beta*g1*(h2 - h3*v) + g2*(-beta*h1 + 2*gamma*h2 + h3*(beta*u - 2*gamma*v)) + g3*(beta*h1*v + h2*(beta*u - 2*gamma*v) - 2*h3*v*(beta*u - gamma*v)), beta*(-beta*g1*h3 + g2*gamma*h3 + g3*(-beta*h1 + gamma*h2 + 2*h3*(beta*u - gamma*v))), beta*g1*gamma*h3 - g2*h3*(alpha**2 + gamma**2) + g3*(beta*gamma*h1 - h2*(alpha**2 + gamma**2) + 2*h3*(alpha**2*v - gamma*(beta*u - gamma*v)))])
                j_mat[2 * idx + 1, :] = np.array([2*alpha*(-g2*(g2 - g3*v) + g3*(g2*v - g3*(beta**2 + v**2)) + h2*(h2 - h3*v) - h3*(h2*v - h3*(beta**2 + v**2))), -2*alpha**2*beta*g3**2 + 2*alpha**2*beta*h3**2 - 2*beta*g1**2 + 4*beta*g1*g3*u - 2*beta*g3**2*u**2 + 2*beta*h1**2 - 4*beta*h1*h3*u + 2*beta*h3**2*u**2 + 2*g1*g2*gamma - 2*g1*g3*gamma*v - 2*g2*g3*gamma*u + 2*g3**2*gamma*u*v - 2*gamma*h1*h2 + 2*gamma*h1*h3*v + 2*gamma*h2*h3*u - 2*gamma*h3**2*u*v, 2*beta*g1*g2 - 2*beta*g1*g3*v - 2*beta*g2*g3*u + 2*beta*g3**2*u*v - 2*beta*h1*h2 + 2*beta*h1*h3*v + 2*beta*h2*h3*u - 2*beta*h3**2*u*v - 2*g2**2*gamma + 4*g2*g3*gamma*v - 2*g3**2*gamma*v**2 + 2*gamma*h2**2 - 4*gamma*h2*h3*v + 2*gamma*h3**2*v**2, 2*beta*(beta*g1*g3 - beta*g3**2*u - beta*h1*h3 + beta*h3**2*u - g2*g3*gamma + g3**2*gamma*v + gamma*h2*h3 - gamma*h3**2*v), 2*alpha**2*g2*g3 - 2*alpha**2*g3**2*v - 2*alpha**2*h2*h3 + 2*alpha**2*h3**2*v - 2*beta*g1*g3*gamma + 2*beta*g3**2*gamma*u + 2*beta*gamma*h1*h3 - 2*beta*gamma*h3**2*u + 2*g2*g3*gamma**2 - 2*g3**2*gamma**2*v - 2*gamma**2*h2*h3 + 2*gamma**2*h3**2*v])
            return j_mat

        def jac_mat(H1_lst, H2_lst, x):
            alpha, beta, gamma, u, v = x
            return jac_mat_simple(H1_lst, H2_lst, x)/ (alpha*beta)**2 + np.einsum('i,j-> ij', fun_vec(H1_lst, H2_lst, x), np.array([-1/(2*alpha**3*beta**2), -1/(2*alpha**2*beta**3), 0, 0, 0]))

        k_vec = np.array([150.0, 100, 1, 10, 30])
        res = sp.optimize.least_squares(fun=lambda x: fun_vec(H1_vec_lst, H2_vec_lst, x),
                                    jac=lambda x: jac_mat(H1_vec_lst, H2_vec_lst, x),
                                    x0=k_vec, method='lm', verbose=1)
        return self.get_K_mat_from_k_vec(*res.x)

    def calculate_intrinsic_parameters_nonlinear(self, V_mat):
        Q_mat = np.einsum('ij, ik -> jk', V_mat, V_mat)

        #if isinstance(self.K_mat, np.ndarray):
        #    k_vec_init = self.get_k_vec_from_K_mat(self.K_mat)
        #else:
        k_vec_init = np.array([100, 100, 0.0, 50, 50])

        def loss(x):
            return self.loss_nonlinear(self.calc_b_vec_from_K_mat_paras(*x), Q_mat)
        def jac(x):
            return self.calc_jac_dloss_db(self.calc_b_vec_from_K_mat_paras(*x), Q_mat).dot(self.calc_jac_db_dk_paras(*x))

        res = minimize(loss, k_vec_init, jac=jac, method='BFGS', options={'disp': True})

        return self.get_K_mat_from_k_vec(*res.x)


    def calculate_intrinsic_parameters(self, V_mat):
        b_vec = self.singular_value_decomposition(V_mat)
        #print('dist', sorted(np.square(V_mat.dot(b_vec))))
        if isinstance(b_vec, np.ndarray):
            B_mat = self.calc_B_mat_from_b_vec(b_vec)
            eig_vals = np.linalg.eigvals(B_mat)
            print(eig_vals)
            ### Make B_mat for postive definiteness
            if np.all(eig_vals > 0):
                #k_paras = self.calculate_K_paras_from_b_vec(b_vec)
                #return self.get_K_mat_from_k_vec(*k_paras)
                return self.calculate_K_mat_cholesky(B_mat)
            elif np.all(eig_vals < 0):
                #k_paras =self.calculate_K_paras_from_b_vec(-b_vec)
                #return self.get_K_mat_from_k_vec(*k_paras)
                return self.calculate_K_mat_cholesky(-B_mat)
            else:
                ### Calculate B_mat under constraints for positive definiteness
                b_vec = self.calc_b_vec_with_pos_def_constraints(V_mat, b_vec)
                #print('Eig', np.linalg.eigvals(B_mat))
                B_mat = self.calc_B_mat_from_b_vec(b_vec)
                #k_paras = self.calculate_K_paras_from_b_vec(-b_vec)
                #return self.get_K_mat_from_k_vec(*k_paras)
                return self.calculate_K_mat_cholesky(B_mat)

    def calculate_extrinsic_parameters(self, A_mat, H_mat_frame):
        A_mat_inv = np.linalg.inv(A_mat)
        scal = 1.0/np.linalg.norm(A_mat_inv.dot(H_mat_frame[:, 0]))
        r0 = scal * A_mat_inv.dot(H_mat_frame[:, 0])
        r1 = scal * A_mat_inv.dot(H_mat_frame[:, 1])
        r2 = np.cross(r0, r1)
        t_vec = scal * A_mat_inv.dot(H_mat_frame[:, 2])
        R_mat_init = np.array([r0, r1, r2]).T
        R_mat = self.calc_true_rotation_matrix(R_mat_init)
        return R_mat, t_vec

    def calc_true_rotation_matrix(self, R_mat_init):
        u, s, vh = sp.linalg.svd(R_mat_init)
        return u.dot(vh)

    def calibrate(self, X_world, X_img):
        ### Calculate homography from point correspondences
        H_mat_frame = self.get_homography_from_pts(X_world, X_img)
        print(H_mat_frame)

        #H_mat_frame = self.get_homography_from_pts_own(X_world, X_img)
        #print(H_mat_frame)
        ### Collect homographies to estimate intrinsic parameters
        self.V_mat = self.collect_homographies(H_mat_frame)
        self.collect_H_mat_vectors(H_mat_frame)

        if self.V_mat.shape[0] >= self.V_mat.shape[1]:
            ### Calculate intrinsic camera parameters
            #K_mat = self.calculate_intrinsic_parameters(self.V_mat)
            K_mat = self.calculate_intrinsic_parameters_nonlinear(self.V_mat)
            K_mat = self.calculate_intrinsic_parameters_least_squares(self.V_mat)
            #K_mat = self.calculate_intrinsic_parameters_least_squares_direct(self.H1_vec_lst, self.H2_vec_lst)
            if isinstance(K_mat, np.ndarray):
                self.K_mat = K_mat
                ### Calculate extrinisic camera parameters with current homography frame and estimates intrinsic camera parameters
                R_mat, t_vec = self.calculate_extrinsic_parameters(K_mat, H_mat_frame)
                # TODO ExtLensDistortian
                # TODO RefineAll
                return R_mat, t_vec, K_mat
        return None, None, None

    def get_full_camera_matrix(self, frame):

        ### Get Points from Arucomarker
        marker_corners_img_lst, marker_id_lst = self.aruco_marker_detector.get_marker_corners_from_frame(frame)
        if isinstance(marker_corners_img_lst, list) and isinstance(marker_id_lst, np.ndarray):
            ### Get Correspondences between image points X_img and world coordinates X_world
            X_world, X_img = self.get_arucomarker_point_correspondences(marker_corners_img_lst, marker_id_lst)
            if isinstance(X_world, np.ndarray):
                ### if new correspondences exists, execute new calibration
                R_mat, t_vec, K_mat = self.calibrate(X_world, X_img)
                #print(K_mat)
                #print(R_mat, t_vec)

                ### if calibration was possible, because enough point correspondences were collected over different frames
                if isinstance(R_mat, np.ndarray):
                    t_vec_t = np.array([t_vec])
                    K_ext = np.concatenate((R_mat, t_vec_t.T), axis=1)
                    self.P_mat = K_mat.dot(K_ext)

        return self.P_mat



if __name__ == '__main__':
    trafo_mat = cZhangMethod()
    trafo_mat_uncertain = cZhangMethod()
    K_mat = np.array([[100, 0.0, 50], [0.0, 100, 50], [0.0, 0.0, 1.0]])
    #trafo_mat.K_mat = K_mat

    for idx in range(190):

        # Simulate different perspectives
        # Method 1: Random positional vector and random angle

        angles = np.random.uniform(0, 0.5*np.pi, size=3)
        t_vec =2 + 3*np.random.uniform(size=3)
        def get_rot_mat_3d(angle, axis=0):
            cos = np.cos(angle)
            sin = np.sin(angle)
            if axis == 0:
                return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
            elif axis == 1:
                return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
            elif axis == 2:
                return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

        R_alpha = get_rot_mat_3d(angles[0], axis=0)
        R_beta = get_rot_mat_3d(angles[1], axis=1)
        R_gamma = get_rot_mat_3d(angles[2], axis=2)
        R_mat = R_gamma.dot(R_beta).dot(R_alpha)

        """
        # Method 2: Polar coordinate system to ensure that picture is in the frame
        angles = np.random.uniform(0, 2 * np.pi, size = 2)
        r = 1.5 + np.random.uniform()

        r1_vec = np.array([np.sin(angles[0]) * np.cos(angles[1]), np.sin(angles[0]) * np.sin(angles[1]), np.cos(angles[0])])
        r2_vec = np.array([-np.sin(angles[1]), np.cos(angles[1]), 0])
        r3_vec = np.cross(r1_vec, r2_vec)

        R_mat = np.array([r1_vec, r2_vec, r3_vec])
        t_vec = -r * r1_vec
        """
        P_mat = np.concatenate((K_mat.dot(R_mat), K_mat.dot(t_vec).reshape((3,1))), axis=1)

        X_in = np.array([[1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0]])
        X_in[:, -1] = 0

        X_in_rot = np.einsum('ij, kj-> ki', R_mat, X_in - t_vec)
        X_out = np.einsum('ij, kj-> ki', K_mat, X_in_rot)

        X_out[:, 0] = X_out[:, 0] / np.squeeze(X_out[:, -1])
        X_out[:, 1] = X_out[:, 1] / np.squeeze(X_out[:, -1])
        X_out[:, 2] = X_out[:, 2] / np.squeeze(X_out[:, -1])

        # Pixel error on camera
        X_out[:, 0] += np.squeeze(np.random.multivariate_normal(mean=np.array([0.0]), cov=np.array([[1]]), size=4))
        X_out[:, 1] += np.squeeze(np.random.multivariate_normal(mean=np.array([0.0]), cov=np.array([[1]]), size=4))

        # ax = plt.figure(1).add_subplot(1,1,1)
        # ax.plot(X_out[:, 0], X_out[:, 1])
        # plt.show()

        print('accurate')
        R_mat_est, t_vec_est, K_mat_est = trafo_mat.calibrate(X_in, X_out[:, :2])

        print('uncertain')
        R_mat_uc, t_vec_uc, K_mat_uc = trafo_mat_uncertain.calibrate(X_in, X_out[:, :2])
        print('Number of points', idx)

    print('Pmat', P_mat / P_mat[2, 3])
    print(K_mat)
    print(R_mat)
    print(t_vec)

    P_mat_est = np.concatenate((K_mat_est.dot(R_mat_est), K_mat_est.dot(t_vec_est).reshape((3, 1))), axis=1)
    print('P_mat_Est', P_mat_est / P_mat_est[2, 3]-  P_mat / P_mat[2, 3])
    print(K_mat_est)
    print(R_mat_est)
    print(t_vec_est)

    P_mat_uc = np.concatenate((K_mat_uc.dot(R_mat_uc), K_mat_uc.dot(t_vec_uc).reshape((3, 1))), axis=1)
    print('P_mat_uc', P_mat_uc / P_mat_uc[2, 3] - P_mat / P_mat[2, 3])
    print(K_mat_uc)
    print(R_mat_uc)
    print(t_vec_uc)

