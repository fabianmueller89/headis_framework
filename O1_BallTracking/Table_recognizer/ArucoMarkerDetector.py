import cv2
import numpy as np
import scipy
from O1_BallTracking.Table_recognizer.TableDetectorClass import cTableDetector
from scipy.optimize import minimize
import copy
### Detection of Aruco Markers, placed correctly on the long edge corners of Headis table

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
#        db           db
#      |----|       |----|
#       _________________   ___
#      |     _______     |  _|_dh ___
#      |    |# ### #|    |         |
#      |    | # ##  |    |         | h
#      |    |###_#_#|    |        _|_
#      |                 |
#      |_________________|
#

#DICT_TABLE = {'L': 2.74, 'B': 1.525, 'H': 0.76} # Headisplatte
DICT_TABLE = {'L': 0.9, 'B': 1.38, 'H': 0.74} # Wohnzimmertisch B= 1.38
#DICT_TABLE = {'L': 0.0, 'B': 0.0, 'H': 0.0} # Wohnzimmertisch B= 1.38
DICT_MARKER = {'d_b': 0.058, 'd_h': 0.03, 'b': 0.096, 'h': 0.096} # Distance from edge to center
DICT_MARKER_ID_POS = {'LL': 0, 'LT': 1, 'LC': 2, 'LM': 66, 'RM': 66, 'RC': 4, 'RT': 3, 'RR': 66} #  Links the aruco marker ids to the positions

DICT_MARKER_ID_POS = {'LL': 0, 'LT': 1, 'LC': 66, 'LM': 66, 'RM': 66, 'RC': 4, 'RT': 3, 'RR': 66} #  Links the aruco marker ids to the positions

class cArucoMarkerDetector(cTableDetector):
    def __init__(self):
        markerSize = 5
        totalMarkers = 250
        self.intr_matrix_cam = np.ones((3,4))# Intrinsic camera matrix
        self.distortion_coefficients = None
        self.marker_dictionary = f'DICT_{markerSize}X{markerSize}_{totalMarkers}'
        self.marker_pos_dict = self.calc_marker_world_center_pos()
        self.transf_mat_calculator = cTransformMatCalculator()

    def calc_marker_world_center_pos(self):
        l_t = DICT_TABLE['L'] / 2
        b_t = DICT_TABLE['B'] / 2
        h_m = DICT_MARKER['h']/ 2 + DICT_MARKER['d_h']
        b_m = DICT_MARKER['b']/ 2 + DICT_MARKER['d_b']
        dict_marker_world_pos = {}
        dict_marker_world_pos['LL'] = np.array([- l_t, - b_t + b_m, - h_m])
        dict_marker_world_pos['LT'] = np.array([- l_t + b_m, - b_t + h_m, 0])
        dict_marker_world_pos['LC'] = np.array([- l_t + b_m, - b_t, -h_m])
        dict_marker_world_pos['LM'] = np.array([- b_m, -b_t, -h_m])
        dict_marker_world_pos['RM'] = np.array([b_m, -b_t, -h_m])
        dict_marker_world_pos['RC'] = np.array([l_t - b_m, - b_t, -h_m])
        dict_marker_world_pos['RT'] = np.array([l_t - b_m, - b_t + h_m, 0])
        dict_marker_world_pos['RR'] = np.array([l_t, - b_t + b_m, - h_m])
        return dict_marker_world_pos

    def calc_marker_shapes(self, marker_key):
        # Aruco Marver starts with left top corner and then clockwise
        b, h = DICT_MARKER['b'] / 2, DICT_MARKER['h']/2
        if marker_key in ['LC', 'LM','RM', 'RC']:
            marker_shapes = np.array([[-b, 0, h], [b, 0, h], [b, 0, -h], [-b, 0, -h]])
        elif marker_key in ['LT', 'RT']:
            marker_shapes = np.array([[b, -h, 0], [-b, -h, 0], [-b, h, 0], [b, h, 0]])
        elif marker_key in ['LL']:
            marker_shapes = np.array([[0, b, h], [0, -b, h], [0, -b, -h], [0, b, -h]])
        elif marker_key in ['RR']:
            marker_shapes = np.array([[0, -b, h], [0, b, h], [0, b, -h], [0, -b, -h]])
        else:
            print('marker key not found')
        return marker_shapes

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
        marker_corners, marker_ids, marker_rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
        return marker_corners, marker_ids

    def create_correspondences(self, marker_corners, marker_ids):
        X_img = None
        X_world = None
        if np.all(marker_ids is not None):  # If there are markers found by detector
            for id, img_corners in zip(marker_ids, marker_corners):  # Iterate in markers
                # Calculate center position of marker
                for key, item in DICT_MARKER_ID_POS.items():
                    if item == id[0]:
                        center = self.marker_pos_dict[key]

                        # Save correspondences
                        if isinstance(X_img, np.ndarray):
                            X_img = np.append(X_img, img_corners[0, :, :], axis=0)
                            X_world = np.append(X_world, center + self.calc_marker_shapes(key), axis=0)
                        else:
                            X_img = copy.deepcopy(img_corners[0, :, :])
                            X_world = copy.deepcopy(center + self.calc_marker_shapes(key))
        return X_img, X_world

    def calculate_normalisation_matrix_T(self, x):
        translation = x.mean(axis=0)
        scaling = np.sqrt(2) / np.mean(np.linalg.norm(x - translation, axis=0))
        dim = x.shape[1]
        S = np.eye(dim) * scaling
        last_row = [0] * dim
        last_row.append(1)
        T = np.vstack((np.hstack((S, -scaling * translation.reshape(-1, 1))),
                       np.array([last_row])))
        return T

    def get_full_camera_matrix(self, frame, linear_b):
        # Detect markers with ids from frame
        marker_corners, marker_ids = self.get_marker_corners_from_frame(frame)

        #cv2.aruco.drawDetectedMarkers(frame, marker_corners)
        #print(marker_ids)
        # Create image to world coordinate correspondences
        X_img, X_world = self.create_correspondences(marker_corners, marker_ids)

        if isinstance(X_img, np.ndarray):
            # Create transformation_matrices for scaling
            T_mat_img = self.calculate_normalisation_matrix_T(X_img)
            T_mat_world = self.calculate_normalisation_matrix_T(X_world)

            # Create inhomogenous coordinates
            vec_h = np.ones((X_img.shape[0],1))
            X_img_h = np.concatenate((X_img, vec_h), axis=1)
            X_world_h = np.concatenate((X_world, vec_h), axis=1)

            # Create scaled Matrices
            X_world_h_scaled = np.einsum('ij, kj -> ki', T_mat_world, X_world_h)
            X_img_h_scaled = np.einsum('ij, kj -> ki', T_mat_img, X_img_h)

            # Calc projection matrix from correspondences
            if linear_b:
                P_mat = self.transf_mat_calculator.calc_projection_matrix_from_correspondences(X_world_h, X_img_h)

                if isinstance(P_mat, np.ndarray):
                    #P_mat = np.linalg.inv(T_mat_img).dot(P_mat_app).dot(T_mat_world)
                    K_mat, Rot_mat, T_vec = self.get_calibration_mat_from_proj_mat(P_mat)
                    print('###### Linear ##########')
                    print(K_mat)
                    print(Rot_mat)
                    print(T_vec)
                    return P_mat
            else:
                K_mat, RT_mat = self.transf_mat_calculator.calc_camera_matrices_from_correspondences(X_world_h,
                                                                                                     X_img_h)
                #K_mat = np.linalg.inv(T_mat_img).dot(K_mat_scal)
                #RT_mat = RT_mat_scal.dot(T_mat_world)
                print('####### Nonlinear #########')
                print(K_mat)
                print(RT_mat)
                return K_mat.dot(RT_mat)

        return None


    def get_calibration_mat_from_proj_mat(self, P_mat):
        M = P_mat[:,:-1]
        m = P_mat[:,-1]
        K_mat, Rot_mat = scipy.linalg.rq(M)
        T_vec = np.linalg.inv(K_mat).dot(m)
        return K_mat, Rot_mat, T_vec

class cTransformMatCalculator():
    def __init__(self):
        self.p_vec_init = np.array([1000.0, 1000.0, 200.0, 200.0, np.pi/2, np.pi/2, np.pi/2, 2.0, 2.0, 2.0])

    def get_num_of_skew_matrices(self, n_dim):
        return int((n_dim * (n_dim-1))/2)

    def get_normalisation_matrix(self, Mat):
        translation = Mat.mean(axis=1)
        scaling = np.sqrt(2) / np.mean(np.linalg.norm(Mat.transpose() - translation, axis=1))
        dim = Mat.shape[0]
        S = np.eye(dim) * scaling
        last_row = [0] * dim
        last_row.append(1)
        T = np.vstack((np.hstack((S, -scaling * translation.reshape(-1, 1))),
                       np.array([last_row])))
        return T


    def get_list_of_skew_matrices(self, n_dim):
        num_skew_mat = self.get_num_of_skew_matrices(n_dim)
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

    def calc_represential_matrix(self, X_in, X_out, H_skew_lst):
        n_corr = X_out.shape[0]  # number of correspondences
        n_in_dim = X_in.shape[1]  # length of input vector = dimension of input
        n_out_dim = X_out.shape[1]  # length of output vector = dimension of output

        # Create matrix A for A P = 0.
        A_mat = np.zeros((n_corr * len(H_skew_lst), n_out_dim * n_in_dim))
        for idx, H_skew in enumerate(H_skew_lst):
            for jdx in range(n_corr):
                A_mat[idx * n_corr + jdx, :] = np.kron(
                    X_in[jdx, :], X_out[jdx, :].dot(H_skew))
        return A_mat

    def optimize_fct(self, p_vec, shape, X_in, X_out):
        P_mat = np.reshape(p_vec, shape)
        return np.max(np.square(X_out - np.einsum('ij, kj-> ki', P_mat, X_in)))

    def create_camera_matrices_from_param_vec(self, p_vec):
        fx, fy, cx, cy, alpha, beta, gamma, tx, ty, tz = p_vec
        # Intrinsic camera Matrix
        K_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        # Extrinsic camera matrix
        R_alpha = self.get_rot_mat_3d(alpha, axis=0)
        R_beta = self.get_rot_mat_3d(beta, axis=1)
        R_gamma = self.get_rot_mat_3d(gamma, axis=2)
        R_mat = R_gamma.dot(R_beta).dot(R_alpha)
        t_vec = np.array([[tx, ty, tz]])
        RT_mat = np.append(R_mat, t_vec.T, axis=1)
        return K_mat, RT_mat

    def optimize_nonlinear_fct(self, p_vec, X_w, X_im):
        K_mat, RT_mat = self.create_camera_matrices_from_param_vec(p_vec)
        #K_mat_inv = np.linalg.inv(K_mat)

        #DX_w = np.einsum('ij, kj-> ki', K_mat_inv, X_im) - np.einsum('ij, kj-> ki', RT_mat, X_w)
        DX_w = X_im - np.einsum('ij, kj-> ki', K_mat.dot(RT_mat), X_w)
        return np.max(np.square(DX_w))

    def get_rot_mat_3d(self, angle, axis = 0):
        cos = np.cos(angle)
        sin = np.sin(angle)
        if axis == 0:
            return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
        elif axis ==1:
            return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        elif axis ==2:
            return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

    def calc_camera_matrices_from_correspondences(self, X_in, X_out):
        ### Optimize intrinsic and extrinsic camera parameters
        res = minimize(lambda x: self.optimize_nonlinear_fct(x, X_in, X_out), self.p_vec_init, method='Nelder-Mead')#,options={'gtol': 1.0e-8, 'disp': False})
        ### Safe parameter for a better initial value
        self.p_vec_init = res.x
        K_mat, RT_mat = self.create_camera_matrices_from_param_vec(res.x)
        return K_mat, RT_mat


    def calc_proj_mat_from_repr_mat(self, X_in, X_out, A_mat):
        n_in_dim = X_in.shape[1]  # length of input vector = dimension of input
        n_out_dim = X_out.shape[1]  # length of output vector = dimension of output
        shape = (n_out_dim, n_in_dim)

        # 1. SVD Method
        u, s, vh = scipy.linalg.svd(A_mat)
        #p_vec_init = vh[np.argmin(s), :]
        p_vec_init = vh[-1, :]

        # 2. Eigenvalue Method
        #w,v = np.linalg.eig((A_mat.T).dot(A_mat))
        #p_vec_init = v[np.argmin(w), :]
        #print(np.reshape(p_vec_init, shape))

        # 3. Random vector
        #p_vec_init = np.random.rand(12)

        P_mat = np.reshape(p_vec_init, shape)
        res = minimize(lambda x: self.optimize_fct(x, shape, X_in, X_out), p_vec_init, method='CG', options={'gtol': 1.0e-8, 'disp': False})
        P_mat = np.reshape(res.x, shape)

        #P_mat = X_in.dot(scipy.linalg.pinv(X_out))
        return P_mat

    def calc_projection_matrix_from_correspondences(self, X_in, X_out):
        # P X_in = X_out
        ### Searching for P s.t. x_out = P x_in for each correspondence k
        ### 1. n*(n-1)/2 Skew matrices H_i s.t. (x_out^T H_i) x_out = 0 = x_out^T H P x_in
        ### 2. For all correspondences k: X_out^T H_i P X_in = 0^T with X_out = [x_out1, x_out2, ...], X_in = [x_in1, x_in2, ...]
        ### 2. For all skew_matrices i: X_out^T H P X_in = 0^T with H = [H_1, H_2, ....]

        ### 3. Vectorization: vec(x_out^T H P x_in) = (x_in^T Kron x_out^T H) vec(P) = vec(0)
        ### 3. Stack over all point correspondences and skew matrices: (x_in^T Kron x_out^T H)_ij vec(P) = vec(0) to A vec(P) = vec(o)
        ### 4. Find homogenous solution by finding null space of stacked matrix A
        ### 5. Euclidean distance error e = sum_i d(x_in, P x_out)
        n_out_dim = X_out.shape[1] # length of output vector = dimension of output
        n_in_dim = X_in.shape[1]  # length of output vector = dimension of output
        n_corres = X_out.shape[0]

        # Check for to less correspondences:
        if not n_corres*self.get_num_of_skew_matrices(n_out_dim) > n_in_dim*n_out_dim:
            return None

        H_skew_lst = self.get_list_of_skew_matrices(n_out_dim)
        A_mat = self.calc_represential_matrix(X_in, X_out, H_skew_lst)
        P_mat = self.calc_proj_mat_from_repr_mat(X_in, X_out, A_mat)
        return P_mat



if __name__ == '__main__':
    P_mat = np.random.rand(3, 4)
    X_in = np.random.rand(12, 4)
    X_in[:, -1] = 1

    X_out = np.einsum('ij, kj-> ki', P_mat, X_in)
    print(P_mat)
    trafo_mat = cTransformMatCalculator()
    P_mat_approx = trafo_mat.calc_matrix_from_correspondences(X_in, X_out)


    print(P_mat_approx)


