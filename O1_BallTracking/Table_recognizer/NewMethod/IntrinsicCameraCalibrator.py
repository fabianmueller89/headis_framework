import numpy as np
from O1_BallTracking.Table_recognizer.NewMethod.ArucoMarcerDetector_new import cArucoMarkerDetector 
from O1_BallTracking.Table_recognizer.NewMethod.Homography import cHomographyCalculator
import O1_BallTracking.Table_recognizer.NewMethod.tools as tls
import cv2
from scipy.optimize import minimize
from scipy.linalg import cholesky

class cIntrinsicCameraCalibrator():
    def __init__(self, marker_id = 3, forgetting_rate = 0.02):
        self.aruco_marker_detector = cArucoMarkerDetector(marker_id)
        self.cHomography_calculator = cHomographyCalculator()
        self.V_mat_squared = np.zeros((6, 6))
        self.n_meas = 0
        self.forgetting_rate = forgetting_rate # in interval ]0,1[ where 0 means not forgetting and 1 forget all
        self.K_mat = np.zeros((3, 3))

    def update_intrinsic_camera_matrix(self, frame):
        # get current data point
        X_world, X_img = self.aruco_marker_detector.get_correspondences_from_frame(frame)

        if self.aruco_marker_detector.status_dict['aruco_marcer_id_found']:
            self.update_intrinsic_camera_matrix_by_correspondences(X_world, X_img)

        return self.K_mat


    def update_intrinsic_camera_matrix_by_correspondences(self, X_world, X_img):
        ### Calculate actual homography from point correspondences TODO with Sigma for h_mat
        self.h_mat_frame = self.cHomography_calculator.get_homography_from_correspondences(X_world, X_img)

        ### transform h_mat_frame for intrinsic matrix calculations TODO Sigma for V_mat
        self.V_mat_frame = self.__transform_homography_frame_mat(self.h_mat_frame)

        ### update V matrix and number of measurements represented in V_matrix
        self.V_mat_squared = self.__update_V_mat_sqr(self.V_mat_frame)
        self.n_meas += 2

        ### Calculate b_vec => eigenvalue determination # TODO Sigma for b_vec
        # self.K_mat = self.calculate_intrinsic_parameters_nonlinear(self.V_mat_squared)
        b_vec = self.__calc_bvec_from_covariance_mat(self.V_mat_squared)

        ### calculate new intrinsic calibration matices from update V_matrix #TODO Sigma for K_mat_paras
        self.K_mat = self.__calc_Kmat_intrinsic_camera_parameters_from_b_vec(b_vec)  # , self.V_mat_squared)
        return self.K_mat


    def __transform_homography_frame_mat(self, H_mat):
        ### Condition to add to Homographies
        v1_vec = self.__get_v_vec(0, 1, H_mat)
        v2_vec = self.__get_v_vec(0, 0, H_mat) - self.__get_v_vec(1, 1, H_mat)
        V_mat_frame = np.stack((v1_vec, v2_vec))
        return V_mat_frame

    def __get_v_vec(self, p, q, H):
        v1 = H[0,p]*H[0,q]
        v2 = H[0,p]*H[1,q] + H[1,p]*H[0,q]
        v3 = H[1,p]*H[1,q]
        v4 = H[2,p]*H[0,q] + H[0,p]*H[2,q]
        v5 = H[2,p]*H[1,q] + H[1,p]*H[2,q]
        v6 = H[2,p]*H[2,q]
        v_vec = np.array([v1, v2, v3, v4, v5, v6])
        return v_vec

    def __update_V_mat_sqr(self, V_mat_frame):
        V_mat_sqr_t =  self.V_mat_squared * self.n_meas/(self.n_meas + 2)
        V_mat_frame_t = V_mat_frame.T @ V_mat_frame/ (self.n_meas + 2)

        V_mat_squared_new = (1.0 - self.forgetting_rate) * V_mat_sqr_t + V_mat_frame_t
        return V_mat_squared_new

    def __calc_bvec_from_covariance_mat(self, V_squared):
        return tls.get_eigvec_from_min_eigval(V_squared)

    """
    def __calc_Kmat_intrinsic_camera_parameters_from_b_vec(self, b_vec, V_squared):
        B_mat = self.__calc_B_mat_from_b_vec(b_vec)
        eig_vals = np.linalg.eigvals(B_mat)
        if np.all(eig_vals > 0):
            return self.__calculate_K_mat_cholesky(B_mat)
        elif np.all(eig_vals < 0):
            return self.__calculate_K_mat_cholesky(-B_mat)
        else:
            ### Calculate B_mat under constraints for positive definiteness
            b_vec = self.__calc_b_vec_with_pos_def_constraints(V_squared, b_vec)
            B_mat = self.__calc_B_mat_from_b_vec(b_vec)
            return self.__calculate_K_mat_cholesky(B_mat)

    def __calc_B_mat_from_b_vec(self, b_vec):
        B_mat = np.array(
            [[b_vec[0], b_vec[1], b_vec[3]], [b_vec[1], b_vec[2], b_vec[4]], [b_vec[3], b_vec[4], b_vec[5]]])
        return B_mat

    def __calc_b_vec_with_pos_def_constraints(self, V_mat_squared, b_vec_init):

        def loss(x, sign=1.):
            return sign * np.dot(x.T, np.dot(V_mat_squared, x))
        def jac(x, sign=1.):
            return sign * 2 * np.dot(x.T, V_mat_squared)

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

    def __calculate_K_mat_cholesky(self, B_mat):
        #print(np.linalg.eigvals(B_mat))
        if np.all(np.linalg.eigvals(B_mat) > 0):
            L_mat = cholesky(B_mat, lower=True)
            K_mat = np.transpose(np.linalg.inv(L_mat)) * L_mat[2,2]
            return K_mat
        else:
            return None
    """


    def calculate_intrinsic_parameters_nonlinear(self, Q_mat):

        #if isinstance(self.K_mat, np.ndarray):
        #    k_vec_init = self.get_k_vec_from_K_mat(self.K_mat)
        #else:
        k_vec_init = self.get_kvec_from_Kmat(self.K_mat)

        def loss(x):
            return self.loss_nonlinear(self.calc_b_vec_from_K_mat_paras(*x), Q_mat)
        def jac(x):
            return self.calc_jac_dloss_db(self.calc_b_vec_from_K_mat_paras(*x), Q_mat).dot(self.calc_jac_db_dk_paras(*x))
        def constr(x):
            b_vec = self.calc_b_vec_from_K_mat_paras(*x)
            return b_vec.dot(b_vec) - 1

        res = minimize(loss, x0=k_vec_init, jac=jac, method='BFGS', options={'disp': True}, constraints=constr)

        return self.get_K_mat_from_k_vec(*res.x)


    def get_K_mat_from_k_vec(self, alpha, beta, gamma, u_c, v_c):
        return np.array([[alpha, gamma, u_c],[0.0, beta, v_c],[0.0, 0.0, 1.0]])

    def get_kvec_from_Kmat(self, K_mat):
        return np.array([K_mat[0,0], K_mat[1,1], K_mat[0,1], K_mat[0,2], K_mat[1,2]])



    def loss_nonlinear(self, x, Q_mat, sign=1.):
        b_vec = self.calc_b_vec_from_K_mat_paras(x[0], x[1], x[2], x[3], x[4])
        return sign * np.dot(b_vec.T, np.dot(Q_mat, b_vec))


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

    def __calc_Kmat_intrinsic_camera_parameters_from_b_vec(self, b):
        w = b[0] * b[2] * b[5] - b[1] ** 2 * b[5] - b[0] * b[4] ** 2 + 2.0 * b[1] * b[3] * b[4] - b[2] * b[3] ** 2
        d = b[0] * b[2] - b[1] ** 2

        alpha = np.sqrt(w / (d * b[0]))
        beta = np.sqrt(w / d ** 2 * b[0])
        gamma = np.sqrt(w / (d ** 2 * b[0])) * b[1]
        u_c = (b[1] * b[4] - b[2] * b[3]) / d
        v_c = (b[1] * b[3] - b[0] * b[4]) / d

        K_mat = np.array([[alpha, gamma, u_c], [0, beta, v_c], [0, 0, 1]])
        return K_mat


if __name__ == '__main__':
    trafo_mat = cIntrinsicCameraCalibrator(forgetting_rate=0.01)
    trafo_mat_uncertain = cIntrinsicCameraCalibrator(forgetting_rate=0.01)
    K_mat = np.array([[100, 0.0, 50], [0.0, 100, 50], [0.0, 0.0, 1.0]])
    # trafo_mat.K_mat = K_mat

    for idx in range(300):

        # Simulate different perspectives
        # Method 1: Random positional vector and random angle

        angles = np.random.uniform(-0.1, 0.1, size=3)
        t_vec = 2 + 0.1 * np.random.uniform(size=3)


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
        P_mat = np.concatenate((K_mat.dot(R_mat), K_mat.dot(t_vec).reshape((3, 1))), axis=1)

        X_in = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
        X_in[:, -1] = 0

        X_in_rot = np.einsum('ij, kj-> ki', R_mat, X_in - t_vec)
        X_out = np.einsum('ij, kj-> ki', K_mat, X_in_rot)

        X_out[:, 0] = X_out[:, 0] / np.squeeze(X_out[:, -1])
        X_out[:, 1] = X_out[:, 1] / np.squeeze(X_out[:, -1])
        X_out[:, 2] = X_out[:, 2] / np.squeeze(X_out[:, -1])


        # ax = plt.figure(1).add_subplot(1,1,1)
        # ax.plot(X_out[:, 0], X_out[:, 1])
        # plt.show()

        print('accurate')
        K_mat_est = trafo_mat.update_intrinsic_camera_matrix_by_correspondences(X_in, X_out[:, :2])

        # Pixel error on camera
        X_out[:, 0] += np.squeeze(np.random.multivariate_normal(mean=np.array([0.0]), cov=np.array([[1]]), size=4))
        X_out[:, 1] += np.squeeze(np.random.multivariate_normal(mean=np.array([0.0]), cov=np.array([[1]]), size=4))

        print('uncertain')
        K_mat_uc = trafo_mat_uncertain.update_intrinsic_camera_matrix_by_correspondences(X_in, X_out[:, :2])
        print('Number of points', idx)

    print('K_mat_true')
    print(K_mat)
    print('K_mat_Est')
    print(K_mat_est)
    print('K_mat_uc')
    print(K_mat_uc)

    """
    video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211114_170239.mp4"
    intr_cam_cal = cIntrinsicCameraCalibrator(marker_id=3, forgetting_rate=0.0)

    ### Read video
    video_cap = cv2.VideoCapture(video_path)
    #video_cap = cv2.VideoCapture(0)
    frame_start_num = 400
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start_num)
    success, start_frame = video_cap.read()
    if not success:
        print('could not read video. Check video name')
    else:
        idx = frame_start_num
        while True:
            success, img = video_cap.read()

            K_mat = intr_cam_cal.update_intrinsic_camera_matrix(img)
            print('idx', idx)
            print(intr_cam_cal.K_mat)#V_mat_squared)

            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            idx += 1
    video_cap.release()
    cv2.destroyAllWindows()
    """