from O9_HelpingFunctions.SkewMatrices import get_list_of_skew_matrices
import numpy as np
import itertools as it
import numpy.linalg as la
from scipy.linalg import null_space
from scipy.optimize import minimize
import scipy as sp

class cOnlineCalibration:
    def __init__(self):
        self.H_skew_lst = get_list_of_skew_matrices(3)
        self.K_matrix = np.zeros((3, 3))
        self.G_matrix = np.zeros((0, 3))

    def add_frame_for_estimating_intrinsic_camera_matrix(self, frame):
        """
        Add new (orthogonal) world coordinate system to image correspondences for estimating intrinsic camera matrices
        :param frame: frame of a movie, where the projected world coordinate system is seen
        :return: adapted matrix K
        """

        ### Detecting exact three lines within the 2d frame
        # Selecting points inside bandwith
        line_vec_lst = None

        ### Determining intersection point of lines as coordinate origin p_0 = [0,0,0]^T
        pt_origin_Ih = None

        ### Determining 3d Coordinate system axis to image line correspondences
        G_new = self.determine_world_COS_axis_to_image_line_correspondences(line_vec_lst, pt_origin_Ih)

        ### Add correspondences to correspondence matrix G, where G * K \approx \vec(0)
        self.G_matrix = np.append(self.G_matrix, G_new, axis = 0)

        ### Filter correspondences out of matrix G (to reduce number and accuracy)

        self.G_matrix = self.G_matrix # TODO Filter G_matrix

        ### Transform System G K = vec(0) into => Eye3 kron G vec(K) = vec(0)
        A_mat = np.kron(np.eye(3), self.G_matrix)
        b_vec = np.zeros(A_mat.shape[0])

        ### Determining elements camera matrix K from all correspondences G: G K = 0
        self.K_matrix = self.determine_intrinsic_cam_mat_from_correspondences(A_mat, b_vec)

    def determine_world_COS_axis_to_image_line_correspondences(self, pt_axis_lst, pt_origin):
        hw_one = pt_origin.dot(self.H_skew_lst[0])
        hw_two = pt_origin.dot(self.H_skew_lst[1])

        n_dim = len(pt_origin)
        n_lines = len(pt_axis_lst)
        G_new = np.zeros((n_lines, n_dim))
        for idx, pt_axis in enumerate(pt_axis_lst):
            G_new[idx, :] = hw_one.dot(pt_axis) * hw_two - hw_two.dot(pt_axis) * hw_one
            #G_new[idx, :] /= np.linalg.norm(G_new[idx, :])
        return G_new

    def determine_intrinsic_cam_mat_from_correspondences(self, A_mat, b_vec):
        ### Symmetric Matrix relationships and other constraints for set restrictions to elements of the intrinsic Kamera matrix:
        # upper triangle matrix in 3D with lower right element equal to 1
        F_mat = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        g_vec = np.array([0, 0, 0, 1])

        ### Solve matrix vector depending on orthogonal relationships and constraints
        k_vec = self.solve_overdetermined_lin_eqts_with_constraints(A_mat, b_vec, F_mat, g_vec)
        #k_vec = self.solve_quadr_sys_with_constraints(A_mat, b_vec, F_mat, g_vec)

        ### Get A_mat from a_vec
        K_mat = np.reshape(k_vec, (3, 3))
        return K_mat

    def solve_quadr_sys_with_constraints(self, A_mat, b_vec, F_mat, g_vec):
        H_mat = np.dot(A_mat.T, A_mat)
        p_vec = np.dot(b_vec.T, A_mat)
        q_scal = np.dot(b_vec, b_vec)
        def loss(x):
            return np.dot(x.T,np.dot(H_mat, x)) - 2 * np.dot(p_vec, x) + q_scal

        def jac(x):
            return 2.0 * (np.dot(x.T, H_mat) + p_vec)

        cons = {'type':'eq', 'fun': lambda x: g_vec - np.dot(F_mat, x), 'jac': lambda x: -F_mat}
        opt = {'disp': True}
        x0 = 100*np.ones(A_mat.shape[1]) #np.random.randn(A_mat.shape[1])
        k_vec = minimize(loss, x0, jac=jac, constraints=cons, method='SLSQP', options=opt)['x']
        #k_vec = minimize(loss, x0, jac=jac, method='SLSQP', options=opt)['x']
        return k_vec

    def solve_overdetermined_lin_eqts_with_constraints(self, A_mat, b_vec, F_mat, g_vec):
        # A_mat x_vec \approx b_vec with F_mat x_vec = g_vec

        # Calculate Basis for constraints F_mat x_vec = g_vec which produce an underdetermined system
        # Provides Transformation with Z_mat : x_vec = Z_mat * d_vec s.t. F_mat x_vec = vec(0)

        # Z_mat creates basis of homogeneous system for x_vec
        Z_mat = null_space(F_mat)

        # Solve F_mat F_mat^T l_vec = g_vec where F_mat^T l_vec = x_vec:
        l_vec = la.inv(F_mat.dot(F_mat.T)).dot(g_vec)
        x_vec_ih = F_mat.T.dot(l_vec)

        # Solution Space of constraints: x_vec = Z_mat d_vec + x_vec_ih
        # Transform A_mat x_vec \approx b_vec to:
        A_mat_z = A_mat.dot(Z_mat)
        b_vec_z = b_vec - A_mat.dot(x_vec_ih)

        # Solve overdetermined equation system WITHOUT constraints and depending on d_vec
        # A_mat_z d_vec = b_vec_z
        ##### d_vec = la.lstsq(A_mat_z, b_vec_z)
        u, s, vh = sp.linalg.svd(A_mat_z)
        d_vec = vh[np.argmin(s), :]

        # Transform back to original space
        x_vec = Z_mat.dot(d_vec) + x_vec_ih
        return x_vec


        # def extend_leqts(l_line, L_mat, b, b_vec):
        #     L_mat = np.append(L_mat, l_line, axis=0)
        #     b_vec = np.append(b_vec, b, axis=0)
        #     return L_mat, b_vec
        #
        # vec_world = np.array([[], [], []])
        # vec_img_lst = np.array([[], [], []])
        # vec_img_lst_h = np.concatenate((pts_img, np.ones(3)), axis=1)
        #
        # L_mat = np.array([[]])
        # b_vec = np.array([])
        #
        # for idx, idy in it.combinations(range(len(vec_img_lst)), 2):
        #     # Create orthogonal relationship
        #     l_line = np.kron(vec_img_lst_h[idx], vec_img_lst_h[idy])
        #     L_mat, b_vec = extend_leqts(l_line, L_mat, 0, b_vec)

def get_rot_mat_3d(angle, axis = 0):
    cos = np.cos(angle)
    sin = np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    elif axis ==1:
        return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    elif axis ==2:
        return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


if __name__ == '__main__':
    online_calib = cOnlineCalibration()

    ### True Intrinsic K-Matrix
    K_matrix = np.array([[120.0, 0.0, 60.0],[0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])

    for i in range(5):
        ### True Extrinsic Camera Matrix
        angle_arr = np.random.uniform(0,2*np.pi, size = 3)
        angle_arr = np.zeros(3)
        R_mat = get_rot_mat_3d(angle_arr[0], axis=0).dot(get_rot_mat_3d(angle_arr[1], axis=1).dot(get_rot_mat_3d(angle_arr[2], axis=2)))
        t_vec = np.array([0.0, 0.0, 2.0])#np.random.uniform(3.0,10.0, size = 3)

        ### Create true 3D coordinates
        #line_wc_lst = np.diag(np.random.uniform(1.0,2.0, size = 3))
        line_wc_lst = np.diag(np.ones(3))

        ### Determine pixel lines
        pts_axis_end_lst = []
        for line_wc in line_wc_lst:
            pt_axis_end= K_matrix.dot(R_mat.dot(line_wc) + t_vec)
            pt_axis_end_Ih = np.array([pt_axis_end[0] / pt_axis_end[2], pt_axis_end[1] / pt_axis_end[2], 1.0])
            pts_axis_end_lst.append(pt_axis_end_Ih)
            ### Back calculation
            #pt_w = np.linalg.inv(K_matrix).dot(R_mat.T.dot(pt_axis_end_Ih)) - K_matrix.dot(t_vec)
            #print(pt_w)


        pt_origin = K_matrix.dot(t_vec)
        pt_origin_Ih = np.array([pt_origin[0]/pt_origin[2], pt_origin[1]/pt_origin[2], 1])

        ### Determining 3d Coordinate system axis to image line correspondences
        G_new = online_calib.determine_world_COS_axis_to_image_line_correspondences(pts_axis_end_lst, pt_origin_Ih)

        ### Add correspondences to correspondence matrix G, where G * K \approx \vec(0)
        online_calib.G_matrix = np.append(online_calib.G_matrix, G_new, axis=0)

    print(np.einsum('kj, ji-> ki', online_calib.G_matrix, K_matrix))

    ### Transform System G K = vec(0) into => Eye3 kron G vec(K) = vec(0)
    A_mat = np.kron(np.eye(3), online_calib.G_matrix)
    b_vec = np.zeros(A_mat.shape[0])

    ### Determining elements camera matrix K from all correspondences G: G K = 0
    K_matrix_approx = online_calib.determine_intrinsic_cam_mat_from_correspondences(A_mat, b_vec)

    print(K_matrix)
    print(K_matrix_approx)

