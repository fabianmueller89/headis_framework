import numpy as np
import pandas as pd
import itertools as it

### data Interface
# get i do j: p_ji^1 for player one


class cPrognose:
    def __init__(self):
        self.margin_set_win = 0.01
        self.minimum_win_sets = 2

    def check_if_win_margin_small(self, p):
        return 1.0 - self.margin_point_win < np.sum(p[-2:])

    def get_miss_point_vector(self, P_mat):
        vec = 1.0 - np.sum(P_mat, axis=0)
        return vec

    def get_extended_matrices(self, P1_mat, P2_mat):
        p1_miss = self.get_miss_point_vector(P1_mat)
        p2_miss = self.get_miss_point_vector(P2_mat)

        p1_miss_ext = np.vstack((0.0 * p1_miss, p1_miss))
        p2_miss_ext = np.vstack((p2_miss, 0.0 * p2_miss))

        P1_mat_ext = np.block([[P1_mat, np.zeros((P1_mat.shape[0], 2))], [p1_miss_ext, np.eye(2)]])
        P2_mat_ext = np.block([[P2_mat, np.zeros((P2_mat.shape[0], 2))], [p2_miss_ext, np.eye(2)]])
        return P1_mat_ext, P2_mat_ext


    def get_extended_service_vec(self, p_service):
        win_states = np.array([0.0, 0.0])
        return np.hstack((p_service, win_states))

    def get_point_course(self, P1_mat, P2_mat, p1_service):
        ### Set win states:
        P1_mat_ext, P2_mat_ext = self.get_extended_matrices(P1_mat, P2_mat)
        p_kp1 = self.get_extended_service_vec(p1_service)

        ### Start set
        p_course = p_kp1
        while not self.check_if_win_margin_small(p_kp1):
            p_kp2 = P2_mat_ext.dot(p_kp1)
            p_course = np.vstack((p_course, p_kp2))
            p_kp1 = P1_mat_ext.dot(p_kp2)
            p_course = np.vstack((p_course, p_kp1))
        return p_course

    def get_output_mat(self, n_states):
        Output_mat = np.zeros((2, n_states))
        Output_mat[0,-2] = 1
        Output_mat[1, -1] = 1
        return Output_mat

    def get_transf_mat_for_point_transition_mat(self, Transition_mat):
        ### Bring transition matrix into form: [[P, 0], [P_miss, I]] with eigenvalues(P) < 1
        ### meaning that states in P can be left, otherwise they are endstates

        ### TODO this modul isnot correct
        eig_val = np.linalg.eigvals(Transition_mat)
        idcs_sorted = np.arange(4)#np.argsort(eig_val)

        num_under_one = np.sum(eig_val < 1)

        Reorder_trafo_mat = np.zeros(Transition_mat.shape)
        for idx, idx_new in enumerate(idcs_sorted):
            Reorder_trafo_mat[idx_new, idx] = 1

        return Reorder_trafo_mat, num_under_one

    def state_transform_linear_system(self, A, C, T):
        A_new = None if not isinstance(A, np.ndarray) else np.einsum('ij,jk,lk->il', T, A, T)
        C_new = None if not isinstance(C, np.ndarray) else np.einsum('jk,lk->jl', C, T)
        return A_new, C_new

    def get_unlimited_transition_mat(self, Transition_mat, num_eigv_under_one):
        n_states = Transition_mat.shape[0]

        P = Transition_mat[0:num_eigv_under_one, 0:num_eigv_under_one]
        P_miss = Transition_mat[num_eigv_under_one:,0:num_eigv_under_one]

        P_miss_unlim = P_miss.dot(np.linalg.inv(np.eye(n_states - num_eigv_under_one) - P))
        I_eye = np.eye(n_states - num_eigv_under_one)

        Transition_unlimited = np.block([[np.zeros((num_eigv_under_one, Transition_mat.shape[0]))],[P_miss_unlim, I_eye]])
        return Transition_unlimited

    def get_point_result(self, P1_mat, P2_mat, p1_service):
        p_service_ext = self.get_extended_service_vec(p1_service)
        P1_mat, P2_mat = self.get_extended_matrices(P1_mat, P2_mat)
        Transition_mat = P1_mat.dot(P2_mat)

        ### Get Transformation matrix to split end states from transition states
        trafo_mat, num_under_one = self.get_transf_mat_for_point_transition_mat(Transition_mat)

        ### This function is much shorter then the detailed point course
        output_mat = self.get_output_mat(len(p_service)+2)

        ### Transform system by Trafo
        Transition_mat, output_mat = self.state_transform_linear_system(Transition_mat, output_mat, trafo_mat)

        ### Get Unlimited Transition matrix
        trans_mat_unlim = self.get_unlimited_transition_mat(Transition_mat, num_under_one)

        x_win_vec = np.einsum('ij, jk, lk, l-> i', output_mat, trans_mat_unlim, trafo_mat, p_service_ext)
        return x_win_vec

    def get_set_result(self, P1_mat, P2_mat, p1_service, p2_service):
        p_p1start = self.get_point_result(P1_mat, P2_mat, p1_service)
        p_p2start = self.get_point_result(P1_mat, P2_mat, p2_service)
        return

    def get_all_game_win_combinations(self):
        all_combinations = np.array(list(it.permutations([0,1], self.minimum_win_sets)))
        idx_wins = np.argwhere(lambda x: np.sum(x) >= self.minimum_win_sets, all_combinations)
        win_combinations = all_combinations[idx_wins]
        return win_combinations

    def get_game_course(self, P1_mat, P2_mat, p1_start, p2_start, start_pl = None):
        res_pl1start = self.get_set_result(P1_mat, P2_mat, p1_start, p2_start)
        res_pl2start = self.get_set_result(P2_mat, P1_mat, p2_start, p1_start)
        winp1_pl1start, winp2_pl1start = res_pl1start[0], res_pl1start[1]
        winp2_pl2start, winp1_pl2start = res_pl2start[0], res_pl2start[1]

        win_order_combinations = np.array(list(it.product([0,1],[0,1],[0,1])))

        if isinstance(start_pl, int):
            if start_pl == 1:
                # sum_v1^max/2 sum_v2^(Max/2+1) pl1^v1 (1-pl1)^(max/2-v1) pl2^v2 pl2^(max/2-v2)
                win_prob_lst = np.array([winp1_pl1start, winp1_pl2start, winp1_pl1start])

            #elif start_pl == 2:
                #win_prob_lst =

        return np.einsum('ij, jk-> i', win_prob_lst, win_order_combinations)



if __name__ == '__main__':
    P_1 = np.array([[0.7, 0.0], [0.5, 0.0]]).T
    P_2 = np.array([[0.8, 0.0], [0.5, 0.0]]).T
    p_service = np.array([0.5, 0.5])

    prog = cPrognose()
    result = prog.get_point_result(P_1, P_2, p_service)
    print(result)