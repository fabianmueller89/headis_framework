import numpy as np
import matplotlib.pyplot as plt


class cMotionModel():
    def __init__(self, n_dim, dt):
        self.dt = dt
        self.n_dim = n_dim
        self.calc_transition_model_parameter()

    def calc_transition_model_parameter(self):
        eye = np.eye(self.n_dim)
        eyedt = self.dt*eye
        eyedt2 = self.dt ** 2 / 2 * eye
        zeros = np.zeros((self.n_dim, self.n_dim))

        self.A_mat = np.block([[eye, eyedt], [zeros, eye]]) # system matrix
        self.B_mat = np.block([[eyedt2],[eyedt]]) # input matrix
        self.C_mat = np.block([[eye, zeros]]) # observer matrix

    def set_input_noise(self, u_vec, U_mat):
        self.u_vec = u_vec
        self.U_mat = U_mat

    def calc_transition(self, x_0, u_0):
        x_new = np.einsum('ij, j -> i', self.A_mat, x_0) + np.einsum('ij, j -> i', self.B_mat, u_0)
        return x_new

class cMotionModelJerk(cMotionModel):
    def calc_transition_model_parameter(self):
        eye = np.eye(self.n_dim)
        eyedt = self.dt*eye
        eyedt2 = self.dt ** 2 / 2 * eye
        eyedt3 = self.dt ** 3 / 6 * eye
        zeros = np.zeros((self.n_dim, self.n_dim))

        self.A_mat = np.block([[eye, eyedt, eyedt2], [zeros, eye, eyedt],[zeros, zeros, eye]])  # system matrix
        self.B_mat = np.block([[eyedt2], [eyedt],[eyedt3]])  # input matrix
        self.C_mat = np.block([[eye, zeros, zeros]])  # observer matrix

class cMotionModelSimple(cMotionModel):
    def calc_transition_model_parameter(self):
        eye = np.eye(self.n_dim)
        eyedt = self.dt * eye
        zeros = np.zeros((self.n_dim, self.n_dim))

        self.A_mat = np.block([[eye, eyedt], [zeros, eye]])  # system matrix
        self.B_mat = np.block([[eye, zeros], [zeros, eye]])  # input matrix
        self.C_mat = np.block([[eye, zeros]])  # observer matrix



class cKalmanfilter:
    def __init__(self, mot_mod, meas_noise_mat):
        self.R_mat = meas_noise_mat # Measurement Noise Matrix
        self.U_mat = mot_mod.U_mat
        self.A_mat = mot_mod.A_mat
        self.B_mat = mot_mod.B_mat
        self.C_mat = mot_mod.C_mat
        self.Q_mat = np.einsum('ij, jk, lk -> il', self.B_mat, self.U_mat ,self.B_mat) # Input Noise matrix
        self.u_0 = mot_mod.u_vec

    def calc_transition_step(self, x_0, P_0, u_0):
        x_trans = np.einsum('ij, j -> i', self.A_mat, x_0) + np.einsum('ij, j -> i', self.B_mat, u_0)
        P_trans = np.einsum('ij, jk, lk -> il', self.A_mat, P_0 ,self.A_mat) + self.Q_mat
        return x_trans, P_trans

    def calc_updated_state(self, z_meas, x_trans, P_trans):
        self.z_diff = z_meas - self.C_mat.dot(x_trans) # pre_fit-residual
        self.S_mat = np.einsum('ij, jk, lk -> il', self.C_mat, P_trans ,self.C_mat) + self.R_mat # Innovation mat
        self.K_mat = np.einsum('ij, kj, kl -> il', P_trans, self.C_mat, np.linalg.inv(self.S_mat)) # Kalman matrix
        x_upt = x_trans + self.K_mat.dot(self.z_diff) # updated mean vector
        P_upt = P_trans - np.einsum('ij, jk, kl -> il', self.K_mat, self.C_mat, P_trans) #  Updated State matrix
        return x_upt, P_upt

    def calc_updated_input(self, x_0, x_upt):
        BTB = np.einsum('ji, jk -> ik', self.B_mat, self.B_mat)
        B_pseudo = np.einsum('ij, kj -> ik', np.linalg.inv(BTB), self.B_mat)
        u_mat_real = B_pseudo.dot(x_upt - self.A_mat.dot(x_0))
        Q_mat_real = self.Q_mat - np.einsum('ij, jk, kl -> il', self.K_mat, self.C_mat, self.Q_mat)
        U_mat_real = np.einsum('ij, jk, lk -> il', B_pseudo, Q_mat_real, B_pseudo)
        return u_mat_real, U_mat_real

    def filter_step(self, x, P, z_meas, valid_measure):
        x_trans, P_trans = self.calc_transition_step(x, P, self.u_0)
        if valid_measure:
            x_upt, P_upt = self.calc_updated_state(z_meas, x_trans, P_trans)
            return x_upt, P_upt
        else:
            return x_trans, P_trans


    def initialize_traj(self, len_traj, n_dim, p_dim):
        self.x_traj = np.zeros((len_traj, n_dim))
        self.p_mat_traj = np.zeros((len_traj, n_dim, n_dim))
        self.u_traj = np.zeros((len_traj, p_dim))
        self.u_mat_traj = np.zeros((len_traj, p_dim, p_dim))

    def safe_in_traj(self, idx, x, P_mat, u, u_mat):
        self.x_traj[idx,:] = x
        self.p_mat_traj[idx,:] = P_mat
        self.u_traj[idx,:] = u
        self.u_mat_traj[idx,:] = u_mat

    def filtering(self, z_meas_trajectory, valid_meas_traj, x_0_assumed, P_0_assumed):
        len_traj = len(z_meas_trajectory)
        n_dim = len(x_0_assumed)
        p_dim = len(self.u_0)
        self.initialize_traj(len_traj, n_dim, p_dim)
        x_new, P_new = self.calc_updated_state(z_meas_trajectory[0], x_0_assumed, P_0_assumed)
        for idx in range(len_traj-1):
            x, P = x_new, P_new
            x_new, P_new = self.filter_step(x, P, z_meas_trajectory[idx+1], valid_meas_traj[idx+1])
            u_new, u_mat = self.calc_updated_input(x, x_new)
            self.safe_in_traj(idx, x, P, u_new, u_mat)
        self.safe_in_traj(len_traj-1, x_new, P_new, self.u_0, self.U_mat)

class cIMMFilter():
    def __init__(self, filters, mode_mat):
        self.filters = filters
        self.mode_mat = mode_mat
        self.len_filt = len(self.filters)

    def gaussian_val(self, val, mu, cov_mat):
        norm = np.sqrt(np.linalg.det(2*np.pi*cov_mat))
        delta_val = val - mu
        return np.exp(-0.5*np.einsum('i,ij,j', delta_val, np.linalg.inv(cov_mat), delta_val))/ norm

    def interaction_and_mode_prediction(self, X_mat_filt, P_mat_filt, mu):
        norm_mod = np.einsum('ij, i->j', self.mode_mat, mu)
        mix_prob_mat = np.einsum('ij, i, j ->ij', self.mode_mat, mu, 1.0/ norm_mod)
        X_mat_star = np.einsum('ij, ik-> jk', mix_prob_mat, X_mat_filt)
        P_mat_star = np.zeros(P_mat_filt.shape)
        for hdx in range(self.len_filt):
            P_mat_star_hdx = np.zeros(P_mat_filt[hdx].shape)
            for idx in range(self.len_filt):
                delta_x_vec = X_mat_filt[idx] - X_mat_star[hdx]
                P_mat_shift_idx = P_mat_filt[idx] + np.einsum('i, j -> ij', delta_x_vec, delta_x_vec)
                P_mat_star_hdx += mix_prob_mat[idx, hdx] * P_mat_shift_idx
            P_mat_star[hdx, :, :] = P_mat_star_hdx
        return X_mat_star, P_mat_star, norm_mod

    def interaction_and_mode_prediction_max(self, X_mat_filt, P_mat_filt, mu):
        norm_mod = np.einsum('ij, i->j', self.mode_mat, mu)
        mix_prob_mat = np.einsum('ij, i, j ->ij', self.mode_mat, mu, 1.0/ norm_mod)
        X_mat_star = np.zeros(X_mat_filt.shape)
        P_mat_star = np.zeros(P_mat_filt.shape)
        for hdx in range(self.len_filt):
            idx_max = np.argmax(mix_prob_mat[:,hdx])
            X_mat_star[hdx, :] = X_mat_filt[idx_max]
            P_mat_star[hdx, :, :] = P_mat_filt[idx_max]
        return X_mat_star, P_mat_star, norm_mod

    def filter_process(self, z_meas, valid_meas, X_mat_star, P_mat_star):
        X_mat_filt = np.zeros(X_mat_star.shape)
        P_mat_filt = np.zeros(P_mat_star.shape)
        for hdx in range(self.len_filt):
            subfilter = self.filters[hdx]
            x, P = subfilter.filter_step(X_mat_star[hdx], P_mat_star[hdx], z_meas, valid_meas)
            X_mat_filt[hdx,:] = x
            P_mat_filt[hdx,:,:] = P
        return X_mat_filt, P_mat_filt

    def combine_to_output(self, X_mat_filt, P_mat_filt, mu):
        ### Get x value
        x = np.einsum('i, ik-> k', mu, X_mat_filt)

        ### Get Covariance matrix
        delta_x_mat = X_mat_filt - x
        P_mat_filt_shift = np.einsum('ik, il-> ikl', delta_x_mat, delta_x_mat) + P_mat_filt
        P = np.einsum('i, ikl-> kl', mu, P_mat_filt_shift)
        return x, P

    def combine_to_output_max(self, X_mat_filt, P_mat_filt, mu):
        ### Get x value
        idx_max = np.argmax(mu)
        x = X_mat_filt[idx_max]

        ### Get Covariance matrix
        delta_x_mat = X_mat_filt[idx_max] - x
        P = np.einsum('l, k-> lk', delta_x_mat, delta_x_mat) + P_mat_filt[idx_max]
        return x, P

    def mode_prob_update(self, z_meas, valid_meas, norm_mod):
        if valid_meas == True:
            self.mode_meas_prob = np.array([self.gaussian_val(filt.z_diff, np.zeros(z_meas.shape), filt.S_mat) for filt in self.filters])
            if np.isclose(np.sum(self.mode_meas_prob), 0.0):
                return norm_mod/np.sum(norm_mod)
            norm = self.mode_meas_prob.dot(norm_mod)
            mu = np.einsum('i, i -> i', self.mode_meas_prob, norm_mod) / norm
            return mu
        else:
            return norm_mod/np.sum(norm_mod)

    def mode_prob_update_max(self, z_meas, valid_meas, norm_mod):
        if valid_meas == True:
            self.mode_meas_prob = np.array([self.gaussian_val(filt.z_diff, np.zeros(z_meas.shape), filt.S_mat) for filt in self.filters])
            if np.isclose(np.sum(self.mode_meas_prob), 0.0):
                self.mode_meas_prob = norm_mod/np.sum(norm_mod)
            idx_max = self.mode_meas_prob.argmax()
            mu = np.zeros(len(norm_mod))
            mu[idx_max] = 1
            return mu
        else:
            return norm_mod/np.sum(norm_mod)



    def filter_one_step(self, X_mat_filt, P_mat_filt, mu, z_meas, valid_meas):
        X_mat_star, P_mat_star, norm_mod = self.interaction_and_mode_prediction(X_mat_filt, P_mat_filt, mu)
        X_mat_filt, P_mat_filt = self.filter_process(z_meas, valid_meas, X_mat_star, P_mat_star)
        mu = self.mode_prob_update(z_meas, valid_meas, norm_mod)
        x_out, P_out = self.combine_to_output(X_mat_filt, P_mat_filt, mu)
        return x_out, P_out, X_mat_filt, P_mat_filt, mu

    def filtering(self, z_meas_trajectory, valid_meas_trajectory, x_0_assumed, P_0_assumed, mu):
        X_mat_filt = np.tile(x_0_assumed, self.len_filt).reshape((self.len_filt, len(x_0_assumed)))
        P_mat_filt = np.tile(P_0_assumed.ravel(), self.len_filt).reshape((self.len_filt, P_0_assumed.shape[0], P_0_assumed.shape[1]))
        len_traj = len(z_meas_trajectory)
        n_dim = len(x_0_assumed)
        self.initialize_traj(len_traj, n_dim)
        for idx in range(len_traj):
            x_out, P_out, X_mat_filt, P_mat_filt, mu = self.filter_one_step(X_mat_filt, P_mat_filt, mu, z_meas_trajectory[idx], valid_meas_trajectory[idx])
            self.safe_in_traj(idx, x_out, P_out, mu)

    def initialize_traj(self, len_traj, n_dim):
        self.x_traj = np.zeros((len_traj, n_dim))
        self.p_mat_traj = np.zeros((len_traj, n_dim, n_dim))
        self.mu_traj =np.zeros((len_traj, self.len_filt))

    def safe_in_traj(self, idx, x, P_mat, mu):
        self.x_traj[idx, :] = x
        self.p_mat_traj[idx, :, :] = P_mat
        self.mu_traj[idx, :] = mu

if __name__ == '__main__':

    ### Set motion model
    dt = 1/ 60.0
    Gravity = 9.81
    mot_mod = cMotionModel(1, dt)
    mot_mod_event_neg = cMotionModel(1, dt)
    K = 5
    mot_mod_event_neg.set_input_noise(np.array([-Gravity*(K+1)]), np.diag([K/2]))

    mot_mod_event_pos = cMotionModel(1, dt)
    mot_mod_event_pos.set_input_noise(np.array([Gravity*(K-1)]), np.diag([K/2]))

    mot_mod_fly = cMotionModel(1, dt)
    mot_mod_fly.set_input_noise(np.array([-Gravity]), np.diag([0.1]))

    mot_mod_simple = cMotionModel(1, dt)
    mot_mod_simple.set_input_noise(np.zeros(2), np.diag([0.1, 10.0]))
    ### Set Kalman Filter: Noise matrix (maybe depending on vel/ blurring effects)
    R_mat = np.diag([0.01])

    kalmfly = cKalmanfilter(mot_mod_fly, R_mat)
    kalmevent_neg = cKalmanfilter(mot_mod_event_neg, R_mat)
    kalmevent_pos = cKalmanfilter(mot_mod_event_pos, R_mat)
    kalmevent_simple = cKalmanfilter(mot_mod_simple, R_mat)

    TransMod = np.array([[0.6, 0.2, 0.2], [0.02, 0.96, 0.02], [0.2, 0.2, 0.6]])
    immfilter = cIMMFilter([kalmevent_neg, kalmfly, kalmevent_pos], TransMod)

    #TransMod = np.array([[0.95, 0.05], [0.1, 0.9]])
    #immfilter = cIMMFilter([kalmfly, kalmevent_simple], TransMod)

    ### Set Real Trajectory
    len_traj = 100
    a_traj = -Gravity * np.ones((len_traj,1))
    a_traj[30, :] = np.array([100])
    a_traj[31, :] = np.array([100])
    a_traj[32, :] = np.array([100])
    a_traj[33, :] = np.array([100])
    a_traj[34, :] = np.array([100])
    a_traj[36, :] = -np.array([300])

    x = np.array([10.0, -3.0])
    x_traj = np.zeros((len_traj, len(x)))
    for idx in range(len_traj):
        x_traj[idx, :] = x
        x = mot_mod.calc_transition(x, a_traj[idx])
    #x_traj = np.append(x_traj, a_traj, axis=1)

    time_lst = dt * np.arange(len_traj)

    ### Set Measured Trajectory
    z_meas_traj = np.einsum('ij, kj -> ki', mot_mod_fly.C_mat, x_traj) + np.random.multivariate_normal(np.array([0.0]), np.array([[0.001]]), len_traj)
    immfilter.filtering(z_meas_traj, np.ones(len(z_meas_traj)), np.array([10.0, 0.0]), np.diag([1.0, 2.0]), np.array([0.01, 0.98, 0.01]))
    #immfilter.filtering(z_meas_traj, np.array([10.0, 0.0]), np.diag([1.0, 2.0]), np.array([0.98, 0.02]))




    ### Show trajectories
    fig = plt.figure(2)
    ax_pos = fig.add_subplot(5, 1, 1)
    ax_vel = fig.add_subplot(5, 1, 2)
    ax_acc = fig.add_subplot(5, 1, 3)
    ax_mu = fig.add_subplot(5,1,4)
    ax_sigma = fig.add_subplot(5, 1, 5)

    ax_pos.plot(time_lst, x_traj[:,0], color = 'black') # Reference position
    ax_vel.plot(time_lst, x_traj[:,1], color = 'black') # Reference velocity
    ax_acc.plot(time_lst, a_traj, color = 'black') # Reference acceleration

    ax_pos.plot(time_lst,z_meas_traj[:,0], color='green')  # Reference position

    ax_pos.plot(time_lst, immfilter.x_traj[:,0], color='red')  # Reference position
    ax_vel.plot(time_lst, immfilter.x_traj[:,1], color='red')  # Reference velocity
    #ax_acc.plot(time_lst, immfilter.x_traj[:,2], color='red')  # Reference velocity
    u_traj =[]
    for idx in range(len_traj-1):
        u, _ = kalmfly.calc_updated_input(immfilter.x_traj[idx], immfilter.x_traj[idx+1])
        u_traj.append(u)
    ax_acc.plot(time_lst[:-1], u_traj, color='red')  # Reference velocity

    ax_mu.plot(time_lst, immfilter.mu_traj[:, 0], color='red')  # Reference acceleration
    ax_mu.plot(time_lst, immfilter.mu_traj[:, 1], color='black')  # Reference acceleration
    ax_mu.plot(time_lst, immfilter.mu_traj[:, 2], color='green')  # Reference acceleration

    sigma = [np.trace(x) for x in immfilter.p_mat_traj]
    ax_sigma.plot(time_lst, sigma, color='red')  # Reference acceleration

    plt.show()