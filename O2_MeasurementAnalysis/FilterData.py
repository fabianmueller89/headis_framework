import numpy as np
import matplotlib.pyplot as plt
import FilteringMeasurements as fm

class cIMMFilterCollection():
    def calc_immfilterNegPos(self, data, valid_meas):
        dt = 1 / 60.0
        Gravity = 9.81
        K = 1
        R_mat = np.diag([0.3**2, 0.1**2, 0.01**2])

        mod_var = 100 * K

        mot_mod_fly = fm.cMotionModel(3, dt)
        mot_mod_fly.set_input_noise(np.array([0.0, 0.0, -Gravity]), np.diag([0.01, 0.01, 0.01]))
        kalmfly = fm.cKalmanfilter(mot_mod_fly, R_mat)

        shift = 2 * K * Gravity

        ### x Direction
        mot_mod_event_xneg = fm.cMotionModel(3, dt)
        mot_mod_event_xneg.set_input_noise(np.array([-shift, 0.0, 0.0]), np.diag(3 * [mod_var]))
        kalmevent_xneg = fm.cKalmanfilter(mot_mod_event_xneg, R_mat)

        mot_mod_event_xpos = fm.cMotionModel(3, dt)
        mot_mod_event_xpos.set_input_noise(np.array([shift, 0.0, 0.0]), np.diag(3 * [mod_var]))
        kalmevent_xpos = fm.cKalmanfilter(mot_mod_event_xpos, R_mat)

        ### y Direction
        mot_mod_event_yneg = fm.cMotionModel(3, dt)
        mot_mod_event_yneg.set_input_noise(np.array([0.0, -shift, 0.0]), np.diag(3 * [mod_var]))
        kalmevent_yneg = fm.cKalmanfilter(mot_mod_event_yneg, R_mat)

        mot_mod_event_ypos = fm.cMotionModel(3, dt)
        mot_mod_event_ypos.set_input_noise(np.array([0.0, shift, 0.0]), np.diag(3 * [mod_var]))
        kalmevent_ypos = fm.cKalmanfilter(mot_mod_event_ypos, R_mat)

        ### z Direction
        mot_mod_event_zneg = fm.cMotionModel(3, dt)
        mot_mod_event_zneg.set_input_noise(np.array([0.0, 0.0, -shift - Gravity]), np.diag(3 * [mod_var]))
        kalmevent_zneg = fm.cKalmanfilter(mot_mod_event_zneg, R_mat)

        mot_mod_event_zpos = fm.cMotionModel(3, dt)
        mot_mod_event_zpos.set_input_noise(np.array([0.0, 0.0, shift - Gravity]), np.diag(3 * [mod_var]))
        kalmevent_zpos = fm.cKalmanfilter(mot_mod_event_zpos, R_mat)

        ### Set Kalman Filter: Noise matrix (maybe depending on vel/ blurring effects)

        kalman_filter_lst = [kalmevent_xneg, kalmevent_xpos, kalmevent_yneg, kalmevent_ypos, kalmevent_zneg,
                             kalmevent_zpos, kalmfly]
        num_kalm = len(kalman_filter_lst)
        num_states = 6

        prob_transtrans = 0.1
        prob_gravtrans = 0.02
        prob_transgrav = 0.2

        # sum over rows for each column has to be one
        TransMod =  prob_transtrans * np.ones((num_kalm, num_kalm))
        TransMod +=  np.eye(num_kalm) * (1 - (num_kalm-1) * prob_transtrans - prob_transgrav)
        TransMod[-1, :] = prob_transgrav * np.ones(num_kalm)
        TransMod[:, -1] = prob_gravtrans * np.ones(num_kalm)
        TransMod[-1, -1] = 1 - prob_gravtrans * (num_kalm - 1)

        immfilter = fm.cIMMFilter(kalman_filter_lst, TransMod)
        immfilter.filtering(data, valid_meas, np.zeros(num_states),
                            np.ones((num_states, num_states)),
                            np.ones(num_kalm) / num_kalm)
        return immfilter

    def calc_immfiltertwo(self, data, valid_meas):
        dt = 1 / 60.0
        Gravity = 9.81

        R_mat = np.diag([0.3**2, 0.1**2, 0.01**2])


        mot_mod_fly = fm.cMotionModel(3, dt)
        mot_mod_fly.set_input_noise(np.array([0.0, 0.0, -Gravity]), np.diag([0.01, 0.01, 0.01]))
        kalmfly = fm.cKalmanfilter(mot_mod_fly, R_mat)

        mot_mod_event = fm.cMotionModel(3, dt)
        mot_mod_event.set_input_noise(np.array([0.0, 0.0, -Gravity]), np.diag(3 * [10.0]))
        kalmevent = fm.cKalmanfilter(mot_mod_event, R_mat)

        ### Set Kalman Filter: Noise matrix (maybe depending on vel/ blurring effects)

        kalman_filter_lst = [kalmevent, kalmfly]
        num_kalm = len(kalman_filter_lst)
        num_states = 6

        TransMod = np.array([[0.8, 0.02],[0.2, 0.98]])
        immfilter = fm.cIMMFilter(kalman_filter_lst, TransMod)
        immfilter.filtering(data, valid_meas, np.zeros(num_states),
                            np.ones((num_states, num_states)),
                            np.ones(num_kalm) / num_kalm)
        return immfilter

    def calc_kalman(self, data, valid_meas):
        dt = 1 / 60.0
        Gravity = 9.81
        R_mat = np.diag([0.05, 0.05, 0.05])

        mot_mod_fly = fm.cMotionModel(3, dt)
        mot_mod_fly.set_input_noise(np.array([0.0, 0.0, -Gravity]), np.diag([0.05, 0.05, 0.05]))
        kalmfly = fm.cKalmanfilter(mot_mod_fly, R_mat)
        kalmfly.filtering(data, valid_meas, np.zeros(6), np.diag([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]))
        return kalmfly


if __name__ == '__main__':
    ### Set motion model
    figone = plt.figure(1)
    ax_x = figone.add_subplot(4, 1, 1)
    ax_y = figone.add_subplot(4, 1, 2)
    ax_z = figone.add_subplot(4, 1, 3)
    ax_mu = figone.add_subplot(4, 1, 4)

    figtwo = plt.figure(2)
    ax_vx = figtwo.add_subplot(4, 1, 1)
    ax_vy = figtwo.add_subplot(4, 1, 2)
    ax_vz = figtwo.add_subplot(4, 1, 3)
    ax_vmu = figtwo.add_subplot(4, 1, 4)

    from mpl_toolkits.mplot3d import Axes3D
    fig_3d = plt.figure(3)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # lacked_data = np.load(
    #    'C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/headies2020/headies2020/git_ignore/outputs/2_view_reconstruction_result.npy')
    lacked_data = np.load(
    'C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/headies2020/headies2020/git_ignore/outputs/one_view_reconstruction_result_right_sample_video.npy')
    #lacked_data = np.load(
    #    'C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/Projectseminar_Headis/Abgabe/SoSe_2021_Headis/SoSe_2021_Headis/code/tracking_result.npy')
    #lacked_data = lacked_data[150:500, :]
    lacked_data[:,3]= np.arange(lacked_data.shape[0])

    immfilter = cIMMFilterCollection().calc_immfilterNegPos(lacked_data[:, 0:3], lacked_data[:, 4])
    immfilterTwo = cIMMFilterCollection().calc_immfiltertwo(lacked_data[:, 0:3], lacked_data[:, 4])
    kalmfilter = cIMMFilterCollection().calc_kalman(lacked_data[:, 0:3], lacked_data[:, 4])

    ax_x.plot(lacked_data[:, -2], lacked_data[:, 0], color='black')  # Reference position
    ax_y.plot(lacked_data[:, -2], lacked_data[:, 1], color='black')  # Reference position
    ax_z.plot(lacked_data[:, -2], lacked_data[:, 2], color='black')  # Reference position

    #ax_x.plot(lacked_data[:, -2], immfilter.x_traj[:, 0], color='red')
    #ax_y.plot(lacked_data[:, -2], immfilter.x_traj[:, 1], color='red')
    #ax_z.plot(lacked_data[:, -2], immfilter.x_traj[:, 0], color='red')  # position

    #ax_x.plot(lacked_data[:, -2], kalmfly.x_traj[:, 0], color='red')  # position x
    #ax_y.plot(lacked_data[:, -2], kalmfly.x_traj[:, 1], color='red')  # position y
    #ax_z.plot(lacked_data[:, -2], kalmfly.x_traj[:, 2], color='red')  # position z

    ax_x.plot(lacked_data[:, -2], immfilter.x_traj[:, 0], color='blue')  # position x
    ax_y.plot(lacked_data[:, -2], immfilter.x_traj[:, 1], color='blue')  # position y
    ax_z.plot(lacked_data[:, -2], immfilter.x_traj[:, 2], color='blue')  # position z
    ax_vx.plot(lacked_data[:, -2], immfilter.x_traj[:, 3], color='blue')  # position x
    ax_vy.plot(lacked_data[:, -2], immfilter.x_traj[:, 4], color='blue')  # position y
    ax_vz.plot(lacked_data[:, -2], immfilter.x_traj[:, 5], color='blue')  # position z
    data = np.sqrt((immfilter.x_traj[1:, 3] - immfilter.x_traj[:-1, 3]) ** 2 + (
                immfilter.x_traj[1:, 4] - immfilter.x_traj[:-1, 4]) ** 2 + (
                               immfilter.x_traj[1:, 4] - immfilter.x_traj[:-1, 4]) ** 2)
    ax_vmu.plot(lacked_data[:-1, -2], data, color='blue')



    ax_x.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 0], color='orange')  # position x
    ax_y.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 1], color='orange')  # position y
    ax_z.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 2], color='orange')  # position z
    ax_vx.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 3], color='orange')  # position x
    ax_vy.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 4], color='orange')  # position y
    ax_vz.plot(lacked_data[:, -2], immfilterTwo.x_traj[:, 5], color='orange')  # position z
    data = np.sqrt((immfilterTwo.x_traj[1:, 3] - immfilterTwo.x_traj[:-1, 3]) ** 2 + (
            immfilterTwo.x_traj[1:, 4] - immfilterTwo.x_traj[:-1, 4]) ** 2 + (
                           immfilterTwo.x_traj[1:, 4] - immfilterTwo.x_traj[:-1, 4]) ** 2)
    ax_vmu.plot(lacked_data[:-1, -2], data, color='orange')

    #ax_x.plot(lacked_data[:, -2][lacked_data[:, -1] == 1], lacked_data[:, 0][lacked_data[:, -1] == 1], color='green')  # position x
    #ax_y.plot(lacked_data[:, -2][lacked_data[:, -1] == 1], lacked_data[:, 1][lacked_data[:, -1] == 1], color='green')  # position y
    #ax_z.plot(lacked_data[:, -2][lacked_data[:, -1] == 1], lacked_data[:, 2][lacked_data[:, -1] == 1], color='green')  # position z

    ax_mu.plot(lacked_data[:, -2], immfilter.mu_traj[:, -1], color='blue')
    ax_mu.plot(lacked_data[:, -2], immfilterTwo.mu_traj[:, -1], color='orange')
    #ax_vmu.plot(lacked_data[:, -2], immfilter.mu_traj[:, -1], color='blue')
    #ax_vmu.plot(lacked_data[:, -2], immfilterTwo.mu_traj[:, -1], color='orange')

    #ax_mu.plot(lacked_data[:, -2], immfilter.mu_traj[:, 2], color='red')
    #ax_mu.plot(lacked_data[:, -2], immfilter.mu_traj[:, 3], color='green')


    ax_3d.plot3D(immfilterTwo.x_traj[:, 0], immfilterTwo.x_traj[:, 1], immfilterTwo.x_traj[:, 2], 'orange')
    ax_3d.set_xlim([-4, 4])
    ax_3d.set_ylim([-4, 4])
    ax_3d.set_zlim([-0.5, 2])
    plt.show()