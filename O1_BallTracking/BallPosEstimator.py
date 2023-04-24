import numpy as np
import pandas as pd
from numpy import sin, cos, pi
from itertools import product
import PlotVisualizer as viz
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import differential_evolution, basinhopping, shgo, dual_annealing
from Helpers.EvoOptimizer import es_comma, es_plus
from scipy.optimize import minimize_scalar

### 3D Position Estimator for motion-blurred balls or low frequent sampled images

def project_sphere_onto_xyplane(xvec, radius):
    # rays start in coordinate origin
    r_sqr = radius**2
    xvec_sqr = np.square(xvec)

    # A u^2 + B uv + C v^2 + D u + E v + F = 0
    A = r_sqr - xvec_sqr[1] - xvec_sqr[2]
    B = xvec[0]* xvec[1]
    C = r_sqr - xvec_sqr[0] - xvec_sqr[2]
    D = xvec[0] * xvec[2]
    E = xvec[1] * xvec[2]
    F = r_sqr - xvec_sqr[0] - xvec_sqr[1]

    # Transform to (x-m)^T M (x-m) = d
    mat = np.array([[A, B],[B, C]])
    b_vec = np.array([D, E])
    m_vec = - np.linalg.inv(mat.T).dot(b_vec)
    d = m_vec.dot(mat).dot(m_vec) - F
    ellipse = {'M': mat, 'm_vec': m_vec, 'd': d}
    return ellipse

def calc_middle_point_to_ball_pos(u_vec, radius, z0):
    r = radius
    u, v = u_vec[0], u_vec[1]

    ball_pos_one = np.array([(-r ** 2 * u ** 2 * z0 + r ** 2 * v ** 2 * z0 + u ** 2 * z0 ** 3 - v ** 2 * z0 ** 3 + (
                -r ** 2 * v + v * z0 ** 2) * (r ** 2 * v ** 3 - v ** 3 * z0 ** 2 + v * z0 ** 2) / z0 + (
                  -r ** 2 * v + v * z0 ** 2) ** 2 * (u ** 2 * z0 + v ** 2 * z0) / z0 ** 2) / (
                 u * (-r ** 2 * v ** 2 + v ** 2 * z0 ** 2 + z0 ** 2)), (-r ** 2 * v + v * z0 ** 2) / z0, z0])
    # ball_pos_two = np.array([(-r ** 2 * u ** 2 * z0 + r ** 2 * v ** 2 * z0 + u ** 2 * z0 ** 3 - v ** 2 * z0 ** 3 - (-u * np.sqrt(
    #     r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) * (
    #               r ** 2 * v ** 3 - v ** 3 * z0 ** 2 + v * z0 ** 2) / (u ** 2 + v ** 2) + (-u * np.sqrt(
    #     r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) ** 2 * (
    #               u ** 2 * z0 + v ** 2 * z0) / (u ** 2 + v ** 2) ** 2) / (
    #              u * (-r ** 2 * v ** 2 + v ** 2 * z0 ** 2 + z0 ** 2)),
    #  -(-u * np.sqrt(r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) / (
    #              u ** 2 + v ** 2)])
    # ball_pos_three = np.array([(-r ** 2 * u ** 2 * z0 + r ** 2 * v ** 2 * z0 + u ** 2 * z0 ** 3 - v ** 2 * z0 ** 3 - (u * np.sqrt(
    #     r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) * (
    #               r ** 2 * v ** 3 - v ** 3 * z0 ** 2 + v * z0 ** 2) / (u ** 2 + v ** 2) + (u * np.sqrt(
    #     r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) ** 2 * (
    #               u ** 2 * z0 + v ** 2 * z0) / (u ** 2 + v ** 2) ** 2) / (
    #              u * (-r ** 2 * v ** 2 + v ** 2 * z0 ** 2 + z0 ** 2)),
    #  -(u * np.sqrt(r ** 2 * u ** 2 + r ** 2 * v ** 2 - u ** 2 * z0 ** 2 - v ** 2 * z0 ** 2 - z0 ** 2) + v * z0) / (
    #              u ** 2 + v ** 2)])

    print(ball_pos_one)
    return ball_pos_one

def check_if_image_pt_inside_ellipse(ivec, ellipse):
    divec = ivec - ellipse['m_vec']
    inside_b =  divec.dot(ellipse['M']).dot(divec) >= ellipse['d']
    return inside_b

def get_blurred_images(xvec, radius, vvec, delta_t):
    shape_image = (150, 150)
    ratios_image = (4, 4)

    data = product(range(shape_image[0]), range(shape_image[1]))
    image_df = pd.DataFrame(data=data, columns=['uidx', 'vidx'])
    image_df['u'] = image_df['uidx']/shape_image[0] * ratios_image[0] - ratios_image[0]/2
    image_df['v'] = image_df['vidx']/shape_image[1] * ratios_image[1] - ratios_image[1]/2
    image_df['val'] = 0

    image_df_tot = image_df.copy()
    num_steps = 10
    for time in np.linspace(0, delta_t, num_steps):
        x_vec_curr = xvec + time * vvec
        ellipse_curr = project_sphere_onto_xyplane(x_vec_curr, radius)
        image_df['val'] = image_df.apply(lambda x: check_if_image_pt_inside_ellipse(np.array([x['u'],x['v']]), ellipse_curr), axis=1)
        image_df_tot['val'] += image_df['val'].to_numpy(float)
    image_df_tot['val'] /= num_steps
    image_df_tot['val'] = image_df_tot['val']
    return image_df_tot

def from_interval_to_intensity(t, m_raise, min_t, max_t):
    delta_t = max_t - t if m_raise > 0.0 else t - min_t
    intensity = min(max(min_t, delta_t), max_t) / (max_t - min_t)
    return intensity

def calc_intensity(xvec, vvec, u, v, R, Dt):
    A = vvec[0] * u + vvec[1] * v + vvec[2]
    B = xvec[0] * u + xvec[1] * v + xvec[2]
    G = u**2 + v**2 + 1

    # a2 * t**2 + a1 * t + a0
    a2 = A**2 - G * vvec.dot(vvec)
    a1 = B * A - G * xvec.dot(vvec)
    a0 = B**2 - G * (xvec.dot(xvec) - R**2)

    if np.isclose(a2, 0.0):
        if np.isclose(a1, 0.0):
            intensity = 1.0 if a0 > 0.0 else 0.0
        else:
            t = - a0/(2*a1)
            intensity = from_interval_to_intensity(t, a1, 0, Dt)
    else:
        D = (a1/a2)**2 - a0/a2
        if D > 0:
            root = np.sqrt(D)
            offset = -a1/a2
            t_low = offset - root
            m_low = a2 * t_low + a1
            intensity_one = from_interval_to_intensity(t_low, m_low, 0, Dt)

            t_up = offset + root
            m_up = a2 * t_up + a1
            intensity_two = from_interval_to_intensity(t_up, m_up, 0, Dt)
            intensity = intensity_one + intensity_two - 1 if a2 < 0 else intensity_one + intensity_two
        else:
            intensity = 1.0 if a2 > 0.0 else 0.0
    return intensity

def get_blurred_image_continuous(xvec, radius, vvec, delta_t, shape_image, ratios_image):
    data = product(range(shape_image[0]), range(shape_image[1]))
    image_df = pd.DataFrame(data=data, columns=['uidx', 'vidx'])
    image_df['u'] = image_df['uidx'] / shape_image[0] * ratios_image[0] - ratios_image[0] / 2
    image_df['v'] = image_df['vidx'] / shape_image[1] * ratios_image[1] - ratios_image[1] / 2
    image_df['val'] = image_df.apply(
        lambda x: calc_intensity(xvec, vvec, x['u'], x['v'], radius, delta_t), axis=1)
    return image_df

def fct_optimized(xvec, vvec, radius, delta_t, ref_image_df):
    est_val_vec = ref_image_df.apply(
        lambda x: calc_intensity(xvec, vvec, x['u'], x['v'], radius, delta_t), axis=1)
    dv = est_val_vec - ref_image_df['val']
    return dv.dot(dv) / len(est_val_vec)

def fct_optimized_fscore(xvec, vvec, radius, delta_t, ref_image_df):
    est_val_vec = ref_image_df.apply(
        lambda x: calc_intensity(xvec, vvec, x['u'], x['v'], radius, delta_t), axis=1)
    fscore = calc_fscore(est_val_vec, ref_image_df['val'], beta = 1)
    return 1 - fscore

def fct_optimized_contour(xvec, radius, delta_t, ref_image_df):
    est_occ_vec = ref_image_df.apply(
        lambda x: calc_intensity(xvec, vvec, x['u'], x['v'], radius, delta_t) > 0.0, axis=1)
    fct_vect = np.vectorize(ref_estimate_comp)
    val_vec = fct_vect(ref_image_df['occ'], est_occ_vec)
    return np.sum(val_vec)

def fct_optimized_fast(xvec, radius, ref_image_df):
    ellipse_curr = project_sphere_onto_xyplane(xvec, radius)
    est_val_vec = ref_image_df.apply(
        lambda x: check_if_image_pt_inside_ellipse(np.array([x['u'], x['v']]), ellipse_curr), axis=1)
    dv = ref_image_df['occ'].to_numpy(float) - est_val_vec.to_numpy(float)
    return dv.dot(dv) / len(est_val_vec)

def calc_fscore(x_prob_arr, x_ref_prob_arr, beta = 1):
    false_neg_prob = np.sum((x_ref_prob_arr > x_prob_arr) * (x_ref_prob_arr - x_prob_arr))
    false_pos_prob = np.sum((x_prob_arr > x_ref_prob_arr) * (x_prob_arr - x_ref_prob_arr))
    true_pos_prob = np.sum((x_ref_prob_arr >= x_prob_arr) * x_prob_arr) + np.sum(
        (x_prob_arr >= x_ref_prob_arr) * x_ref_prob_arr)
    # true_pos_prob = np.sum(np.min(ref_image_df['val'], est_val_vec))
    precision = true_pos_prob / (true_pos_prob + false_pos_prob)
    recall = true_pos_prob / (true_pos_prob + false_neg_prob)
    if precision > 0 and recall > 0:
        fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    else:
        fscore = 0.0
    return fscore

def fct_optimized_fast_fscore(xvec, radius, ref_image_df, beta = 1):
    ellipse_curr = project_sphere_onto_xyplane(xvec, radius)
    est_val_vec = ref_image_df.apply(
        lambda x: check_if_image_pt_inside_ellipse(np.array([x['u'], x['v']]), ellipse_curr), axis=1)
    f_score = calc_fscore(est_val_vec, ref_image_df['val'], beta = beta)
    return 1 - f_score

def ref_estimate_comp(ref, est):
    if est == False:
        return 0.0
    elif est == ref:
        return -1 # that's good
    else:
        return 1 # that's bad

def cut_image(image_df):
    image_cut_df = image_df.copy(deep=True)
    uidxvec = image_cut_df.groupby(by= 'uidx').apply(lambda x: np.any(x['val'] > 0))
    uidxvec = uidxvec[uidxvec]
    image_cut_df = image_cut_df[(image_cut_df['uidx']>= min(uidxvec.index.to_numpy())) & (image_cut_df['uidx']<= max(uidxvec.index.to_numpy()))]

    vidxvec = image_cut_df.groupby(by='vidx').apply(lambda x: np.any(x['val'] > 0))
    vidxvec = vidxvec[vidxvec]
    image_cut_df = image_cut_df[(image_cut_df['vidx']>= min(vidxvec.index.to_numpy())) & (image_cut_df['vidx']<= max(vidxvec.index.to_numpy()))]
    return image_cut_df

def get_global_solution(objective, bounds, x0, method):
    if method == 'evo':
        return differential_evolution(objective, bounds, popsize=3)
    elif method == 'annealing':
        return dual_annealing(objective, bounds)
    elif method== 'shgo':
        return shgo(objective, bounds)
    elif method =='basin':
        return basinhopping(objective, x0=x0)
    elif method == 'Nelder-Mead':
        return opt.minimize(objective, x0=x0, method='Nelder-Mead', options={'maxiter': None, 'maxfev': None, 'xatol': 0.05, 'fatol': 0.0001, 'adaptive': True})
    elif method == 'es_c':
        return es_comma(objective,np.asarray(bounds), 100, step_size=0.1, mu=10, lam=30)
    elif method == 'es_p':
        return es_plus(objective, np.asarray(bounds), 100, step_size=0.1, mu=10, lam=30)
    else:
        return None


if __name__ == '__main__':
    delta_t = 1.0/30.0
    ### 1. Create Ball with 3D coordinates and velocity
    Radius = 0.4
    xvec = np.array([1.5, 1.5, 2])
    vvec = np.array([30.0, 30.0, 0.0])
    print('xvec0', xvec, 'vvec0', vvec)
    print('xvecT', xvec + vvec * delta_t, 'vvecT', -vvec)
    print('xvec0.5T', xvec + 0.5*vvec * delta_t, 'vvecT', -vvec)
    #vvec = np.zeros(3)

    shape_image = (150, 150)
    ratios_image = (4.0, 4.0)

    ### 2. Transform 3D Ball into 2D blurred-space
    ImageBlurred_df = get_blurred_images(xvec, Radius, vvec, delta_t)
    visual_discrete = viz.cVisualizer(1)
    visual_discrete.plot_potential2d_plot_from_df('u', 'v', 'val', ImageBlurred_df)
    plt.show()

    ImageBlurred_cont_df = get_blurred_image_continuous(xvec, Radius, vvec, delta_t, shape_image, ratios_image)
    visual = viz.cVisualizer(2)
    visual.plot_potential2d_plot_from_df('u', 'v', 'val', ImageBlurred_cont_df)
    #visual.plot_potential2d_surface_plot_from_df('u','v','val',ImageBlurred_df)

    ### 3. Extract relevant part of picture
    ImageBlurred_cut_df = cut_image(ImageBlurred_cont_df)
    umin, umax = ImageBlurred_cut_df['u'].min(), ImageBlurred_cut_df['u'].max()
    vmin, vmax = ImageBlurred_cut_df['v'].min(), ImageBlurred_cut_df['v'].max()

    ### 4. Estimate 3D Parameters of blurred 2D image; optional overlap with (un-)structured background
    ### A: Calculate Initial static guesses
    ## bounds for x,y and vx,vy are depending on the distance to the focus center
    bounds_pos = [[umin, umax], [vmin, vmax], [0.0, 10]]
    bounds_vel = [[-(umax-umin)/delta_t, (umax-umin)/delta_t], [-(vmax-vmin)/delta_t, (vmax-vmin)/delta_t],[-20,20]]
    bounds = bounds_pos + bounds_vel

    #ImageBlurred_cut_df['occ'] = ImageBlurred_cut_df['val'] > 0.0
    ImageBlurred_cont_df['val'] = np.random.uniform(0,1, size=len(ImageBlurred_cont_df))
    def objective_guess(xvec):
        u,v,z = xvec
        #return fct_optimized_contour(np.array([x * z, y * z, z]), Radius, delta_t, ImageBlurred_cut_df)
        #return fct_optimized_fast(np.array([x * z, y * z, z]), Radius, ImageBlurred_cut_df)
        world_pos = calc_middle_point_to_ball_pos((u,v), Radius, z)
        return fct_optimized_fast_fscore(world_pos, Radius, ImageBlurred_cont_df, beta= 0.75)
    x0_pos = np.array([(xmax + xmin) / 2 for xmin, xmax in bounds_pos])
    print('Evolutionary algorithm started')
    #res_occ = get_global_solution(objective_guess, bounds_pos, x0_pos, method='evo')
    #res_vec = res_occ.x

    #z_lst = np.linspace(1, 4, num=11, endpoint=True)
    #score_lst = list(map(lambda z: objective_guess((x0_pos[0],x0_pos[1],z)), z_lst))
    #score_lst = list(map(lambda z: fct_optimized_fast_fscore((x0_pos[0]*z, x0_pos[1]*z, z), Radius, ImageBlurred_cont_df), z_lst))
    #idx_min = np.argmin(score_lst)
    #res_vec = calc_middle_point_to_ball_pos((x0_pos[0],x0_pos[1]), Radius, z_lst[idx_min])
    #print(z_lst)
    #print(score_lst)

    res_scal = minimize_scalar(lambda z: objective_guess((x0_pos[0],x0_pos[1],z)), bounds=bounds_pos[2], method='brent', tol = 0.001)
    print(res_scal)
    res_vec = calc_middle_point_to_ball_pos((x0_pos[0],x0_pos[1]), Radius, res_scal.x)
    #res_vec = x0_pos

    ImageOcc_df = get_blurred_image_continuous(res_vec, Radius, np.zeros(3), delta_t, shape_image, ratios_image)
    visual_precheck = viz.cVisualizer(3)
    visual_precheck.plot_potential2d_plot_from_df('u', 'v', 'val', ImageOcc_df)
    visual_precheck.ax.set_title('precheck')
    print(res_vec)
    print('Resulting state Precheck', np.array([res_vec[0] * res_vec[2], res_vec[1] * res_vec[2], res_vec[2]]))
    plt.show()
    ### B Detailed optimization
    def objective(xvec):
        x,y,z,vx,vy,vz = xvec
        return fct_optimized_fscore(np.array([x * z, y * z, z]), np.array([vx * z, vy * z, vz]), Radius, delta_t, ImageBlurred_cut_df)

    x0 = np.append(res_vec, np.zeros(3)) #np.array([(xmax-xmin)/2 for xmin, xmax in bounds_vel])) # only for basin and Nelder-Mead
    method = ['evo', 'annealing', 'shgo', 'basin', 'Nelder-Mead', 'es_c', 'es_p'][-3]
    res1 = get_global_solution(objective, bounds, x0, method = method)

    ImageFirst_df = get_blurred_image_continuous(np.array([res1.x[0] * res1.x[2], res1.x[1] *res1.x[2], res1.x[2]]), Radius, np.array([res1.x[3] * res1.x[2], res1.x[4] *res1.x[2], res1.x[5]]),
                                                 delta_t, shape_image, ratios_image)
    visual_opt = viz.cVisualizer(4)
    visual_opt.plot_potential2d_plot_from_df('u', 'v', 'val', ImageFirst_df)
    visual_opt.ax.set_title(method)
    print('First optimization', res1)

    ### 5. Compare with original coordinates
    print('Resulting state', np.array([res1.x[0] * res1.x[2], res1.x[1] * res1.x[2], res1.x[2]]), np.array([res1.x[3] * res1.x[2], res1.x[4] *res1.x[2], res1.x[5]]))
    print('xvec0', xvec, 'vvec0', vvec)
    print('xvecT', xvec + vvec* delta_t, 'vvecT', -vvec)

    plt.show()