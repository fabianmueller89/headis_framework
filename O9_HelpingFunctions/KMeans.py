import numpy as np
import matplotlib.pyplot as plt
import copy

class cDistObj:
    def __init__(self):
        self.num_maximization_iteration = 1

    def dist_to_center(self, datum, center):
        pass

    def center_from_data(self, data_subset_lst, center_lst):
        pass

### intersection pt is determined, where all lines are intersected
class cInterAngle(cDistObj):
    def __init__(self):
        self.inter_pt = np.array([0.0, 0.0])
        self.num_maximization_iteration = 1

    def get_normal_vector(self, angle):
        return np.array([np.cos(angle), np.sin(angle)])

    def dist_to_center(self, datum, angle):
        a_vec = self.get_normal_vector(angle)
        return a_vec.dot(datum - self.inter_pt) ** 2

    def center_from_data(self, data_subset_lst, center_lst):
        ### update internal parameters
        self.update_internal_parameter(data_subset_lst, center_lst)

        ### update cluster centers
        for idx in range(len(center_lst)):
            data_set = data_subset_lst[idx]
            A = np.sum(np.square(data_set[:, 1] - self.inter_pt[1]) - np.square(data_set[:, 0]- self.inter_pt[0]))
            B = np.sum(np.prod(data_set - self.inter_pt, axis=1))
            angle = np.arctan2(-2*B, A)/2.0
            center_lst[idx] = angle
        return center_lst

    def update_internal_parameter(self, data_subset_lst, center_lst):
        num_dim = len(self.get_normal_vector(0))
        b_vec = np.zeros(num_dim)
        A_mat = np.zeros((num_dim, num_dim))
        for idx in range(len(center_lst)):
            data_set = data_subset_lst[idx]
            normal_vec = self.get_normal_vector(center_lst[idx])
            b_vec += normal_vec * np.sum(np.einsum('l, il->i', normal_vec, data_set))
            A_mat += len(data_set) * np.einsum('l, k-> lk', normal_vec, normal_vec)
        self.inter_pt = np.linalg.solve(A_mat, b_vec)

class cClusterLine2(cDistObj):
    def __init__(self):
        self.num_dim = 2

    def get_normal_vector(self, angle):
        return np.array([np.cos(angle), np.sin(angle)])

    def dist_to_center(self, datum, line):
        angle, dist = line
        a_vec = self.get_normal_vector(angle)
        return (a_vec.dot(datum) - dist) ** 2

    def center_from_data(self, data_subset_lst, center_lst):
        center_lst_new = []
        for idx in range(len(center_lst)):
            data_set = data_subset_lst[idx]
            len_data = len(data_set)
            A = np.sum(np.square(data_set[:, 1]) - np.square(data_set[:, 0]))
            B = np.sum(np.prod(data_set, axis=1))
            C = np.sum(data_set[:, 1])**2 - np.sum(data_set[:, 0])**2
            D = np.prod(np.sum(data_set, axis=0))

            x = len_data * A - C
            y = len_data * B - D

            angle_center = np.arctan2(-2*y, x)/2.0
            a_vec = self.get_normal_vector(angle_center)
            vec_add = np.sum(data_set, axis=0)/len_data
            dist_center = a_vec.dot(vec_add)
            line = np.array([angle_center, dist_center])
            center_lst_new.append(line)
        return center_lst_new

class cCenterPoint(cDistObj):
    def __init__(self, num_dim):
        self.num_dim = num_dim

    def dist_to_center(self, datum, center_point):
        return np.linalg.norm(datum - center_point)

    def center_from_data(self, data_subset_lst, center_lst):
        center_lst_new = []
        for idx in range(len(center_lst)):
            data_set = data_subset_lst[idx]
            center_pt = np.mean(data_set, axis = 0)
            center_lst_new.append(center_pt)
        return center_lst_new

class cKMeans:
    def __init__(self, init_mode, dist_obj, iter_max = 100, num_clust_centers = 3, cluster_centers = None):
        self.initilization_mode = init_mode
        self.iter_max = iter_max
        self.dist_obj = dist_obj

        if isinstance(cluster_centers, list):
            self.cluster_centers = cluster_centers
            self.num_clust_centers = len(self.cluster_centers)
        else:
            self.num_clust_centers = num_clust_centers
            self.cluster_centers = [None] * self.num_clust_centers

    def update_cluster_centers(self):
        data_subset_lst = []
        for idx in range(len(self.cluster_centers)):
            data_subset = self.data_set[self.memberships_data2clusters == idx]
            data_subset_lst.append(data_subset)
        self.cluster_centers = self.dist_obj.center_from_data(data_subset_lst, self.cluster_centers)

    def update_memberships(self):
        for idx in range(len(self.data_set)):
            self.memberships_data2clusters[idx] = np.argmin(np.array([self.dist_obj.dist_to_center(self.data_set[idx], center) for center in self.cluster_centers]))

    def fit(self, data_set):
        self.data_set = data_set

        ### Initialization
        if self.initilization_mode == 'membership':
            self.memberships_data2clusters = np.random.randint(0, self.num_clust_centers, size= len(self.data_set))
            #self.memberships_data2clusters = np.concatenate((0.0*np.ones(30),1.0*np.ones(30), 2.0*np.ones(30)))
        elif self.initilization_mode == 'centers':
            self.memberships_data2clusters = np.zeros(len(self.data_set))
            self.update_memberships()
        self.memberships_data2clusters_old = 1 + self.memberships_data2clusters
        steps = 1
        #while not np.all(np.isclose(self.memberships_data2clusters_old, self.memberships_data2clusters)):
        while steps < self.iter_max:
            self.update_cluster_centers()
            self.memberships_data2clusters_old = copy.deepcopy(self.memberships_data2clusters)
            self.update_memberships()
            steps += 1
            print(steps)

    def initialize_memberships(self):
        self.

def create_pts_around_line(line, low, up, var, size):
    alpha, dist = line
    ### Calculate Transformation from 2D line
    a_vec = np.array([np.cos(alpha), np.sin(alpha)])
    a_sqr = a_vec.dot(a_vec) # = 1
    x0 = dist/a_sqr * a_vec
    T = np.array([[-np.sin(alpha), np.cos(alpha)], [np.cos(alpha), np.sin(alpha)]])

    ### Draw uniform distributed value along line
    x = np.random.uniform(low, up, size=size)
    y = np.random.uniform(-var, var, size=size)
    #y = np.random.normal(0, var, size=size)

    x_pts = np.vstack((x, y)).T
    return x0 + np.einsum('ij, kj->ki', T, x_pts)

def get_pts_on_line(line):
    alpha, dist = line
    x0 = dist * np.array([np.cos(alpha), np.sin(alpha)])
    dvec = np.array([-np.sin(alpha), np.cos(alpha)])
    pts = np.array([x0 - dvec, x0 + dvec]).T
    return pts[0,:], pts[1,:]

def test_k_means():
    ### Test 1: Kmeans Center Point oit of clouds
    center_true_lst = [np.array([1, 1]), np.array([1, 2]), np.array([2, 1])]
    data_set_one = np.random.multivariate_normal(center_true_lst[0], 0.01 * np.diag([1, 100]), size=30)
    for center_true in center_true_lst[1:]:
        random_pts = np.random.multivariate_normal(center_true, 0.01 * np.diag([1, 100]), size=30)
        data_set_one = np.concatenate((data_set_one, random_pts), axis=0)
        print(center_true)

    dist_obj = cCenterPoint(2)
    kmeans = cKMeans('membership', dist_obj, num_clust_centers=len(center_true_lst))
    kmeans.fit(data_set_one)
    for center_km in kmeans.cluster_centers:
        print(center_km)

    fig = plt.figure(1)
    axis = fig.add_subplot(111)
    axis.scatter(data_set_one[:, 0], data_set_one[:, 1], s=80, c=0.25 * (1 + kmeans.memberships_data2clusters),
                 marker="x")
    for center_km in kmeans.cluster_centers:
        axis.scatter(center_km[0], center_km[1], s=80, c=0.25 * (1 + np.arange(len(kmeans.cluster_centers))),
                     marker="o")
    for center_true in center_true_lst:
        axis.scatter(center_true[0], center_true[1], s=80, c=0.25 * (1 + np.arange(len(center_true_lst))), marker=".")
    plt.show()

def test_k_means_lines():
    ### Test 2: Kmeans Line
    lines_true_lst = [np.array([0.0, 0.0]), np.array([1.4, 0.0]), np.array([2.5, 0.0])]
    # data_set_two = np.array([[1,2], [3,4]])
    data_set_two = create_pts_around_line(lines_true_lst[0], 0, 2, 0.05, size=30)
    print(lines_true_lst[0])
    for line_true in lines_true_lst[1:]:
        random_pts = create_pts_around_line(line_true, 0, 2, 0.05, size=30)
        data_set_two = np.concatenate((data_set_two, random_pts), axis=0)
        print(line_true)

    dist_obj = cClusterLine2()
    kmeansline = cKMeans('membership', dist_obj, num_clust_centers=len(lines_true_lst))
    # kmeansline = cKMeans('centers', dist_obj, cluster_centers=lines_true_lst, iter_max=200)
    kmeansline.fit(data_set_two)
    for center_km in kmeansline.cluster_centers:
        print(center_km)

    fig = plt.figure(2)
    axis = fig.add_subplot(111)
    axis.scatter(data_set_two[:, 0], data_set_two[:, 1], s=80, c=0.25 * (1 + kmeansline.memberships_data2clusters),
                 marker="x")
    for center_km in kmeansline.cluster_centers:
        x_lst, y_lst = get_pts_on_line(center_km)
        axis.plot(x_lst, y_lst, color='red')
    for line_true in lines_true_lst:
        x_lst, y_lst = get_pts_on_line(line_true)
        axis.plot(x_lst, y_lst, color='black')
    plt.show()

def test_k_means_angles_inter():
    ### Test 3: Kmeans Line with center point
    lines_true_lst = [np.array([0.0, 0.0]), np.array([1.4, 0.0]), np.array([2.5, 0.0])]
    # data_set_two = np.array([[1,2], [3,4]])
    data_set_two = create_pts_around_line(lines_true_lst[0], 1, 2, 0.05, size=30)
    print(lines_true_lst[0])
    for line_true in lines_true_lst[1:]:
        random_pts = create_pts_around_line(line_true, 1, 2, 0.05, size=30)
        data_set_two = np.concatenate((data_set_two, random_pts), axis=0)
        print(line_true)

    #cluster_center_guess = np.linspace(0, 2*np.pi, len(lines_true_lst), endpoint=False).tolist()
    cluster_center_guess = [np.pi/2.0, 0.0, -0.1*np.pi]
    intersection_point_quess = np.mean(data_set_two, axis=0)

    dist_obj = cInterAngle()
    dist_obj.inter_pt = intersection_point_quess
    #kmeansline = cKMeans('membership', dist_obj, iter_max=100, num_clust_centers=len(lines_true_lst))
    kmeansline = cKMeans('centers', dist_obj, cluster_centers=cluster_center_guess, iter_max=200)
    kmeansline.fit(data_set_two)
    for center_km in kmeansline.cluster_centers:
        print(center_km)

    fig = plt.figure(2)
    axis = fig.add_subplot(111)
    axis.scatter(data_set_two[:, 0], data_set_two[:, 1], s=80, c=0.25 * (1 + kmeansline.memberships_data2clusters),
                 marker="x")
    for center_km in kmeansline.cluster_centers:
        pts = np.array([dist_obj.inter_pt, dist_obj.inter_pt + np.array([-np.sin(center_km), np.cos(center_km)])]).T
        axis.plot(pts[0,:], pts[1,:], color='red')
    for line_true in lines_true_lst:
        x_lst, y_lst = get_pts_on_line(line_true)
        axis.plot(x_lst, y_lst, color='black')
    plt.show()

def test_comparison():
    ### Create lines with one intersection point

    ###

    ### Function Score: (max(D_true, D_algo) - D_true)/ D_true => 0: Perfect fit , infty => worst fit
    pass

if __name__ == '__main__':
    #test_k_means()
    #test_k_means_lines()
    #test_k_means_angles_inter()

    ### Test 4: Combi: KMeans (2DNorm) -> Memberships; KMeans (Angle+Inter) über Membership und 1. calculation of center
    lines_true_lst = [np.array([0.0, 0.0]), np.array([1.4, 0.0]), np.array([2.5, 0.0])]
    # data_set_two = np.array([[1,2], [3,4]])
    data_set_two = create_pts_around_line(lines_true_lst[0], 1, 2, 0.05, size=30)
    print(lines_true_lst[0])
    for line_true in lines_true_lst[1:]:
        random_pts = create_pts_around_line(line_true, 1, 2, 0.05, size=30)
        data_set_two = np.concatenate((data_set_two, random_pts), axis=0)
        print(line_true)

    # 1. Kmeans (2DNorm)
    dist_obj_2dnorm = cCenterPoint(2)
    cluster_center_init = [data_set_two[np.random.randint(0, len(data_set_two))]]
    for idx in range(len(lines_true_lst)-1):
        dist_obj_2dnorm.dist_to_center() data_set_two -
        cluster_center_init.append()


    # kmeansline = cKMeans('membership', dist_obj, iter_max=100, num_clust_centers=len(lines_true_lst))

    kmeans2dnorm = cKMeans('kmeans++', dist_obj_2dnorm, iter_max=200) # membership kmeans
    kmeans2dnorm.fit(data_set_two)

    kmeans2dnorm.memberships_data2clusters

    dist_obj_angle = cInterAngle()
    kmeansAngle = cKMeans('membership', dist_obj_angle, iter_max=100, num_clust_centers=len(lines_true_lst))
    kmeansAngle.fit(data_set_two)

    for center_km in kmeansAngle.cluster_centers:
        print(center_km)

    fig = plt.figure(2)
    axis = fig.add_subplot(111)
    axis.scatter(data_set_two[:, 0], data_set_two[:, 1], s=80, c=0.25 * (1 + kmeansAngle.memberships_data2clusters),
                 marker="x")
    for center_km in kmeansAngle.cluster_centers:
        pts = np.array([dist_obj_angle.inter_pt, dist_obj_angle.inter_pt + np.array([-np.sin(center_km), np.cos(center_km)])]).T
        axis.plot(pts[0,:], pts[1,:], color='red')
    for line_true in lines_true_lst:
        x_lst, y_lst = get_pts_on_line(line_true)
        axis.plot(x_lst, y_lst, color='black')
    plt.show()

