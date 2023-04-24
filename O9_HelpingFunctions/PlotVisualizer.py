### All Visualizations

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import ticker

class cVisualizer():
    def __init__(self, num_fig):
        self.fig = plt.figure(num_fig)
        self.ax = self.fig.add_subplot(111)

    def get_XYZ_arrays_from_df(self, x_label, y_label, z_label, df):
        df = df.sort_values(by = [x_label, y_label], ascending=[True, True])
        x_arr = df[x_label].values
        y_arr = df[y_label].values
        z_arr = df[z_label].values
        shape = (len(np.unique(x_arr)), len(np.unique(y_arr)))
        X = np.reshape(x_arr, shape)
        Y = np.reshape(y_arr, shape)
        Z = np.reshape(z_arr, shape)
        return X, Y, Z

    def plot_potential2d_plot_from_df(self, x_label, y_label, z_label, df, **kwargs):
        X, Y, Z = self.get_XYZ_arrays_from_df(x_label, y_label, z_label, df)
        v_max = df[z_label].max()
        v_min = df[z_label].min()

        #CS = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.9, cmap = "RdBu_r", vmin=-v_max, vmax =v_max)
        #cset = ax.contour(X, Y, Z, zdir='y', offset=-30, cmap="RdBu_r")
        #cset = ax.contour(X, Y, Z, zdir='z', offset=-0.1, cmap="RdBu_r")
        #CS = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=-v_max, vmax=v_max)
        #CS = ax.tricontour(x_arr, y_arr, z_arr, levels = 14, cmap = "RdBu_r")

        CS = self.ax.contourf(X, Y, Z, levels=21, cmap="gray")#, vmin=0, vmax = 1,  **kwargs)
        #CS = ax.contourf(X, Y, Z, levels=80, cmap="RdBu_r", vmax=1.0, vmin=0.0)
        #self.ax.set_xscale('log')
        cbar = self.fig.colorbar(CS)
        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.ax.set_title(z_label)
        #plt.show()
        return CS

    def plot_contour_plot_from_df(self,x_label, y_label, z_label, df, **kwargs):
        X, Y, Z = self.get_XYZ_arrays_from_df(x_label, y_label, z_label, df)
        CS = self.ax.contour(X, Y, Z, colors= 'k', **kwargs)
        self.ax.clabel(CS, inline=True, fontsize=10, fmt='%1.2f', **kwargs)
        self.ax.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.ax.set_title(z_label)
        return CS

    def plot_trajectories_from_df(self, x_label, y_label, u_label, v_label, a_label, df, **kwargs):
        X, Y, U = self.get_XYZ_arrays_from_df(x_label, y_label, u_label, df)
        X, Y, V = self.get_XYZ_arrays_from_df(x_label, y_label, v_label, df)
        X, Y, A = self.get_XYZ_arrays_from_df(x_label, y_label, a_label, df)
        SP = self.ax.streamplot(X.T, Y.T, U.T, V.T, density=0.75, color='k')#, linewidth=A.T)
        return SP

    def plot_potential2d_surface_plot_from_df(self, x_label, y_label, z_label, df, **kwargs):
        self.ax = self.fig.add_subplot(111, projection = '3d')
        X, Y, Z = self.get_XYZ_arrays_from_df(x_label, y_label, z_label, df)
        v_max = df[z_label].max()
        v_min = df[z_label].min()

        CS = self.ax.plot_surface(X, Y, Z, rstride=15, cstride=15, alpha=0.9, cmap = "Greys", vmin=v_min, vmax =v_max,  **kwargs)

        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.ax.set_title(z_label)
        return CS

    def plot_3dcurve_family_from_df(self, x_label, y_label, z_label, p_label, df):
        self.ax = self.fig.add_subplot(111, projection='3d')
        df = df.sort_values(by=[x_label, p_label], ascending=[True, True])
        for p_val in np.unique(df[p_label].values):
            df_p = df[df[p_label] == p_val]
            self.ax.plot(df_p[x_label], df_p[y_label], df_p[z_label], label = str(p_label) + '=' + str(p_val))

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_zlabel(z_label)

        self.ax.set_zlim((-0.001, 0.001))

    def plot_curve_family_from_df(self, x_label, p_label, y_label, df, **kwargs):
        df = df.sort_values(by=[x_label, p_label], ascending=[True, True])
        for p_val in np.unique(df[p_label].values):
            df_p = df[df[p_label] == p_val]
            self.ax.plot(df_p[x_label], df_p[y_label], label = str(p_label) + '=' + str(p_val), **kwargs)

        self.set_xlabel(x_label)
        self.set_ylabel(y_label)

    def plot_curve_from_df(self, x_label, y_label, df, **kwargs):
        self.ax.plot(df[x_label], df[y_label], **kwargs)
        self.set_xlabel(x_label)
        self.set_ylabel(y_label)

    def plot_from_lst(self, list_x, list_y, x_label, y_label, **kwargs):
        self.ax.plot(list_x, list_y, **kwargs)
        self.set_xlabel(x_label)
        self.set_ylabel(y_label)

    def plot_probabilisitic_region(self, df, x_label, y_label, **kwargs):
        y_mean = df[y_label + 'mean']
        y_std = df[y_label + 'std']
        self.ax.plot(df[x_label], y_mean, **kwargs)
        self.ax.fill_between(df[x_label], y_mean + y_std, y_mean - y_std, **kwargs)

    def plot_boxplots_from_lst(self, data, labels, boxplot_b = False, **kwargs):
        if boxplot_b:
            self.ax.boxplot(data, **kwargs)
            self.ax.yaxis.set_tick_params(direction='out')
            self.ax.set_yticks(np.arange(1, len(labels) + 1))
            self.ax.set_yticklabels(labels)

        else:
            self.ax.boxplot(data,**kwargs)
            #self.ax.violinplot(data, **kwargs)
            self.ax.xaxis.set_tick_params(direction='out')
            self.ax.set_xticks(np.arange(1,len(labels)+1))
            self.ax.set_xticklabels(labels)

    def set_xlabel(self, x_label):
        self.ax.set_xlabel(x_label)

    def set_ylabel(self, y_label):
        self.ax.set_ylabel(y_label)

    def plot_all(self, **kwargs):
        self.ax.legend()
        plt.show(**kwargs)

    def safe_as_latex(self, name):
        import tikzplotlib
        self.ax.legend()
        file_name = "C:/Users/Fabian/Documents/TUD Promotion/Thesis/figures/results/" + name + ".tex"
        tikzplotlib.save(file_name, figure = self.fig)
        self.delete_praembles(file_name)

    def delete_praembles(self, file_name):
        a_file = open(file_name)
        list_of_lines = a_file.readlines()
        idx_removals_lst = []
        axis_b = False
        for idx, line in enumerate(list_of_lines):
            if line.find('begin{axis}')!= -1:
                axis_b = True
            if axis_b or (line.find('tikzpicture')!= -1) or (line.find('end{axis}')!= -1) or (line.find('addlegendentry{') != -1):
                idx_removals_lst.append(idx)
            if axis_b and (line.find(']')!= -1):
                axis_b = False
        for idx in sorted(idx_removals_lst, reverse=True):
            del list_of_lines[idx]
        a_file = open(file_name, "w")
        a_file.writelines(list_of_lines)
        a_file.close()