from matplotlib import pyplot as plt
import os
import numpy as np

def plot_one(x,y, x_label, y_label, label, title, dir_name, fig_name):
    """
    plot x and y values and save the plot in the specified directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.figure(figsize=(10, 5))
    if x is None or (np.isscalar(x) and np.isnan(x)):
        x = np.arange(len(y))
    plt.plot(x, y, label=label)
    # plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = fig_name
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300)
    plt.show()
    
def plot_two(x1,y1, x2, y2, x_label, y_label, title, label1, label2, dir_name, fig_name):
    """
    plot 2 values of x and y on the same graph, and save the plot in the specified directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.figure(figsize=(10, 5))
    if x1 is None or (np.isscalar(x1) and np.isnan(x1)):
        x1 = np.arange(len(y1))
    if x2 is None or (np.isscalar(x2) and np.isnan(x2)):
        x2 = np.arange(len(y2))
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    # plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = fig_name
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    full_path = os.path.join(dir_name, fig_name)
    plt.savefig(full_path, dpi=300)
    plt.show()