import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import snntorch.spikeplot as splt
import imageio
import os

labels_map = {
  -1: "unclassified",
   0: "proton",
   1: "kaon",
   2: "pion",
   3: "other"
}
nClasses = len(labels_map)-1

def get_views(data):
    data_views = []

    # Z vs. X
    data_zx = torch.sum(data, 0).view(10, -1)
    data_views.append(data_zx)
    # X vs. t
    data_tzx = data.view(100, 10, -1)
    data_tx = torch.sum(data_tzx, 1)
    data_views.append(data_tx.t())
    # Z vs. t
    data_tz = torch.sum(data_tzx, 2)
    data_views.append(data_tz.t())

    return data_views

def get_timeslice(data, t):
    data_tzx = data.view(100, 10, -1)
    return data_tzx[t]


def plot_views(index, dataset, save=False):
    try:
        iter(index)
    except TypeError:
        index = [index]
        
    n_ex, views = len(index), 3
    ax_labels = ["Sensor id. X", "Timestep", "Timestep"]
    ay_labels = ["Sensor id. Z", "Sensor id. X", "Sensor id. Z"]
    left = [0.05, 0.4, 0.75]

    for i in range(n_ex):
        print(f"{index[i]}:")
        fig, ax = plt.subplots(1, views, figsize=(3*views, 3))        
        data, p_class = dataset[index[i]]
        data_views = get_views(data)
        for j in range(views):
            # ax[j].set_title(labels_map[p_class])
            ax[j].set_xlabel(ax_labels[j], fontsize=20)
            ax[j].set_ylabel(ay_labels[j], fontsize=20)
            ax[j].set_position([left[j], 0.05, 0.27, 0.8])
            if ax_labels[j] != 'Timestep':
                ax[j].set_xticks(np.arange(0,11,1))
                ax[j].set_xticks(np.arange(-0.5,11,1), minor=True)
            else:
                ax[j].set_xticks(np.arange(0,110,10))
                ax[j].set_xticks(np.arange(-0.5,100.5,10), minor=True)
            ax[j].set_yticks(np.arange(0,11,1))
            ax[j].set_yticks(np.arange(-0.5,11,1), minor=True)
            ax[j].grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
            ax[j].tick_params(which='minor', bottom=False, left=False)
            im = ax[j].imshow(data_views[j], cmap="hot", norm=SymLogNorm(linthresh=1, vmin=0, vmax=1e6), aspect='auto')
                
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('SymLog Scale')
        if save:
            plt.savefig(f"Images/views_{index[i]}.png",  bbox_inches='tight')
        plt.show()


def plot_time_views(index, dataset, t=None):
    try:
        iter(index)
    except TypeError:
        index = [index]
    n_ex = len(index)
    ax_labels = ["Sensor id. X"]
    ay_labels = ["Sensor id. Z"]
    

    for i in range(n_ex):
        print(f"{index[i]}:")
              
        data, p_class = dataset[index[i]]
        if t is None:
            t_indices = (data != 0).any(dim=1).nonzero(as_tuple=True)[0]
        else:
            t_indices = t
        data_views = get_timeslice(data, t_indices)
        
        rows = (len(t_indices)-1)//3+1
        columns = min(3, len(t_indices))
        fig, ax = plt.subplots(rows, columns, figsize=(3*columns, 3*rows)) 
        ax = ax.ravel()
        for j in range(len(t_indices)):
            ax[j].set_title(f"{labels_map[p_class]}: t={t_indices[j]}")
            ax[j].set_xlabel(ax_labels[0])
            ax[j].set_ylabel(ay_labels[0])
            ax[j].set_xticks(np.arange(0,11,1))
            ax[j].set_xticks(np.arange(-0.5,11,1), minor=True)
            ax[j].set_yticks(np.arange(0,11,1))
            ax[j].set_yticks(np.arange(-0.5,11,1), minor=True)
            ax[j].grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
            ax[j].tick_params(which='minor', bottom=False, left=False)
            im = ax[j].imshow(data_views[j], cmap="hot", norm=SymLogNorm(linthresh=1, vmin=0, vmax=1e6), aspect='auto')
                
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('SymLog Scale')
        plt.show()


def plot_time_gif(index, dataset, cleanup=True):
    try:
        iter(index)
    except TypeError:
        index = [index]
    n_ex = len(index)
    ax_labels = ["Sensor id. X"]
    ay_labels = ["Sensor id. Z"]
    

    for i in range(n_ex):
        filenames = []
        print(f"{index[i]}:")
              
        data, p_class = dataset[index[i]]
        data_views = data.view(100, 10, -1)
        
        for j in range(data_views.shape[0]):
            plt.figure(figsize=(3, 3)) 
            #plt.title(f"{labels_map[p_class]}: t={j}")
            plt.title(f"t={j}", fontsize=20)
            plt.xlabel(ax_labels[0], fontsize=20)
            plt.ylabel(ay_labels[0], fontsize=20)
            plt.xticks(np.arange(0,11,1))
            plt.xticks(np.arange(-0.5,11,1), minor=True)
            plt.yticks(np.arange(0,11,1))
            plt.yticks(np.arange(-0.5,11,1), minor=True)
            plt.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
            plt.tick_params(which='minor', bottom=False, left=False)
            im = plt.imshow(data_views[j], cmap="hot", norm=SymLogNorm(linthresh=1, vmin=0, vmax=1e6), aspect='auto')
                
            cbar = plt.colorbar(im, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('SymLog Scale')
            filename = f'plot_{j}.png'
            plt.savefig(filename, bbox_inches='tight')
            filenames.append(filename)
            plt.close()

        with imageio.get_writer(f'animated_plot_{index[i]}.gif', mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)
        
        if cleanup:
            for filename in filenames:
                os.remove(filename)