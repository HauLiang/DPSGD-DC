import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.io import loadmat
import re
from matplotlib.patches import Ellipse
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


def draw_brace(ax, x, y_start, y_end, text=None):
    path = TextPath((0, 0), r"$\}$", size=1.2, prop=dict(size=15))
    trans = Affine2D().scale(0.1, y_end - y_start).translate(x, (y_start + y_end) / 2)
    patch = PathPatch(trans.transform_path(path), lw=2, color='gray', fill=False)
    ax.add_patch(patch)


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


fig, ax = plt.subplots()

batch_size = 400
batch_size2 = 100
D = np.array([20, 60, 100])

epochs = np.array([50, 100, 150, 200, 250, 300, 350, 400,450,500,550,600,650,700,750,800,850,900,950,1000])

reds = [plt.cm.Reds(i) for i in np.linspace(0.6, 1.0, 6)]
blues = [plt.cm.Blues(i) for i in np.linspace(0.4, 0.8, 6)]

i=-1

for D_index in D:
    file_name = f'./privacy_level_with_D/epsilon_results_{batch_size2}_D_{D_index}.txt'

    with open(file_name, 'r') as fid:
    
        fid.readline()

        data = []

        for line in fid:
            if 'Monte Carlo Run' in line:

                run_data = line.split(':')[1].strip()
                run_values = list(map(float, run_data.split(',')))
                data.append(run_values)

        data = np.array(data).T

        i +=1
        print('Handling the {}-th file'.format(i+1))
        if i+1 == 1:
            ax.plot(epochs, np.mean(data, axis=1), '-', color=reds[5], marker='o', markersize = 7, linewidth=2.5, label=r"D$=${}".format(D_index))


        elif i+1 == 2:
            ax.plot(epochs, np.mean(data, axis=1), '--', color=reds[4], marker='s', markersize = 7, linewidth=2.5, label=r"D$=${}        b$=$100".format(D_index))


        if i +1== 3:
            ax.plot(epochs, np.mean(data, axis=1), '-.', color=reds[3], marker='D', markersize = 7, linewidth=2.5, label=r"D$=${}".format(D_index))



i = -1

for D_index in D:

    file_name = f'./privacy_level_with_D/epsilon_results_{batch_size}_D_{D_index}.txt'

    with open(file_name, 'r') as fid:

        fid.readline()

        data = []

        for line in fid:
            if 'Monte Carlo Run' in line:
                run_data = line.split(':')[1].strip()
                run_values = list(map(float, run_data.split(',')))
                data.append(run_values)

        data = np.array(data).T
        i += 1
        print('Handling the {}-th file'.format(i + 4))
            
        if i +1== 1:
            ax.plot(epochs, np.mean(data, axis=1), ':', color=blues[5], marker='P', markersize = 7, linewidth=2.5, label=r"D$=${}".format(D_index))

            
        elif i +1== 2:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='-.', color=blues[4], marker='v', markersize = 7, linewidth=2.5, label=r"D$=${}".format(D_index))


        elif i +1== 3:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='--', color=blues[3], marker='*',  markersize = 7, linewidth=2.5, label=r"D$=${}".format(D_index))


ax.set_xlabel('Number of epochs $E$')
ax.set_ylabel('Estimated privacy level')
ax.grid(True)
fig.tight_layout()


ellipse = Ellipse(
    xy=(100, 0.2),
    width=40,
    height=0.2,
    angle=0,
    color="black",
    fill=False,
    alpha=1.0,
    linewidth=2,
    zorder=10,
)
ax.add_patch(ellipse)

ellipse2 = Ellipse(
    xy=(100, 0.53),
    width=40,
    height=0.3,
    angle=0,
    color="black",
    fill=False,
    alpha=1.0,
    linewidth=1.5,
    zorder=10,
)
ax.add_patch(ellipse2)

_ = ax.annotate('b$=$100',
            xy = (98, 0.69), xycoords = 'data',
            xytext = (-33, 36), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('b$=$400',
            xy = (112, 0.11), xycoords = 'data',
            xytext = (35, -10), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$100',
            xy = (350, 0.76), xycoords = 'data',
            xytext = (30, -120), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$60',
            xy = (500, 0.78), xycoords = 'data',
            xytext = (37, -123), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$20',
            xy = (700, 0.59), xycoords = 'data',
            xytext = (22, -87), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))


_ = ax.annotate('D$=$100',
            xy = (450, 1.17), xycoords = 'data',
            xytext = (-40, 30), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$60',
            xy = (350, 1.0001), xycoords = 'data',
            xytext = (-60,62), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$20',
            xy = (250, 0.79), xycoords = 'data',
            xytext = (-80, 102), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))


plt.savefig('figure_3a.pdf', format='pdf', dpi=600)
plt.show()

