import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.io import loadmat
import re
from scipy.optimize import minimize_scalar

# -----------------------------
T = np.arange(0, 301)   # 0:300

alpha = 1.1
L = 1
C = 2
sigma = 4
D = 1
n = 16
b = 2
eta = 0.2

# Privacy expression
privacy = lambda beta: (
    2 * alpha * C**2 / beta / n / b / sigma**2
    + alpha * (1 + eta * L)**2 * D**2 / (2 * eta**2 * sigma**2 * (1 - beta))
)

# Minimize privacy(beta) on (0,1)
res = minimize_scalar(privacy, bounds=(1e-12, 1 - 1e-12), method='bounded')
upper = res.fun

# Trivial upper bound (naive)
naive_value = 2 * alpha * (D + eta/b * C)**2 / (eta**2 * sigma**2)
naive = np.ones_like(T) * naive_value

# Our bound
our = 2 * alpha * C**2 / (n * b * sigma**2) * T
our = np.minimum(our, upper)


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)

fig, ax = plt.subplots()

T = np.arange(0, 301)

ax.plot(T, naive, '--', color='darkred', linewidth=2.5, label="Trivial upper bound".format(1))
ax.plot(T, our, '-', color='darkblue', linewidth=2.5, label="Ours (Theorem 3.4)".format(1))

x_start, y_start = 250, 1.6
x_end, y_end = 250, 4.9

ax.annotate(
    '',
    xy=(x_end, y_end),
    xytext=(x_start, y_start),
    arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
)

ax.text(205, 3.5, 'Significant Gap',
        fontsize=15, color='black',
        ha='center', va='center')

ax.set_xlabel('Number of iterations $T$')
ax.set_ylabel('Theoretical privacy level $\epsilon$')
ax.grid(True)
fig.tight_layout()
fig.legend(loc = (0.17, 0.647), fontsize=13)

plt.savefig('figure_5.pdf', format='pdf', dpi=600)
plt.show()

