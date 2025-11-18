import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize_scalar

# ==============================
T = np.arange(0, 201)   # 0:200

alpha = 1.1
L = 1
C = 2
sigma = 4
D = 1
n = 8
b = 2
eta = 0.2

# ==============================

# Feldman et al. 2018
feld = 2 * alpha * C**2 / (b**2 * sigma**2) * T

# Standard composition (Mironov 2017 + Mironov et al. 2019)
compos = 8 * alpha * C**2 / (n**2 * sigma**2) * T

# Altschuler & Talwar 2022
apple_bound = 8 * alpha * C**2 / (n**2 * sigma**2) * D * n / eta / C
apple = np.minimum(compos, apple_bound)

# Kong & Ribero 2024
m = 1.0
kong_scalar = (
    alpha / (2 * sigma**2)
    * (D * np.sqrt(1 + 2 * eta * m * (1 + m / 2 / (L + m))) + 2 * eta * C / b) ** 2
    / eta**2
)
kong = np.ones_like(T) * kong_scalar

# Our bound (Theorem 3.4)
privacy = lambda beta: (
    2 * alpha * C**2 / (beta * n * b * sigma**2)
    + alpha * (1 + eta * L)**2 * D**2 / (2 * eta**2 * sigma**2 * (1 - beta))
)

res = minimize_scalar(privacy, bounds=(1e-12, 1 - 1e-12), method='bounded')
upper = res.fun

our = 2 * alpha * C**2 / (n * b * sigma**2) * T
our = np.minimum(our, upper)

# ==============================

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


fig, ax = plt.subplots()

T = np.arange(0, 201)

ax.plot(T, feld, '-.', color='darkgoldenrod', linewidth=2.5, label="Feldman et al. 2018".format(1))
ax.plot(T, compos, '--', color='darkblue', linewidth=2.5, label="Mironov 2017 + Mironov et al. 2019".format(1))
ax.plot(T, apple, '-.', color='darkorange', linewidth=2.5, label="Altschuler & Talwar 2022".format(1))
ax.plot(T, kong, ':', color='darkred', linewidth=2.5, label="Kong & Ribero 2024".format(1))
ax.plot(T, our, '-', color='darkcyan', linewidth=2.5, label="Ours (Theorem 3.4)".format(1))

x_start, y_start = 250, 1.6
x_end, y_end = 250, 4.9

ax.annotate(
    '',
    xy=(x_end, y_end),
    xytext=(x_start, y_start),
    arrowprops=dict(arrowstyle='<->', color='black', linewidth=2)
)

# ax.text(205, 3.5, 'Significant Gap',
#         fontsize=15, color='black',
#         ha='center', va='center')

_ = ax.annotate('Feldman et al. 2018',
            xy = (108, 15), xycoords = 'data',
            xytext = (-38, 58), textcoords = 'offset points', fontsize = 14,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('Mironov 2017 +\nMironov et al. 2019',
            xy = (168, 5.6), xycoords = 'data',
            xytext = (-44, 45), textcoords = 'offset points', fontsize = 14,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('Kong & Ribero 2024',
            xy = (25, 2), xycoords = 'data',
            xytext = (-40, 75), textcoords = 'offset points', fontsize = 14,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('Altschuler & Talwar 2022',
            xy = (78, 0.6), xycoords = 'data',
            xytext = (-60, 41), textcoords = 'offset points', fontsize = 14,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('Ours (Theorem 3.4)',
            xy = (125, 1.5), xycoords = 'data',
            xytext = (-38, 60), textcoords = 'offset points', fontsize = 14,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

ax.set_xlabel('Number of iterations $T$')
ax.set_ylabel('Theoretical privacy level $\epsilon$')
ax.grid(True)
fig.tight_layout()
# fig.legend(loc = (0.15, 0.647), fontsize=13)

plt.savefig('figure_1.pdf', format='pdf', dpi=600)
plt.show()

