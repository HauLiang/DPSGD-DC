import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize_scalar
from matplotlib.patches import Ellipse


# ==============================
E = np.arange(0, 1001)   # 0:1000
alpha = 1.1
L = 0.1
C = 20
sigma = 0.002
n = 1000
eta = 0.1

b_set = np.array([100, 400])
D_set = np.array([2.0, 2.5, 3.0])
our = np.zeros((len(b_set) * len(D_set), len(E)))


# ==============================
flag = 0
for i in range(len(b_set)):
    b = b_set[i]
    T = E * n / b
    for j in range(len(D_set)):
        D = D_set[j]

        # privacy = @(beta) 2*alpha*C^2/beta/b/n/sigma^2 + alpha*(1+eta*L)^2*D^2/2/sigma^2/(1-beta);
        def privacy(beta):
            return (
                2 * alpha * C**2 / (beta * b * n * sigma**2)
                + alpha * (1 + eta * L)**2 * D**2 / (2 * sigma**2 * (1 - beta))
            )

        res = minimize_scalar(privacy, bounds=(1e-12, 1 - 1e-12), method='bounded')
        upper = res.fun

        cur = 8 * alpha * C**2 / (n**2 * sigma**2) * T
        cur = np.minimum(cur, upper)

        our[flag, :] = cur
        flag += 1

normal_our = our / np.max(our)


# ==============================
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['mathtext.cal'] = "STIXGeneral:script"

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


fig, ax = plt.subplots()

batch_set = np.array([100, 400])
color_set = ["darkgoldenrod", "darkblue", "darkorange", "darkred", "darkcyan", "darkviolet"]
label_set = ["b$=$100","b$=$400"]
D_set = np.array(["D=20","D=60","D=100"])

reds = [plt.cm.Reds(i) for i in np.linspace(0.6, 1.0, 6)]
blues = [plt.cm.Blues(i) for i in np.linspace(0.4, 0.8, 6)]

line_set = ['-', '--', '-.', ':', (0, (3, 2, 1, 5)), (0, (3, 5, 1, 5))]

flag = 0
for i in range(batch_set.size):
    for j in range(D_set.size):
        if i == 0:
            choose_color = reds[5-j]
        else:
            choose_color = blues[5-j]

        choose_line = line_set[flag]
        normal_our_i = normal_our[flag]
        normal_our_i = np.array(normal_our_i).reshape((E.size,))

        ax.plot(E, normal_our_i, color = choose_color, linestyle = choose_line, linewidth=2.5)

        flag += 1


x_start, y_start = 250, 1.6
x_end, y_end = 250, 4.9


x1, y1 = 0, 0
x2, y2 = 200, 0.45


m = (y2 - y1) / (x2 - x1)
theta = np.degrees(np.arctan(m))


ax.set_xlabel('Number of epochs $E$')
ax.set_ylabel('Normalized theoretical privacy level $\epsilon$')
ax.grid(True)


ell = Ellipse(
    (0.13, 0.63),
    width=0.07, height=0.45,
    angle=-8,
    transform=ax.transAxes,
    edgecolor='black', facecolor='none', linewidth=2, zorder=20
)
ax.add_patch(ell)

ell2 = Ellipse(
    (0.39, 0.6),
    width=0.07, height=0.45,
    angle=-30,
    transform=ax.transAxes,
    edgecolor='black', facecolor='none', linewidth=2, zorder=20
)
ax.add_patch(ell2)

_ = ax.annotate('b$=$100',
            xy = (78, 0.78), xycoords = 'data',
            xytext = (-38, 36), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('b$=$400',
            xy = (370, 0.67), xycoords = 'data',
            xytext = (-33, 36), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$100',
            xy = (600, 1), xycoords = 'data',
            xytext = (5, -55), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$100',
            xy = (800, 0.965), xycoords = 'data',
            xytext = (5, -55), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$60',
            xy = (600, 0.71), xycoords = 'data',
            xytext = (5, -55), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$60',
            xy = (800, 0.675), xycoords = 'data',
            xytext = (5, -47), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$20',
            xy = (600, 0.465), xycoords = 'data',
            xytext = (5, -55), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))

_ = ax.annotate('D$=$20',
            xy = (800, 0.45), xycoords = 'data',
            xytext = (5, -55), textcoords = 'offset points', fontsize = 16,
            arrowprops=dict(arrowstyle='-|>',color = 'black', lw = 1.5, connectionstyle="arc3,rad=0"))


ax.axvline(
    x=112,
    ymin=0, ymax=0.69,
    linestyle='--',
    linewidth=1.5,
    color='black',
    label='x = 3'
)

ax.axhline(
    y=0.71,
    xmin=0, xmax=0.14,
    linestyle='--',
    linewidth=1.5,
    color='black',
    label='y = 0.5'
)

xticks = ax.get_xticks()
ax.set_xticklabels([' '] * len(xticks), fontsize=14)

vline_x = 112
ax.annotate(r'$\mathcal{O}(\frac{(1+\eta L)^2nbD^2}{\eta^2 C^2})$',
    xy=(vline_x, 0),
    xytext=(0, -15),
    textcoords='offset points',
        ha='center', va='top', fontsize=14,
        transform=ax.get_xaxis_transform())

yticks = ax.get_yticks()
ax.set_yticklabels([' '] * len(yticks), fontsize=58)

vline_y = 0.75
ax.annotate(r'$\mathcal{O}\left(\frac{\alpha(1+\eta L)^2D^2}{\eta^2 \sigma_{\mathrm{DP}}^2}\right)$',
    xy=(0, vline_y),
    xytext=(-43, -15),
    rotation=90,
    rotation_mode='anchor',
    textcoords='offset points',
        ha='center', va='top', fontsize=14,
        transform=ax.get_xaxis_transform())


fig.subplots_adjust(bottom=0.12)


plt.savefig('figure_3b.pdf', format='pdf', dpi=600)
plt.show()

