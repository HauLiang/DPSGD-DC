import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize_scalar

# ============================================

E = np.arange(0, 401)

alpha = 1.1
L = 0.1
C = 20
sigma = 0.002
D = 20
n = 10000
eta = 0.1

b_set = np.array([100, 200, 300, 400, 600, 1000])
our = np.zeros((len(b_set), len(E)))


# ----- 计算 privacy(beta) 的上界“upper” -----
privacy = lambda beta: (
    2 * alpha * C**2 / beta / n / sigma**2
    + alpha * (1 + eta * L)**2 * D**2 / (2 * eta**2 * sigma**2 * (1 - beta))
)

res = minimize_scalar(privacy, bounds=(1e-12, 1 - 1e-12), method='bounded')
upper = res.fun


# ----- 计算 our(i,:) = min( linear_T , upper ) -----
for i in range(len(b_set)):
    b = b_set[i]
    T = E * n / b

    linear_term = 2 * alpha * C**2 / (n * b * sigma**2) * T
    our[i, :] = np.minimum(linear_term, upper)

# ----- 归一化 normal_our -----
normal_our = our / np.max(our)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


fig, ax = plt.subplots()

E = np.arange(0, 401)
batch_set = np.array([100, 200, 300, 400, 600, 1000])
color_set = ["darkgoldenrod", "darkblue", "darkorange", "darkred", "darkcyan", "darkviolet"]
label_set = ["b$=$100","b$=$200","b$=$300","b$=$400","b$=$600","b$=$1000"]

line_set = ['-', '--', '-.', ':', (0, (3, 2, 1, 5)), (0, (3, 5, 1, 5))]

for i in range(batch_set.size):

    normal_our_i = normal_our[i]
    normal_our_i = np.array(normal_our_i).reshape((E.size,))

    choose_color = color_set[i]
    choose_label = label_set[i]
    choose_line = line_set[i]

    ax.plot(E, normal_our_i, linestyle = choose_line, color=choose_color, linewidth=2.5, label=choose_label.format(1))


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

ax.set_xlabel('Number of epochs $E$')
ax.set_ylabel('Normalized theoretical privacy level $\epsilon$')
ax.grid(True)
fig.legend(loc = (0.145, 0.52), fontsize=13)

fig.subplots_adjust(bottom=0.12)
# fig.tight_layout(pad=13.0)

plt.savefig('figure_2b.pdf', format='pdf', dpi=600)
plt.show()

