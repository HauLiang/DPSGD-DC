import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)

fig, ax = plt.subplots()

batch_sizes = np.array([100, 200, 300, 400, 600,  1000])
epochs = np.array([50, 100, 150, 200, 250, 300, 350, 400])

i = -1

for batch_size in batch_sizes:
    # File path
    file_name = f'./privacy_level/epsilon_results_{batch_size}.txt'
    
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
        print('Handling the {}-th file'.format(i+1))
        if i+1 == 1:
            ax.plot(epochs, np.mean(data, axis=1), '-', color='darkgoldenrod', marker='o', markersize = 7, linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='goldenrod', alpha=0.1)


        elif i+1 == 2:
            ax.plot(epochs, np.mean(data, axis=1), '--', color='darkblue', marker='s', markersize = 7,linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='blue', alpha=0.1)

            
        elif i +1== 3:
            ax.plot(epochs, np.mean(data, axis=1), '-.', color='darkorange', marker='D', markersize = 7,linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='orange', alpha=0.1)


        elif i +1== 4:
            ax.plot(epochs, np.mean(data, axis=1), ':', color='darkred', marker='P', markersize = 7,linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='red', alpha=0.1)


        elif i +1== 5:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='-.', color='darkcyan', marker='v',markersize = 7, linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='cyan', alpha=0.1)

        
        elif i +1== 6:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='--', color='darkviolet', marker='*', markersize = 7,linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='violet', alpha=0.1)


ax.set_xlabel('Number of epochs $E$')
ax.set_ylabel('Estimated privacy level')
ax.grid(True)
fig.tight_layout()
fig.legend(loc = (0.145, 0.6), fontsize=13)

plt.savefig('figure_2a.pdf', format='pdf', dpi=600)
plt.show()

