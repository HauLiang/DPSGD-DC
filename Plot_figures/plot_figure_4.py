import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 16}

mpl.rc('font', **font)


fig, ax = plt.subplots()

batch_sizes = np.array([100, 200, 300, 400, 600, 1000])
epochs = np.arange(1, 401)
i = -1

for batch_size in batch_sizes:

    file_name = f'./train_loss/train_loss_{batch_size}.txt'
    
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
            ax.plot(epochs, np.mean(data, axis=1), '-', color='darkgoldenrod', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='goldenrod', alpha=0.1)

        elif i+1 == 2:
            ax.plot(epochs, np.mean(data, axis=1), '--', color='darkblue', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='blue', alpha=0.1)
            
        elif i +1== 3:
            ax.plot(epochs, np.mean(data, axis=1), '-.', color='darkorange', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='orange', alpha=0.1)
            
        elif i +1== 4:
            ax.plot(epochs, np.mean(data, axis=1), ':', color='darkred', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='red', alpha=0.1)
            
        elif i +1== 5:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='-.', color='darkcyan', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='cyan', alpha=0.1)
        
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='red', alpha=0.1)
        
        elif i +1== 6:
            ax.plot(epochs, np.mean(data, axis=1), linestyle='--', color='darkviolet', linewidth=2.5, label=r"b$=${}".format(batch_size))

            # Calculate the 95% confidence intervals
            acc_conf_int = np.percentile(data, [2.5, 97.5], axis=1)
            acc_lb = acc_conf_int[0]
            acc_ub = acc_conf_int[1]
            ax.fill_between(epochs, acc_lb[:], acc_ub[:], color='violet', alpha=0.1)


ax.set_xlabel('Number of epochs $E$')
ax.set_ylabel('Training loss')
ax.grid(True)
fig.tight_layout()
fig.legend(loc = (0.727, 0.6),fontsize=13)


plt.savefig('figure_4.pdf', format='pdf', dpi=600)
plt.show()

