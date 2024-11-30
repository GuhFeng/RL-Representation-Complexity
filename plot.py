import numpy as np
import matplotlib.pyplot as plt
fit_models = ["model", "reward", "policy", "value"]
types = ["next_state", "reward", "action", "Q_value"]
envs = ["HalfCheetah-v4", "Humanoid-v4", "InvertedPendulum-v4", "Ant-v4"]


def plot(size, hidden_dim, num_layer):
    data = {}
    std = {}
    for fit_model, type in zip(fit_models, types):
        results = []
        stds = []
        for env in envs:
            result = []
            for i in range(4):
                result += [np.loadtxt(
                    f"./Error_stats/{env}-{fit_model}-{i}-{hidden_dim}-{num_layer}-{size}.txt")[-2]]
            result = np.array(result)
            result = result**.5
            results += [result.mean()]
            stds += [result.std()]
            print(f"{fit_model}-{env}:{result.mean():.4f}±{result.std():.4f}")
        data[fit_model] = results
        std[fit_model] = stds
    width = 0.16
    x_list = [[], [], [], []]
    colors = ["#A72EA9", "#3F7A96", "#76AECC", "#A9DDEF"]
    for i in range(4):
        x_list[0].append(i)
        x_list[1].append(i + width)
        x_list[2].append(i + 2 * width)
        x_list[3].append(i + 3 * width)

    labels = ["Transition Kernel", "Reward Function",
              "Optimal Policy", "Optimal Q-Function"]

    plt.figure()
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plt.ylabel("Approximation  Error")
    error_params = dict(elinewidth=0.8, ecolor='black', capsize=1.6)
    i = 0
    for k in data.keys():
        if i == 2:
            plt.bar(x_list[i], data[k], width=width, label=labels[i], align='edge', tick_label=[
                    env[:-3] for env in envs], color=colors[3 - i], yerr=std[k], error_kw=error_params)
        else:
            plt.bar(x_list[i], data[k], width=width, label=labels[i], align='edge',
                    color=colors[3 - i], yerr=std[k], error_kw=error_params)
        i += 1

    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(f"./figures/{size}-{hidden_dim}-{num_layer}.pdf")

for size in [300000]:
    for hidden_dim in [32, 48,64]:
        for num_layer in [2, 3]:
            plot(size, hidden_dim, num_layer)
