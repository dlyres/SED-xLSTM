import os
import matplotlib.pyplot as plt
import numpy as np


def show(args, model_auc_cross, model_list):
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 5))
    num_models = len(model_auc_cross)
    x1 = np.arange(1, len(model_auc_cross[0]) + 1)
    plt.yticks(np.linspace(0, 1, num=11))
    plt.ylim([0.8, 1])
    color_list = ['red', 'blue', 'green', 'yellow', 'cyan']
    labels = ["1 fold", "2 fold", "3 fold", "4 fold", "5 fold", "AVG"]
    colors = color_list[:num_models]
    bar_width = 0.8 / num_models
    width = 0
    for acc_list, color in zip(model_auc_cross, colors):
        plt.xticks(x1 + (bar_width * (num_models - 1) / 2), labels=labels)
        plt.bar(x1 + width, acc_list, color=color, width=bar_width)
        width += bar_width
    plt.ylabel("AUC")
    plt.legend(labels=model_list, loc='upper left', fontsize='small')
    # plt.show()

    models = '_'.join(model_list)
    save_dir = os.path.join(os.path.dirname(__file__), f'{args.dataset_name}/cross_validation')
    save_path = os.path.join(save_dir, f'{models}_auc.png')

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path)