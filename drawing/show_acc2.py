import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


matplotlib.use('Agg')

matplotlib.rcParams.update({'font.size': 15})

def show(total_acc_list, final_class_acc, args, model_list=None, num_fold=None):
    plt.figure(figsize=(10, 6), facecolor='white')

    x1 = np.arange(1, args.epochs + 1)
    plt.xlabel("Training epochs")
    plt.ylabel("Acc(%)")
    plt.yticks(np.linspace(0, 100, num=11))
    plt.ylim([0, 100])
    markers = ['o', '*', 'D']

    num_models = len(model_list)
    color_list = ['red', 'blue', 'orange']
    linestyle_list = ['-', '-', '--']
    colors = color_list[:num_models]
    for acc_list, color, linestyles, marker in zip(total_acc_list, colors, linestyle_list, markers):
        y1 = acc_list * 100
        plt.plot(x1, y1, color=color, linestyle=linestyles)
    plt.legend(labels=model_list, loc='lower right', prop={'size': 15})
    plt.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5)
    plt.show()

    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, 'acc.png')

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)

def show_each(total_acc_list, final_class_acc, args, model_list=None, num_fold=None):
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    stimulus = np.array([8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6])
    width = 0
    color_list = ['red', 'blue', 'yellow', 'green', 'cyan']
    num_models = len(model_list)
    colors = color_list[:num_models]
    bar_width = 0.8 / num_models
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel("Stimulus Frequency(Hz)")
    ax.set_ylabel("Acc(%)")
    ax.set_yticks(np.linspace(0, 100, num=11))
    ax.set_ylim([60, 100])
    for acc_list, class_acc, model_name, color in zip(total_acc_list, final_class_acc, model_list, colors):
        x2 = np.arange(1, len(class_acc) + 1)
        y2 = class_acc * 100
        ax.set_xticks(x2 + (bar_width * (num_models - 1) / 2))
        ax.set_xticklabels([f'{i}' for i in stimulus])

        plt.bar(x2 + width, y2, color=color, width=bar_width)
        width += bar_width
    plt.legend(labels=model_list, loc='upper center', prop={'size': 15}, ncol=3)
    plt.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    model_list = ['SSGFormer', 'VGT', 'VIT']
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_models', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, default='Benchmark')
    args = parser.parse_args()
    models_acc_list = np.random.rand(3, 50)
    models_class_list = np.random.rand(3, 8)

    show_each(models_acc_list, models_class_list, args, model_list=model_list)