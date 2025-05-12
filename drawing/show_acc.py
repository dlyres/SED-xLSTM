import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


matplotlib.use('Agg')

matplotlib.rcParams.update({'font.size': 15})

def show(total_acc_list, final_class_acc, args, model_list=None, num_fold=None):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(bottom=0.2)
    fig, a = plt.subplots(1, 2, figsize=(20, 6), facecolor='white')
    x1 = np.arange(1, args.epochs + 1)
    a[0].set_xlabel("Training epochs")
    a[0].set_ylabel("Acc(%)")
    a[0].set_yticks(np.linspace(0, 100, num=11))
    a[0].set_ylim([0, 100])

    a[1].set_xlabel("Frequency/Phase")
    a[1].set_ylabel("Acc(%)")
    a[1].set_yticks(np.linspace(0, 100, num=11))
    a[1].set_ylim([0, 100])

    if num_fold is None:
        save_dir = os.path.join(os.path.dirname(__file__), f'{args.dataset_name}/no_cross_validation')
        save_path = os.path.join(save_dir, f'{args.model_name}.png')
    else:
        save_dir = os.path.join(os.path.dirname(__file__), f'{args.dataset_name}/cross_validation')
        save_path = os.path.join(save_dir, f'{args.model_name}{num_fold}.png')

    os.makedirs(save_dir, exist_ok=True)

    # 单模型
    if args.num_models == 1:
        y1 = total_acc_list * 100
        a[0].set_title(f"{args.dataset_name}数据集，{args.model_name}模型准确率训练变化趋势", y=1.05)
        a[0].plot(x1, y1)
        a[0].legend(labels=[f'{args.model_name}'], loc='upper right', fontsize='small')

        x2 = np.arange(1, len(final_class_acc) + 1)
        y2 = final_class_acc * 100
        a[1].set_title(f"{args.dataset_name}数据集，{args.model_name}模型每个分类的准确率", y=1.05)
        a[1].set_xticks([i for i in range(1, len(final_class_acc) + 1)])
        a[1].bar(x2, y2, color='blue')
        a[1].legend(labels=[f'{args.model_name}'], loc='upper left', fontsize='small')

    # 多模型非交叉验证
    elif num_fold is None:
        # a[0].set_title(f"{args.dataset_name}数据集，各个模型准确率训练变化趋势", y=1.05)
        # a[1].set_title(f"{args.dataset_name}数据集，各个模型每个分类的准确率", y=1.05)
        num_models = len(model_list)
        models = '_'.join(model_list)
        save_path = os.path.join(save_dir, f'{models}.png')
        stimulus = np.array([8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6])
        width = 0
        color_list = ['red', 'blue', 'yellow', 'green', 'cyan']
        colors = color_list[:num_models]
        bar_width = 0.8 / num_models
        for acc_list, class_acc, model_name, color in zip(total_acc_list, final_class_acc, model_list, colors):
            y1 = acc_list * 100
            a[0].plot(x1, y1)

            x2 = np.arange(1, len(class_acc) + 1)
            y2 = class_acc * 100
            a[1].set_xticks(x2 + (bar_width * (num_models - 1) / 2))
            a[1].set_xticklabels([f'{i} Hz' for i in stimulus])
            a[1].bar(x2 + width, y2, color=color, width=bar_width)
            width += bar_width
        a[0].legend(labels=model_list, loc='upper left', fontsize='small')
        a[1].legend(labels=model_list, loc='upper left', fontsize='small')
    else:
        return None

    # plt.show()
    fig.savefig(save_path)

# 多模型交叉验证可视化
def show_cross(args, model_acc_cross, model_list):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 5))
    num_models = len(model_acc_cross)
    x1 = np.arange(1, len(model_acc_cross[0]) + 1)
    plt.yticks(np.linspace(0, 100, num=11))
    plt.ylim([0, 100])
    color_list = ['red', 'blue', 'green', 'yellow', 'cyan']
    labels = ["1折", "2折", "3折", "4折", "5折", "平均"]
    colors = color_list[:num_models]
    bar_width = 0.8 / num_models
    width = 0
    for acc_list, color in zip(model_acc_cross, colors):
        y1 = [i * 100 for i in acc_list]
        plt.xticks(x1 + (bar_width * (num_models - 1) / 2), labels=labels)
        plt.bar(x1 + width, y1, color=color, width=bar_width)
        width += bar_width
    plt.ylabel("分类准确率（%）")
    plt.legend(labels=model_list, loc='upper left', fontsize='small')
    # plt.show()

    models = '_'.join(model_list)
    save_dir = os.path.join(os.path.dirname(__file__), f'{args.dataset_name}/cross_validation')
    save_path = os.path.join(save_dir, f'{models}.png')

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path)

