import argparse
import os
import time
import torch
import drawing.show_acc as sc
import drawing.show_acc2 as sc2
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from utils.model_maker import *
from utils.train_views import *
from dataloader.Dataloader import SSVEPDataset, make_dataloader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建自定义数据集实例和dataloader
    train_dataset = SSVEPDataset(args.dataset_name, train=True, cross_validation=args.cross_validation)
    test_dataset = SSVEPDataset(args.dataset_name, train=False, cross_validation=args.cross_validation)
    train_dataloader = make_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = make_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True)

    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print(f"训练集大小为{train_data_size}")
    print(f"测试集大小为{test_data_size}")

    # 创建模型
    model = make_model(args)
    scheduler = None

    if (args.optim == 'SGD'):
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if (args.isReduceLR == True):
        scheduler = lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode=args.mode,
                                                   patience=args.patience,
                                                   factor=args.factor,
                                                   verbose=True)
    writer = SummaryWriter(f"tensorboard/{args.dataset_name}/no_cross_validation/{args.model_name}")
    train_step = 0
    test_step = 0
    final_class_acc = np.zeros(args.num_classes)  # 最终各分准确率
    total_acc_list = []  # 总体预测准确率趋势
    start_time = time.time()
    model_auc = 0

    # 使用dataloader迭代训练集数据
    for epoch in range(args.epochs):
        total_train_loss = train_one_epoch(model=model,
                                           optimizer=optim,
                                           data_loader=train_dataloader,
                                           device=device,
                                           epoch=epoch,
                                           in_c=args.in_c,
                                           dataset_name=args.dataset_name)

        total_test_loss, class_acc, total_acc, auc, itr = evaluate(model=model,
                                                         data_loader=test_dataloader,
                                                         device=device,
                                                         epoch=epoch,
                                                         in_c=args.in_c,
                                                         dataset_name=args.dataset_name,
                                                         num_classes=args.num_classes)
        model_auc = auc
        total_acc_list.append(total_acc)
        if (args.isReduceLR == True):
            scheduler.step(total_test_loss)

        loss_scalar = f"{args.dataset_name}: model={args.model_name}, " \
                      f"in_c={args.in_c}, optim={args.optim}, lr={args.lr}, bs={args.batch_size}, " \
                      f"epoch={args.epochs}, isReduceLR={args.isReduceLR}"

        if (args.model_name == 'grvt_model' or args.model_name == 'lsvt_model' or args.model_name == 'spaFormer_model'):
            loss_scalar = loss_scalar + f", patch_size={args.patch_size}, embed_dim={args.embed_dim}, " \
                                        f"depth={args.depth}, num_heads={args.num_heads}"

        writer.add_scalar(f"{loss_scalar}/train_loss", total_train_loss, train_step)
        writer.add_scalar(f"{loss_scalar}/test_loss", total_test_loss, test_step)
        train_step = train_step + 1
        test_step = test_step + 1
        final_class_acc = class_acc

    total_acc_list = np.round(np.array(total_acc_list), decimals=3)
    final_class_acc = np.round(final_class_acc, decimals=3)

    end_time = time.time() - start_time
    print("{}模型训练所用时间为：{:.2f}分钟".format(args.model_name, end_time / 60))
    print("auc为{:.4f}".format(model_auc))

    save_dir = f'weights/no_cross_validation/{args.dataset_name}/{args.model_name}.pth'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    torch.save(model.state_dict(), save_dir)
    writer.close()

    if args.num_models != 1:
        return total_acc_list, final_class_acc
    if args.accuracy:
        sc.show(total_acc_list, final_class_acc, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练基本参数
    parser.add_argument('--dataset_name', type=str, default='BETA')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cross_validation', type=bool, default=False)

    # 学习率衰减参数
    parser.add_argument('--isReduceLR', type=bool, default=True)  # 是否采用学习率衰减
    parser.add_argument('--patience', type=int, default=5)  # 损失没有下降的轮数
    parser.add_argument('--factor', type=float, default=0.5)  # 衰减比率
    parser.add_argument('--mode', type=str, default='min')
    parser.add_argument('--verbose', type=bool, default=True)

    parser.add_argument('--model_name', type=str, default='SSGFormer')

    # 模型对比（多个模型）
    parser.add_argument('--num_models', type=int, default=3)

    # 模型参数
    parser.add_argument('--in_c', type=int, default=2)
    parser.add_argument('--patch_size', type=tuple, default=(6, 64))  # 不同数据集需要切换patch_size
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--in_c_spa', type=int, default=4)
    parser.add_argument('--depth_spa', type=int, default=4)
    parser.add_argument('--kernel_size_spa', type=tuple, default=(7, 7))

    # 准确率训练变化
    parser.add_argument('--accuracy', type=bool, default=True)

    args = parser.parse_args()

    if args.num_models == 1:
        main(args)
    else:
        model_list = ['VIT+SAM+Bi-GRU', 'VIT+Bi-GRU', 'VIT+Bi-LSTM', 'VIT+CBAM', 'VIT']
        models_acc_list = []
        models_class_list = []
        for model_name in model_list:
            args.model_name = model_name
            total_acc_list, final_class_acc = main(args)
            print(f"{args.model_name}模型训练完毕")
            models_acc_list.append(total_acc_list)
            models_class_list.append(final_class_acc)
        if args.accuracy:
            sc2.show(models_acc_list, models_class_list, args, model_list=model_list)
