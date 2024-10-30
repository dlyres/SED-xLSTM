import argparse
import numpy as np
import torch
from dataloader.Dataloader import SSVEPDataset, make_dataloader
from utils.train_views import evaluate
import utils.model_maker as mk


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(args):
    test_dataset = SSVEPDataset(args.dataset_name, train=False, cross_validation=False)
    test_dataloader = make_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_size = len(test_dataset)
    print(f"test_data_size:{test_data_size}")
    if args.cross_validation:
        model_name = args.model_name + args.num_fold
        model_path = f'weights/cross_validation/{args.dataset_name}/' + model_name + '.pth'
    else:
        model_path = f'weights/no_cross_validation/{args.dataset_name}/' + args.model_name + '.pth'

    model = mk.make_model(args)
    model.load_state_dict(torch.load(f'{model_path}'))
    total_test_loss, class_acc, total_acc, auc, itr = evaluate(model=model,
                                                     data_loader=test_dataloader,
                                                     device=device,
                                                     epoch=0,
                                                     in_c=args.in_c,
                                                     dataset_name=args.dataset_name,
                                                     num_classes=args.num_classes)
    final_class_acc = np.round(class_acc, decimals=4)
    print("each class accuracy:{}, total accuracy:{:.4f}, auc:{:.4f}".format(final_class_acc, total_acc, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BETA')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--model_name', type=str, default='SSGFormer')
    parser.add_argument('--cross_validation', type=bool, default=False)
    parser.add_argument('--num_fold', type=str, default='0.2')

    # different dataset needs to transform patch_size
    parser.add_argument('--patch_size', type=tuple, default=(6, 64))
    parser.add_argument('--in_c', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--in_c_spa', type=int, default=4)
    parser.add_argument('--depth_spa', type=int, default=4)
    parser.add_argument('--kernel_size_spa', type=tuple, default=(7, 7))
    args = parser.parse_args()
    main(args)
