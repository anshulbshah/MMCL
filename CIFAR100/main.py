import argparse
import os
import pandas
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import svm_losses
import utils
from model import Model
from termcolor import cprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(net, data_loader, train_optimizer, crit, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    kxz_losses,kyz_losses = 0.0, 0.0
    for iii, (pos_1, pos_2, target, index) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        features = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1)], dim=1)


        kxz_loss,kyz_loss = crit(features) 
        loss = kxz_loss + kyz_loss 
        kxz_losses += kxz_loss.item()*batch_size
        kyz_losses += kyz_loss.item()*batch_size

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    metrics = {
        'total_loss':total_loss / total_num,
        'kxz_loss':kxz_losses / total_num,
        'kyz_loss': kyz_losses / total_num,
    }

    return metrics

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    temperature = 0.5
    with torch.no_grad():
        # generate feature bank
        for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, _ in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar100', type=str, help='Choose loss function')
    parser.add_argument('--criterion_to_use', default='default', type=str, help='Beta annealing')
    parser.add_argument('--val_freq', default=25, type=int, help='Beta annealing')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--dataset_location', default='../data', type=str, help='Choose loss function')
    parser.add_argument('--num_workers', default=4, type=int, help='Choose loss function')

    # mmcl args
    parser.add_argument('--C_reg', type=float, default=100.0)
    parser.add_argument('--topK', type=int, default=128)
    parser.add_argument('--gamma', type=str, default="50")
    parser.add_argument('--kernel_type', type=str, default="rbf")
    parser.add_argument('--drop_sigma', type=str, default='75,125')
    parser.add_argument('--reg', type=float, default=0.1)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=1E-3)
    parser.add_argument('--stop_condition', type=float, default=0.01)
    parser.add_argument('--solver_type', type=str, default='nesterov')
    parser.add_argument('--use_norm', type=str, default='false')
    parser.add_argument('--run_name', type=str, default='MMCL')


    # args parse
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name
    if args.drop_sigma != 'no':
        sigma_drop_epochs = args.drop_sigma.split(',')
        sigma_drop_epochs=  [int(aa) for aa in sigma_drop_epochs]        
    else:
        sigma_drop_epochs = []

    run_name = args.run_name

    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name,args.dataset_location)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    epoch_start = 1

    if args.criterion_to_use == 'mmcl_inv':
        crit = svm_losses.MMCL_inv(sigma=args.gamma, batch_size=args.batch_size, anchor_count=2, C=args.C_reg)
    elif args.criterion_to_use == 'mmcl_pgd':
        crit = svm_losses.MMCL_pgd(sigma=args.gamma, batch_size=args.batch_size, anchor_count=2, C=args.C_reg, \
                                   solver_type=args.solver_type, use_norm=args.use_norm)
    
    # training loop
    if not os.path.exists('../results'):
        os.mkdir('../results')
    if not os.path.exists('../results/{}'.format(dataset_name)):
        os.mkdir('../results/{}'.format(dataset_name))
    if not os.path.exists('../results/{}/{}/'.format(dataset_name,run_name)):
        os.mkdir('../results/{}/{}/'.format(dataset_name,run_name))    


    for epoch in range(epoch_start, epochs + 1):

        if epoch in sigma_drop_epochs:
            cprint(f'Sigma right now {crit.sigma}','red')
            crit.sigma = str(float(crit.sigma)/10.0)
            cprint(f'Sigma after drop {crit.sigma}','green')
            
        metrics = train(model, train_loader, optimizer, crit, args)

        metrics['epoch'] = epoch
        metrics['lr'] = get_lr(optimizer)


        if epoch % args.val_freq == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            torch.save(model.state_dict(), '../results/{}/{}/model_{}.pth'.format(dataset_name,run_name,epoch))
            metrics['top1'] = test_acc_1
            metrics['top5'] = test_acc_5

