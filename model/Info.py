
import os
import sys
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops


def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0
    criterion = nn.CrossEntropyLoss()

    if use_unsup_loss:
        for data, data2 in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()
            
            out = model(data)
            sup_loss = criterion(out, data.y)
            unsup_loss = model.unsup_loss(data2)
            if separate_encoder:
                unsup_sup_loss = model.unsup_sup_loss(data2)
                loss = sup_loss + (unsup_loss + unsup_sup_loss) * lamda
            else:
                loss = sup_loss + unsup_loss * lamda

            loss.backward()

            sup_loss_all += sup_loss.item()
            unsup_loss_all += unsup_loss.item()
            if separate_encoder:
                unsup_sup_loss_all += unsup_sup_loss.item()
            loss_all += loss.item() * data.num_graphs

            optimizer.step()

        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
        else:
            print(sup_loss_all, unsup_loss_all)
        return loss_all / len(train_loader.dataset)
    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data)
            sup_loss = criterion(out, data.y)
            # sup_loss = F.mse_loss(model(data), data.y)
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        _, pred = out.max(dim=1)
        correct += pred.eq(data.y).sum().item()
        # error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return correct / len(loader.dataset)


def seed_everything(seed=1):
    pass
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# In[3]:


from model import Net
from arguments import arg_parse
args = arg_parse()

# DS_list = ['PROTEINS','NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'PTC_MR', 'REDDIT-MULTI-5K', 'COLLAB']
# DS_list = ['TWITTER-Real-Graph-Partial']
DS_list = ['IMDB-BINARY']
# target = args.target
dim = 32
epochs = 60
batch_size = 32
lamda = args.lamda
use_unsup_loss = True
args.separate_encoder = False
separate_encoder = False
rate_test = 0.2
rate_val = 0.1
rate_labelled_list = [0.1]
rate_unlabelled_list = [0.5]
times = 3
print(args)
# In[4]:

for DS in DS_list:
    
    for rate_labelled in rate_labelled_list:
        
        for rate_unlabelled in rate_unlabelled_list:
        
            with open(DS+'unlabel.log', 'a+') as f:
                f.write('******************** {}  {}************************\n'.format(rate_labelled, rate_unlabelled))
                
            accuracies = []
            
            for _ in range(1):

                path = osp.join(osp.dirname(osp.realpath('__file__')), '..', 'data', DS)
                dataset = TUDataset(path, name=DS, cleaned=False) # .shuffle()
                np.random.seed(seed=2)
                idx = list(np.random.permutation(len(dataset)))
                dataset = dataset[idx]

                dataset_num_features = max(dataset.num_features, 1)
                num_labels = dataset.num_classes

                test_num = int(len(dataset) * rate_test)
                val_num = int(len(dataset) * (rate_val))
                train_num = int(len(dataset) * 0.2)
                unlabeled_num = len(dataset) - test_num - val_num - train_num

                idx = list(range(len(dataset)-test_num, len(dataset)))
                idx += list(range(train_num+unlabeled_num, train_num+unlabeled_num+val_num))
                idx += list(range(train_num))
                idx += list(range(train_num, train_num + unlabeled_num))
                old_len = len(dataset)
                dataset = dataset[idx]
                dataset = dataset[:len(dataset)-int((0.5 - rate_unlabelled)*len(dataset))]
                unlabeled_num = len(dataset) - test_num - val_num - train_num

                test_dataset = dataset[:test_num]
                val_dataset = dataset[test_num:test_num+val_num]
                train_unlabel_dataset = dataset[test_num+val_num:]
                tmp = int(old_len * (rate_val + 0.2 - rate_labelled))
                train_num -= tmp - val_num
                val_num += tmp - val_num
                train_dataset = dataset[test_num+val_num:test_num+val_num+train_num]

                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                unsup_train_loader = DataLoader(train_unlabel_dataset, batch_size=batch_size, shuffle=True)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net(dataset_num_features, dataset, dim, use_unsup_loss, separate_encoder).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

                val_acc = test(val_loader)
                test_acc = test(test_loader)
                print('Epoch: {:03d}, Validation ACC: {:.7f}, Test ACC: {:.7f},'.format(0, val_acc, test_acc))

                best_val_acc = None
                for epoch in range(1, epochs):
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    loss = train(epoch, use_unsup_loss)
                    val_acc = test(val_loader)
                    scheduler.step(val_acc)
                    train_acc = test(train_loader)

                    if best_val_acc is None or val_acc >= best_val_acc:
    #                     print('Update')
                        test_acc = test(test_loader)
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), 'parameter.pkl')
#                         model.load_state_dict(torch.load('parameter.pkl'))


                    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation ACC: {:.7f}, '
                          'Test ACC: {:.7f},'.format(epoch, args.lr, loss, val_acc, test_acc))

#                     with open(DS+'unlabel.log', 'a+') as f: 
#                         f.write('{},{},{},{}\n'.format(epoch, train_acc, val_acc, test_acc))
                accuracies.append(test_acc)
             
            print(DS, np.mean(accuracies),  np.std(accuracies))
            with open(DS+'unlabel.log', 'a+') as f:
                f.write('{},{},{},{}\n'.format(DS, accuracies[0], np.mean(accuracies),  np.std(accuracies)))
