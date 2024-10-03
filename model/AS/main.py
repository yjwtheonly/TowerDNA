
import os
import os.path as osp
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import ot
import time
from argparse import ArgumentParser
from torch.optim import lr_scheduler
import copy

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torchnet import meter
from torch.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss, Dropout, Softplus


class AccMeter():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.correct_num = 0.0
        self.total_num = 0
        self.iters = 0

    def add(self, preds, targets):
        self.correct_num += torch.sum(preds.eq(targets))
        self.total_num += preds.shape[0]
        self.iters += 1

    def reset(self):
        self.correct_num = 0.0
        self.total_num = 0.0
        self.iters = 0

    def value(self):
        return float(self.correct_num) / (self.total_num + 1e-10)

def make_args():
    parser = ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64, help='batch size')
    parser.add_argument("--epochs",
                        type=int,
                        default=600,
                        help="number of epochs")
    parser.add_argument('--use_tb',
                        type=bool,
                        default=False,
                        help='whether use tensorboard for logs')
    parser.add_argument('--device',
                        type=int,
                        default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset',
                        type=str,
                        default='OPV',
                        help='which dataset to use')
    parser.add_argument('--save_model',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--workers',
                        default=0,
                        help='number of workers to load data')
    parser.add_argument('--shuffle',
                        default=True,
                        help='whether shuffle data before training')
    parser.add_argument('--multi_gpu',
                        default=False,
                        help='use multi gpu for training')
    parser.add_argument('--use_default',
                        default=True,
                        help='whether use augments in args')
    parser.add_argument('--lr', default=1e-3, help='learning rate')

    parser.add_argument('--train_data_num',
                        type=int,
                        default=80000,
                        help='use how many training data')

    # prediction
    parser.add_argument('--prop_name',
                        default='homo',
                        help='which property to predict')

    # universal active learning settings
    parser.add_argument('--batch_data_num', type=int, default=5000)
    parser.add_argument('--test_freq', default=1)

    #qbc settings
    parser.add_argument('--qbc_ft_epochs', default=5)
    parser.add_argument('--process_num',
                        type=int,
                        default=4,
                        help='how many cards or process you want')
    parser.add_argument('--model_num', default=4)
    parser.add_argument('--test_use_all',
                        default=False,
                        help='whether use all models when testing')

    # k-center settings
    parser.add_argument('--init_data_num',
                        default=5000,
                        help='initial data size')
    parser.add_argument('--k_center_ft_epochs',
                        default=10,
                        help='finetuning epochs for k center method')

    # bayes active learning settings
    parser.add_argument('--bald_ft_epochs',
                        default=5,
                        help='finetuning epochs for bayes active learning')
    parser.add_argument('--mc_sampling_num',
                        default=80,
                        help='monte carlo sampling number')

    # run_al settings
    parser.add_argument(
        '--al_method',
        type=str,
        default='k_center',
        help=
        'AL method in run_al.py, must be in random, bayes, k_center, msg_mask, dropout'
    )
    parser.add_argument(
        '--ft_method',
        type=str,
        default='fixed_epochs',
        help=
        'finetuning method in run_al.py, must be in fixed_epochs, varying_epochs, by_valid'
    )
    parser.add_argument(
        '--ft_epochs',
        type=int,
        default=20,
        help='the max epochs number for fixed epochs finetuning')
    parser.add_argument(
        '--re_init',
        type=bool,
        default=False,
        help=
        'whether to re-initialize the model after each iteration, advised to use by_valid ft_method if set True'
    )
    parser.add_argument(
        '--data_mix',
        type=bool,
        default=False,
        help='whether finetuning only use part of original data')
    parser.add_argument('--data_mixing_rate',
                        type=float,
                        default=1,
                        help='how much data to use in the original dataset')

    parser.add_argument(
        '--test_checkpoint',
        type=str,
        default=True,
        help=
        'whether re-train a big model to test the mae at the checkpoint dataset like [10000,20000,30000,40000]'
    )

    parser.add_argument('--mask_n_ratio',
                        type=float,
                        default=0.4,
                        help='the ratio of the nodes to be masked')

    args = parser.parse_args([])

    return args


class ShiftSoftplus(Softplus):
    """
    Shiftsoft plus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, input):
        return self.softplus(input) - np.log(float(self.shift))
class GIN(nn.Module):
    def __init__(self, num_features, dim, num_classes, num_gc_layers, edge_classes, node_classes = 20, norm = False):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.norm = norm
        
        self.cls_classifier = Sequential(Dropout(p=0.2), nn.Linear(dim, num_classes))
        self.node_classifier = nn.Linear(dim, node_classes)
        self.edge_classifier = nn.Linear(dim, edge_classes)
#         self.prop_regressor = nn.Linear(dim,1) #???
        
        self.dense_layer1 = nn.Linear(dim, dim)
        self.dense_layer2 = nn.Linear(dim,dim)
        
        self.mask_n_ratio = 0.1
        self.mask_msg_ratio = 0.1
        
        self.activation = ShiftSoftplus()
        
        for i in range(num_gc_layers):
            if i:
                tmpn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                tmpn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(tmpn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)
        self.init_emb()
        
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def set_mean_std(self, mean, std):
        self.mean_per_atom = mean.clone().detach()
        self.std_per_atom = std.clone().detach()                
    
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        old_x = x
            
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.shape[1], 1)).to(device)
            
        mask = torch.randint(0,edge_index.shape[1],[int(self.mask_n_ratio*batch.shape[0])])
        src_ids, dst_ids = edge_index[0], edge_index[1]
        src_ids, dst_ids = torch.Tensor([src_ids[i] for i in mask]).long().squeeze(), torch.Tensor([dst_ids[i] for i in mask]).long().squeeze()
        
        x[src_ids] = 0
        
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

        node = self.dense_layer1(x)
        act_node = self.activation(node)
        res = self.dense_layer2(act_node)
        
#         prop = self.prop_regressor(res).squeeze()
        
#         if self.norm:
#             prop = prop * self.std_per_atom.to(node.device) + self.mean_per_atom.to(node.device)
            
        embeddings_g = global_mean_pool(res, batch)
        
        #get prediction
        node_preds = self.node_classifier(res[src_ids])
        edge_preds = self.edge_classifier(torch.abs(res[src_ids] - res[dst_ids]))
        cls_preds = self.cls_classifier(embeddings_g)
        
#         prop_preds = global_add_pool(prop, batch)
        
        return res, node_preds, edge_preds, (src_ids, dst_ids, mask), cls_preds, embeddings_g, edge_attr, old_x

class Weakly_Supervised_Trainer(object):
    def __init__(self, args, wal_settings):
        self.args = args
        self.wal_settings = wal_settings
        self.cls_tags = 0
        self.iters = 0
        self.method = wal_settings['cls_method']

    def run(self, model, dataset, optimizer, device, writer=None, p_labels=None, level='g'):
        if self.method == 'ot':
            self._run_ot(model, dataset, optimizer, device, writer, p_labels, level)
        else:
            raise ValueError
            
    def _run_ot(self,
                model,
                dataset,
                optimizer,
                device,
                writer=None,
                p_labels=None,
                level='g'):
        settings = self.wal_settings
        
        print("trainging teacher")
        model.to(device)
        if p_labels is not None:
            p_labels = p_labels.to(device)
        loss_fn = nn.CrossEntropyLoss()
        MSE_fn = nn.MSELoss()
        MAE_fn = nn.L1Loss()
        loss_meter = meter.AverageValueMeter()
        n_loss_meter = meter.AverageValueMeter()
        e_loss_meter = meter.AverageValueMeter()
        c_loss_meter = meter.AverageValueMeter()
        p_loss_meter = meter.AverageValueMeter()
        n_acc_meter = meter.ConfusionMeter(settings['cls_num'])# clustering num might be too big, do not use confusion matrix
        e_acc_meter = meter.ConfusionMeter(settings['edge_classes'])
        p_mae_meter = meter.AverageValueMeter()
        c_acc_meter = AccMeter(settings['cls_num'])
        init_lr = self.args.lr
        info = {
            'n_loss': [],
            'n_acc': [],
            'c_loss': [],
            'c_acc': [],
            'p_loss': [],
            'p_mae': []
        }
        cls_tags = 0
        edge_bins = torch.linspace(0, 30, 150).to(device)  # 0.2 per bin
        K = settings['num_labels']
        N = len(dataset)

        # q = np.ones(K)/K     # cls distribution
        # p = np.ones(N)/N     # instance distribution

        # C = np.ones([N, K]) * np.log(K) / N  # prob_tensor  (cost function)
        # Q = np.ones([N, K]) / (K * N)  # the tag is a prob distribution

        # # Now I replace it by a normal distribution 4 is decided by 100000*Gauss(4)~10
        q = np.exp(-(np.linspace(-4, 4, K)**2) / 2) / (np.sqrt(2 * np.pi))
        q = q / q.sum()
        p = torch.ones(N) / N
        #
        C = np.ones([N, K]) * np.log(K) / N  # cost matrix
        Q = np.copy(np.tile(q, (N, 1))) / N  # joint distribution

        model.set_mean_std(torch.zeros([1]), torch.ones([1]))
        old_dataset = dataset
        for epoch in range(self.args.ft_epochs):
            
#             Idx = list(np.random.permutation(len(dataset)))
            Idx = list(range(len(dataset)))
            dataset = old_dataset[Idx]

            train_loader = DataLoader(dataset,
                                  batch_size=self.args.batchsize,
                                  shuffle=False)

            loss_meter.reset()
            n_loss_meter.reset()
            e_loss_meter.reset()
            c_loss_meter.reset()
            p_loss_meter.reset()
            n_acc_meter.reset()
            e_acc_meter.reset()
            c_acc_meter.reset()
            p_mae_meter.reset()
            model.train()

            # prepare pesudo labels via optimal transport
            if epoch % settings['cls_epochs'] == 1:
                time0 = time.time()
                Q = ot.sinkhorn(p, q, C, 0.04)
                print('optimal transport finished {}'.format(time.time() - time0))
            
            psum = 0
            for idx, data in enumerate(train_loader):
                
                length = len(data.y)
                ids = Idx[psum:psum+length]
                
                data = data.to(device)
                
                # make pesudo labels vis optimal transport
                cls_labels = torch.tensor(np.argmax(Q[list(ids)],axis=1), requires_grad=False).to(device).long()

                atom, atom_preds, edge_preds, (src, dst, edge_ids), cls_preds, embeddings_g, edge_attr, x = model(data)
                
                edge_labels = torch.clone(edge_attr[edge_ids]).requires_grad_(False)
                edge_labels = torch.argmax(edge_labels, 1)
                
                node_labels = torch.clone(x[src]).requires_grad_(False)
                node_labels = torch.argmax(node_labels, 1)
            

                n_pred_cls = torch.argmax(atom_preds, dim=1)
                e_pred_cls = torch.argmax(edge_preds, dim=1)
                c_pred_cls = torch.argmax(cls_preds, dim=1)
                cls_logits = torch.log(F.softmax(cls_preds, dim=1))

                n_loss = loss_fn(atom_preds, node_labels)
                e_loss = loss_fn(edge_preds, edge_labels)
                c_loss = loss_fn(cls_preds, cls_labels)

                p_loss = torch.Tensor([0.]).to(device)
                loss = c_loss + n_loss + e_loss
                if level == 'w':
                    p_loss = loss_fn(cls_preds, p_labels[ids])
                    loss = loss * 1e-3 + p_loss
#                     p_mae = MAE_fn(prop_preds, p_labels[ids])


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                C[ids] = -cls_logits.detach().cpu().numpy() 
                psum += length

                loss_meter.add(loss.detach().item())
                n_loss_meter.add(n_loss.detach().item())
                e_loss_meter.add(e_loss.detach().item())
                c_loss_meter.add(c_loss.detach().item())
                n_acc_meter.add(n_pred_cls, node_labels)
                e_acc_meter.add(e_pred_cls, edge_labels)
                c_acc_meter.add(c_pred_cls, cls_labels)
                p_loss_meter.add(p_loss.detach().item())
#                 p_mae_meter.add(p_mae.detach().item()) if p_labels is not None else p_mae_meter.add(0)

            # n_loss_test, n_acc_test= test(args,test_loader,model,device)
            n_acc = 100 * sum(n_acc_meter.value()[i, i]
                              for i in range(settings['cls_num'])) / n_acc_meter.value().sum()
            e_acc = 100 * sum(e_acc_meter.value()[i, i]
                              for i in range(settings['edge_classes'])) / e_acc_meter.value().sum()
            print(
                "Epoch {:2d}, training: loss: {:.7f}, node {:.4f} acc: {:.4f} edge {:.4f} acc {:.4f} clustering: loss: {:.4f} acc {:.4f} props: loss {:.5f}  mae {:.5f} level {}"
                .format(epoch,
                        loss_meter.value()[0],
                        n_loss_meter.value()[0], n_acc,
                        e_loss_meter.value()[0], e_acc,
                        c_loss_meter.value()[0], c_acc_meter.value(),
                        p_loss_meter.value()[0],
                        p_mae_meter.value()[0], level))

            info['n_loss'].append(n_loss_meter.value()[0])
            info['n_acc'].append(n_acc)
            info['c_loss'].append(c_loss_meter.value()[0])
            info['p_loss'].append(p_loss_meter.value()[0])
            info['p_mae'].append(p_mae_meter.value()[0])
            self.iters += 1
        return info


# In[25]:


def test(loader, model):
    model.eval()
    correct = 0
    A= []
    for data in loader:
        data = data.to(device)
        atom, atom_preds, edge_preds, (src, dst, edge_ids), out, embeddings_g, edge_attr, x = model(data)
        _, pred = out.max(dim=1)
        A.append(pred)
        correct += pred.eq(data.y).sum().item()
        # error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    ret = torch.cat(A, 0)
    return ret, correct / len(loader.dataset)


# In[33]:


def check_point_test(al_settings, train_dataset, val_dataset, test_dataset, teacher_model, device, best_val_acc, best_test_acc):

    model = GIN(al_settings['num_features'], 
                            al_settings['dim'],
                            al_settings['num_labels'], 
                            al_settings['n_conv'], 
                            edge_classes = al_settings['edge_classes'], 
                            node_classes=al_settings['cls_num'])

    model.load_state_dict(copy.deepcopy(teacher_model.state_dict()),strict=False) #
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=al_settings['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=al_settings['batch_size'],
                              shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset,
                             batch_size=al_settings['batch_size'],
                             shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=al_settings['batch_size'],
                              shuffle=False)
#     print('Start checkpoint testing  label num {}'.format(len(train_dataset)))
#     print('dataset mean {} std {}'.format(train_dataset.mean.item(),
#                                           train_dataset.std.item()))
   
    test_acc = 0
    loss_fn = CrossEntropyLoss()
    print("training student")
    for i in range(al_settings['student_epochs']):
        
        model.train()
        total_loss = 0.0
        
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            
            atom, atom_preds, edge_preds, (src, dst, edge_ids), cls_preds, embeddings_g, edge_attr, x = model(data)
            loss = loss_fn(cls_preds, data.y)
            total_loss += loss.item() * data.num_graphs
            
            loss.backward()
            optimizer.step()
            
        _, val_acc = test(val_loader, model)
        _, test_acc = test(test_loader, model)
        if (best_val_acc is None or val_acc > best_val_acc):
            best_val_acc = val_acc
#             if (test_acc>best_test_acc):
            best_test_acc = test_acc
            
        print('Epoch: {:03d}, Loss: {:.7f}, Validation ACC: {:.7f}, '
                          'Test ACC: {:.7f}, best val ACC: {:.7f}, best test ACC: {:.7f}'.format(i, total_loss / len(train_dataset), val_acc, test_acc, best_val_acc, best_test_acc))

    return model, best_val_acc, best_test_acc


# In[38]:


args = make_args()
args.labeled_batch_size = 32
args.unlabeled_batch_size = 32
args.ft_epochs = 12
al_settings = {
        'dim': 32,
        'n_conv': 3,
        'cutoff': 30.0,
        'width': 0.1,
        'norm': False,
        'output_dim': 1,
        'atom_ref': None,
        'pre_train': None,
        'lr': 1e-3,
        'epochs': 150,
        'batch_size': 32,
        'n_patience': 55,
        'cls_method': 'ot',
        'prop_bins': 25,
        'cls_num': 20,
        'cls_epochs': 3,
        'iters': 6,
        'student_epochs': 15,
        'init_method': 'k_medroids',
    }
print(args)
print()

seed = 2
# DS_list = ['MSRC_21', 'Mutagenicity', 'PROTEINS', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'PTC_MR', 'REDDIT-MULTI-5K']
DS_list = ['PROTEINS', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']
# DS_list = ['PROTEINS']
rate_test = 0.2
rate_val = 0.1
rate_labelled_list = [0.002, 0.006]
rate_unlabelled_list = [0.5]
times = 3


# In[37]:


for DS in DS_list:
    
    print("***************",DS,"************************")
    for rate_labelled in rate_labelled_list:
        
        for rate_unlabelled in rate_unlabelled_list:
        
            with open(DS+'unlabel.log', 'a+') as f:
                f.write('******************** {}  {}************************\n'.format(rate_labelled, rate_unlabelled))
            
            accuracies = []

            for _ in range(times):

                path = osp.join(osp.dirname(osp.realpath('__file__')), '..', 'data', DS)
                dataset = TUDataset(path, name=DS, cleaned=False) # .shuffle()
                
                np.random.seed(seed=21)
                
                idx = list(np.random.permutation(len(dataset)))
                dataset = dataset[idx]

                num_features = max(dataset.num_features, 1)
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
                tmp = int(old_len * (rate_val + 0.2 - rate_labelled))
                train_num -= tmp - val_num
                val_num += tmp - val_num
                train_dataset = dataset[test_num+val_num:test_num+val_num+train_num]
                train_unlabel_dataset = dataset[test_num+val_num:]
                print("unlabel label", len(train_unlabel_dataset), len(train_dataset))

#                 labeled_trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.labeled_batch_size)
#                 unlabeled_trainloader = None
#                 if len(train_unlabel_dataset)>0:
#                     unlabeled_trainloader = DataLoader(train_unlabel_dataset, shuffle=True, batch_size=args.unlabeled_batch_size)
#                 validloader = DataLoader(val_dataset, shuffle=False, batch_size=args.labeled_batch_size)
#                 testloader = DataLoader(test_dataset, shuffle=False, batch_size=args.labeled_batch_size)

                device = torch.device('cuda')
                print(device)
                
                edge_classes = 1
                if(dataset.data.edge_attr is not None):
                    edge_classes = dataset.data.edge_attr.shape[1]
                al_settings['num_features'] = num_features
                al_settings['num_labels'] = num_labels
                al_settings['edge_classes'] = edge_classes
                al_settings['cls_num'] = 1
                if(dataset.data.x is not None):
                    al_settings['cls_num'] = dataset.data.x.shape[1]
                print(al_settings)
                
                model_teacher = GIN(al_settings['num_features'], 
                            al_settings['dim'],
                            al_settings['num_labels'], 
                            al_settings['n_conv'], 
                            edge_classes = al_settings['edge_classes'], 
                            node_classes=al_settings['cls_num'])
                
                optimizer = torch.optim.Adam(model_teacher.parameters(), lr=args.lr * 0.1)
                p_labels = torch.zeros(len(train_unlabel_dataset)).long()
                for i in range(train_num):
                    p_labels[i] = train_unlabel_dataset[i].y[0]
                al_trainer = Weakly_Supervised_Trainer(args, al_settings)
                
                al_trainer.run(model_teacher,
                   train_unlabel_dataset,
                   optimizer,
                   device,
                   None,
                   None,
                   level='g')
                
                Iters = 6
                best_val_acc = None
                best_test_acc = 0
                for ite in range(Iters):
                    
                    print("Iters: %d" % (ite))
                    model_student, best_val_acc, best_test_acc = check_point_test(al_settings, 
                                                                                 train_dataset,
                                                                                 val_dataset, test_dataset,
                                                                                 model_teacher, device, best_val_acc, best_test_acc)
                    unlabel_loader = DataLoader(dataset=train_unlabel_dataset,
                              batch_size=al_settings['batch_size'],
                              shuffle=False)
                    p_labels,_ = test(unlabel_loader, model_student)
                    for i in range(train_num):
                        p_labels[i] = train_unlabel_dataset[i].y[0]
                        
                    train_info = al_trainer.run(model_teacher,
                                    train_unlabel_dataset,
                                    optimizer,
                                    device,
                                    None,
                                    p_labels,
                                    level='w')
                accuracies.append(best_test_acc)
                
            print(DS, np.mean(accuracies),  np.std(accuracies))
            with open(DS+'unlabel.log', 'a+') as f:
                f.write('{},{},{},{}\n'.format(DS, accuracies[0], np.mean(accuracies),  np.std(accuracies)))
                


# In[39]:


# # for DS in DS_list:
# # #      'REDDIT-BINARY', 'PTC_MR', 'COLLAB', 'REDDIT-MULTI-5K'
# DS = 'PTC_MR'
# path = osp.join(osp.dirname(osp.realpath('__file__')), '..', 'data', DS)
# dataset = TUDataset(path, name=DS, cleaned=False) # .shuffle()
# np.random.seed(seed=2)
# idx = list(np.random.permutation(len(dataset)))
# dataset = dataset[idx]
# #     print(dataset.edge_attr)
# loader = DataLoader(dataset, batch_size=4, shuffle=True)
# for idx,data in enumerate(loader):
#     if(idx == 0):
# #         print(type(data)) #torch_geometric.data.batch.Batch, torch_geometric.data.Data is its base class
# #         print(data.edge_index.max()) # 2 * edge
#         print(data.batch.shape[0])
#         print(data.batch)
#         print(len(data.y))
# #         print(data.edge_attr) # edge * feature

# for idx,data in enumerate(loader):
#     if(idx == 0):
# #         print(type(data)) #torch_geometric.data.batch.Batch, torch_geometric.data.Data is its base class
# #         print(data.edge_index.max()) # 2 * edge
#         print(data.batch.shape[0])
#         print(data.batch)
#         print(len(data.y))
# #         print(data.edge_attr) # edge * feature
