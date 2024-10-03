#%%
import argparse
import os
import torch
import numpy as np
import random
from torch_geometric.data import Data
import torch.nn.functional as F
from torch import nn, autograd

#%%
def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 'Make dirs of # {} '.format(directory)
    else:
        return "the dirs already exist! Cannot be created"
    
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/TUDataset', help='root directory for the data set. data/CFG or data/TUDataset')
    parser.add_argument('--dataset', type=str, default="ffmpeg", help='indicate the specific data set (ffmpeg/OpenSSL)')
    parser.add_argument('--graph_size_min', type=int, default=50, help='min node size for one graph ')
    parser.add_argument('--graph_size_max', type=int, default=200, help='max node size for one graph ')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--model', type=str, default='simgnn', help='model to use')
    parser.add_argument('--label_rate', type=int, default=50, help='label rate(%) for semi-supervised learning')
    parser.add_argument('--semi_method', type=str, default='pseudo', help='semi-supervised method')

    parser.add_argument('--train', action='store_true', help='whether to train the model')
    parser.add_argument('--test', action='store_true', help='whether to test the model')

    
    # parameters for simgnn
    parser.add_argument('--histogram', action='store_true', help='whether to use histogram')
    # parser.add_argument('--histogram', type=bool, default=True, help='whether to use histogram')

    # Save parameters
    parser.add_argument('--log_path', type=str, default='log', help='directory to save models')
    return parser

def set_pseudo_args(args):
    args.pseudo_step = 10 # 10%
    return args

def set_simgnn_args(args):

    args.gnn_operator = 'gcn'
    args.filters_1 = 32
    args.filters_2 = 64
    args.filters_3 = 16
    args.tensor_neurons =  16
    args.bins = 16
    args.num_NTN_mlp = 1
    args.bottle_neck_neurons = 16

    return args

class graph(object):
    def __init__(self, node_num=0, label=None, name=None, prefix_name_label=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.prefix_name_label = prefix_name_label
        self.features = []  # node feature matrix
        self.succs = []
        self.preds = []
        if node_num > 0:
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
    
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def to_torch_geometric_data(self):

        x = torch.tensor(np.array(self.features), dtype=torch.float)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        for u in range(self.node_num):
            for v in self.succs[u]:
                edge_index = torch.cat((edge_index, torch.tensor([[u], [v]], dtype=torch.long)), dim=1)
                edge_index = torch.cat((edge_index, torch.tensor([[v], [u]], dtype=torch.long)), dim=1)
        label = torch.tensor([self.label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=label)
        return data
    
def generate_multiple_pair(graphs, classes_train, classes_test, num = 3):

    y = []
    id1 = []
    id2 = []
    len_class = len(classes_train)
    num = min(max(num, len_class // 3), 12)
    assert len_class > 1    

    for cls in range(len_class):
        tot_g = len(classes_test[cls])
        g_ids = classes_test[cls].copy()

        # add positive pairs
        for i in range(tot_g):
            for _ in range(num):
                id1.append(g_ids[i])
                g_id = np.random.randint(0, len(classes_train[cls]))
                id2.append(classes_train[cls][g_id])
                y.append(1)

        # add negative pairs
        for i in range(tot_g):
            for _ in range(num):
                cls2 = np.random.randint(len_class)
                while (len(classes_train[cls2]) == 0) or (cls2 == cls):
                    cls2 = np.random.randint(len_class)
                g_id2 = np.random.randint(0, len(classes_train[cls2]))
                id1.append(g_ids[i])
                id2.append(classes_train[cls2][g_id2])
                y.append(0)
    graph_list1 = [graphs[i] for i in id1]
    graph_list2 = [graphs[i] for i in id2]
    return id1, id2, graph_list1, graph_list2, y

def generate_test_pair(graphs, classes_train, classes_test):

    y = []
    id1 = []
    id2 = []
    len_class = len(classes_train)
    assert len_class > 1
    squeeze_train = []
    squeeze_test = []
    class_train = []
    class_test = []
    for i in range(len_class):
        squeeze_train.extend(classes_train[i])
        squeeze_test.extend(classes_test[i])
        class_train.extend([i] * len(classes_train[i]))
        class_test.extend([i] * len(classes_test[i]))

    for i, id1_ in enumerate(squeeze_test):
        for j, id2_ in enumerate(squeeze_train):
            id1.append(id1_)
            id2.append(id2_)
            if class_test[i] == class_train[j]:
                y.append(1)
            else:
                y.append(0)

    graph_list1 = [graphs[i] for i in id1]
    graph_list2 = [graphs[i] for i in id2]
    return id1, id2, graph_list1, graph_list2, y, squeeze_test, squeeze_train, class_train

def generate_linear_pair(graphs, classes):

    y = []
    id1 = []
    id2 = []
    len_class = len(classes)
    assert len_class > 1    

    num = min(max(1, len_class // 3), 12)

    for cls in range(len_class):
        tot_g = len(classes[cls])
        g_ids = classes[cls].copy()

        # add positive pairs
        for i in range(0, tot_g):
            for _ in range(num):
                id1.append(g_ids[i])
                g_id = np.random.randint(0, tot_g)
                id2.append(g_ids[g_id])
                y.append(1)

        # add negative pairs
        for i in range(tot_g):
            for _ in range(num):
                cls2 = np.random.randint(len_class)
                while (len(classes[cls2]) == 0) or (cls2 == cls):
                    cls2 = np.random.randint(len_class)
                g_id2 = np.random.randint(0, len(classes[cls2]))
                id1.append(g_ids[i])
                id2.append(classes[cls2][g_id2])
                y.append(0)
    graph_list1 = [graphs[i] for i in id1]
    graph_list2 = [graphs[i] for i in id2]
    return id1, id2, graph_list1, graph_list2, y

def generate_quadratic_pair(graphs, classes, batch_size):

    # Only used for test
    id1 = []
    id2 = []
    y = []
    bound = batch_size * 100
    len_class = len(classes)
    for cls_i, cls in enumerate(range(len_class)):
        tot_g = len(classes[cls])
        for i in range(tot_g):
            for j in range(i, tot_g):
                id1.append(classes[cls][i])
                id2.append(classes[cls][j])
                y.append(1)
                if len(id1) >= bound:
                    graph_list1 = [graphs[i] for i in id1]
                    graph_list2 = [graphs[i] for i in id2]
                    yield id1, id2, graph_list1, graph_list2, y
                    id1 = []
                    id2 = []
                    y = []
            for cls2_i, cls2 in enumerate(range(len_class)):
                if cls2_i <= cls_i:
                    continue
                for j in range(len(classes[cls2])):
                    id1.append(classes[cls][i])
                    id2.append(classes[cls2][j])
                    y.append(0)
                    if len(id1) >= bound:
                        graph_list1 = [graphs[i] for i in id1]
                        graph_list2 = [graphs[i] for i in id2]
                        yield id1, id2, graph_list1, graph_list2, y
                        id1 = []
                        id2 = []
                        y = []
    if len(id1) > 0:
        graph_list1 = [graphs[i] for i in id1]
        graph_list2 = [graphs[i] for i in id2]
        yield id1, id2, graph_list1, graph_list2, y
    return 
class MB(autograd.Function):
    
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def mb(inputs, indexes, features, momentum=0.5):
    return MB.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_samples, args, temp=0.05, momentum=0.2):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.args = args

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # the source-like samples labels
        self.register_buffer('labels', torch.zeros(num_samples).long())
        # the psuedo-labels
        self.register_buffer('pred_labels', torch.zeros(num_samples).long())

    def forward(self, inputs, inputs_s, inputs_s1, indexes, k=10):
        # inputs: B*hidden_dim, features: L*hidden_dim
        inputs_out = mb(inputs, indexes, self.features, self.momentum)
        inputs_out /= self.temp  # B*L
        
        # generate local information
        B = inputs.size(0)
        local = (inputs.mm(inputs_s.t()) + inputs.mm(inputs_s1.t()))/2 # B*B
        _ , neibor_idx = torch.topk(inputs_out,k)  # B*k
        neibor_ftr = self.features[neibor_idx].permute(0,2,1) #B*2048*k
        
        _local=(torch.bmm(inputs.unsqueeze(1),neibor_ftr)).sum(-1) # B * 1
        local = (local + _local.expand_as(local))*(torch.eye(B).cuda())
        local /= self.temp


        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        # Achieve adaptive contrastive learning
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        
        sim = torch.zeros(labels.max()+1, B).float().cuda()  # L * B 
        sim.index_add_(0, labels, inputs_out.t().contiguous())
        # add the local information
        sim.index_add_(0, targets, local.contiguous())
        
        
        nums = torch.zeros(labels.max()+1, 1).float().cuda() 
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        
        nums_help = torch.zeros(labels.max()+1, 1).float().cuda() 
        nums_help.index_add_(0, targets, torch.ones(B,1).float().cuda())
        nums+=(nums_help>0).float()*(k+1)
        #avoid divide 0
        mask = (nums>0).float() 
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        
        loss1 = F.nll_loss(torch.log(masked_sim+1e-6), targets)

        
        # Adaptation loss (MMD with memory bank)
        target_mask = (labels>=self.args.class_num).float()
        target_idx = torch.nonzero(target_mask,as_tuple=True)
        source_idx = torch.nonzero(1-target_mask,as_tuple=True)
        target_label = self.pred_labels[target_idx].clone()
        source_label = self.pred_labels[source_idx].clone()
        
        ad_sim_target = torch.zeros(self.args.class_num,B).float().cuda()
        ad_sim_target.index_add_(0, target_label, inputs_out.t().contiguous()[target_idx])
        t_nums = torch.zeros(self.args.class_num, 1).float().cuda()
        t_nums.index_add_(0, target_label, torch.ones(self.num_samples,1)[target_idx].float().cuda())
        ad_sim_source = torch.zeros(self.args.class_num,B).float().cuda()
        ad_sim_source.index_add_(0, source_label, inputs_out.t().contiguous()[source_idx])
        s_nums = torch.zeros(self.args.class_num, 1).float().cuda()
        s_nums.index_add_(0, source_label, torch.ones(self.num_samples,1)[source_idx].float().cuda())
        t_nums_mask =  (t_nums>0).float() 
        s_nums_mask =  (s_nums>0).float() 
        ad_sim_target /= (t_nums*t_nums_mask+(1-t_nums_mask)).clone().expand_as(ad_sim_target)
        ad_sim_source /= (s_nums*s_nums_mask+(1-s_nums_mask)).clone().expand_as(ad_sim_source)
        # B*C
        ad_sim_target = ad_sim_target.t()
        ad_sim_source = ad_sim_source.t()
        
        batch_labels = self.pred_labels[indexes].clone()
        ad_help_t = torch.index_select(ad_sim_target,1,batch_labels)
        ad_help_s = torch.index_select(ad_sim_source,1,batch_labels)
        ad_t = torch.diagonal(ad_help_t).unsqueeze(1)
        ad_s = torch.diagonal(ad_help_s).unsqueeze(1)
        ad_sim = torch.cat((ad_s,ad_t),1) #B*2
        
        if self.args.ad_method == 'LMMD': 
            batch_mask = target_mask[indexes].clone()
            batch_help = torch.ones(B).cuda()*(-1)+2*batch_mask # target-like 1, source-like -1
            batch_help = batch_help.unsqueeze(1)
            batch_label = torch.cat((batch_help*-1,batch_help),1) # B*2
            loss2 = (ad_sim*batch_label).sum(1).mean()
        else:
            batch_mask = (targets<self.args.class_num).long() # if source-like adapt to the target features
            exp_ad = torch.exp(ad_sim)
            ad_sums = exp_ad.sum(1,keepdim=True) + 1e-6
            exp_sims = exp_ad/ad_sums
            loss2 = (F.nll_loss(torch.log(exp_sims+1e-6), batch_mask, reduction='none')).mean()
        return loss1,loss2
