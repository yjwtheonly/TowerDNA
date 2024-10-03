#%%
import utils
import torch
import dataloader
import os
import numpy as np
import pickle as pkl
from model.SimGNN import SimGNN
from torch_geometric.loader import DataLoader
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train_one_epoch(model, optimizer, graphs, seen_classes, batch_size, device):
    
    id1_train, id2_train, graphs1_ori, graphs2_ori, y_ori = utils.generate_linear_pair(graphs, seen_classes)

    model.train()
    perm = np.random.permutation(len(graphs1_ori))  # Random shuffle
    graphs1 = [graphs1_ori[idx] for idx in perm]
    graphs2 = [graphs2_ori[idx] for idx in perm]
    y = [y_ori[idx] for idx in perm]

    data1 = DataLoader(graphs1, batch_size=batch_size, shuffle=False)
    data2 = DataLoader(graphs2, batch_size=batch_size, shuffle=False)

    cum_loss = 0.0
    num = 0
    st = 0
    for index, (graph1, graph2) in enumerate(tqdm(list(zip(data1, data2)))):
        
        batch_output = model(graph1.to(device), graph2.to(device))
        yy = torch.tensor(y[st:st+len(graph1)], dtype=torch.float32, device=device)
        st += len(graph1)
        mse_loss = torch.nn.functional.mse_loss(batch_output, yy)
        
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        
        cum_loss += mse_loss
        num = num + 1
    cum_loss /= num
    return cum_loss

def eval_one_epoch(model, graphs1_ori, graphs2_ori, y_ori, batch_size, device, eval = False):
     
    model.eval()   
    with torch.no_grad():
        tot_pred = []
        tot_truth = []
        data1 = DataLoader(graphs1_ori, batch_size=batch_size, shuffle=False)
        data2 = DataLoader(graphs2_ori, batch_size=batch_size, shuffle=False)
        st = 0
        mse = 0.0
        for index, (graph1, graph2) in enumerate(tqdm(list(zip(data1, data2)))):
            batch_output = model(graph1.to(device), graph2.to(device))
            y = torch.tensor(y_ori[st:st+len(graph1)], dtype=torch.float32, device=device)
            if ((graph1.to(device).y == graph2.to(device).y) == (y == 1)).sum() != len(graph1):
                print(graph1.y[:10])
                print(graph2.y[:10])
                print(y[:10])
                raise Exception('Error!')
            st += len(graph1)
            pred = batch_output.cpu().numpy()
            # if not eval:
            # pred[pred >= 0.5] = 1
            # pred[pred < 0.5] = 0

            tot_pred += list(pred)
            tot_truth += list((y.cpu().numpy()))
            mse += torch.nn.functional.mse_loss(batch_output, y)
        mse /= len(data1)
    fpr, tpr, thresholds = roc_curve(tot_truth, tot_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    return mse, auc_score, tot_pred

def eval(args, BestModel_FILE, Result_FILE, dataset, device):

    if args.model == 'simgnn':
        args = utils.set_simgnn_args(args)
    else:
        raise ValueError('No such model!')
    classes_train = dataset.train_classes
    classes_dev = dataset.dev_classes
    classes_test = dataset.test_classes
    graphs = dataset.graphs
    # graph.label
    # for i in range(len(classes_train)):
    #     classes_train[i] = classes_train[i] + classes_dev[i]
    input_dim = graphs[0].x.shape[1]
    model = SimGNN(args, input_dim).to(device)
    model.load_state_dict(torch.load(f'{BestModel_FILE}.pth'))
    
    print('Processing dataset ...')
    id1_, id2_, graph_list1_, graph_list2_, y_, squeeze_test, squeeze_database,row_class = utils.generate_test_pair(graphs, classes_train, classes_test)

    print('Database graph number: ', len(squeeze_database), 'Test graph number: ', len(squeeze_test))
    print('Test pair number: ', len(graph_list1_))
    std_dense = np.array(y_).reshape(len(squeeze_test), len(squeeze_database))
    # return [], std_dense, squeeze_test, squeeze_database, row_class, graph_list1_, graph_list2_

    mse, auc_score, pred = eval_one_epoch(model, graph_list1_, graph_list2_, y_, args.batch_size, device)
    pred_dense = np.array(pred).reshape(len(squeeze_test), len(squeeze_database))

    print('calculating p @ 1 and p @ 10 ...')
    # return pred_dense, std_dense, squeeze_test, squeeze_database, row_class, graph_list1_, graph_list2_

    index = np.argsort(pred_dense, axis=1)
    index = index[:, ::-1]

    select_nm = 0
    maxx = 0
    for i in range(len(pred_dense)):
        ap = 0
        add = 0
        for j in range(len(std_dense[0])):
            if std_dense[i][index[i, j]] == 1:
                add += 1
                ap += add / (j + 1)
        ap /= add
        if ap > maxx:
            maxx = ap
            select_nm = i
    ith = 0
    iid = index[select_nm][ith]
    print(f'The {ith}-th similar to {select_nm} is {iid}')
    print(f'query {select_nm}\'s node num is {graph_list1_[select_nm * pred_dense.shape[1]].x.shape[0]}, category is {graph_list1_[select_nm * pred_dense.shape[1]].y.item()}')
    print(f'database {iid}\'s node num is {graph_list2_[iid].x.shape[0]}, category is {graph_list2_[iid].y.item()}')
    print(f'Similarity score is pred:{pred_dense[select_nm][iid]}, std:{std_dense[select_nm][iid]}')
    print(graph_list1_[select_nm * pred_dense.shape[1]].edge_index.shape)

    ap = 0
    add = 0
    for j in range(len(std_dense[0])):
        if std_dense[select_nm][index[select_nm, j]] == 1:
            add += 1
            ap += add / (j + 1)
    print(f'mAP of selected_num:', ap / add)
    
    def cal_p(std, pred_dense):
        p1 = 0
        p10 = 0
        p20 = 0
        index = np.argsort(pred_dense, axis=1)
        index = index[:, ::-1]
        mAP = 0
        mRR = 0
        assert len(std) == len(pred_dense) 

        for i in range(len(pred_dense)):

            assert np.max(index[i]) == len(std[0]) - 1
            if std[i][index[i][1]] == 1:
                p1 += 1
            for j in range(10):
                if std[i][index[i][j]] == 1:
                    p10 += 1
            for j in range(20):
                if std[i][index[i][j]] == 1:
                    p20 += 1
            for j in range(len(std[0])):
                if std[i][index[i, j]] == 1:
                    mRR += 1 / (j + 1)
                    break
            add = 0
            ap = 0
            for j in range(len(std[0])):
                if std[i][index[i, j]] == 1:
                    add += 1
                    ap += add / (j + 1)
            mAP += ap / add

        mRR /= len(pred_dense)
        mAP /= len(pred_dense)
        p1 /= len(pred_dense)
        p10 /= len(pred_dense) * 10
        p20 /= len(pred_dense) * 20
        return p1, p10, p20, mAP, mRR
    p1, p10, p20, mAP, mRR = cal_p(std_dense, pred_dense)
    p1_std, p10_std, p20_std, mAP_std, mRR_std = cal_p(std_dense, std_dense)

    print('AUC: ', auc_score, 'P@1: ', p1, 'P@10: ', p10, 'P@20: ', p20, 'P@1_std: ', p1_std, 'P@10_std: ', p10_std, 'P@20_std: ', p20_std, 'mAP: ', mAP, 'mRR: ', mRR)
    with open(f'{Result_FILE}.txt', 'w') as f:
        f.write(f'AUC: {auc_score}\n')
        f.write(f'P@1: {p1}\n')
        f.write(f'P@10: {p10}\n')
        f.write(f'P@20: {p20}\n')
        f.write(f'mAP: {mAP}\n')
        f.write(f'mRR: {mRR}\n')
        f.write(f'P@1_std: {p1_std}\n')
        f.write(f'P@10_std: {p10_std}\n')
        f.write(f'P@20_std: {p20_std}\n')
        f.write(f'mAP_std: {mAP_std}\n')
        f.write(f'mRR_std: {mRR_std}\n')
        f.write(f'Test pair: {len(graph_list1_)}\n')

def pseudo_train(args, BestModel_FILE, Record_FILE, dataset, device):

    if args.model == 'simgnn':
        args = utils.set_simgnn_args(args)
    else:
        raise ValueError('No such model!')
    if args.semi_method == 'pseudo':
         args = utils.set_pseudo_args(args)
    else:
        raise ValueError('No such semi_method!')

    Record = []

    print('Processing dataset ...')
    classes_train = dataset.train_classes
    classes_dev = dataset.dev_classes
    graphs = dataset.graphs
    input_dim = graphs[0].x.shape[1]
    print('Finish processing dataset, dim of node feature: ', input_dim)

    seen_classes = []
    unseen_classes = []
    for i in range(len(classes_train)):
        bound = int(len(classes_train[i]) * args.label_rate / 100)
        seen_classes.append(classes_train[i][:bound])
        unseen_classes.append(classes_train[i][bound:])

    if len(graphs) < 1000:
        id1_dev, id2_dev, graph_list1_dev, graph_list2_dev, y_dev, _, _, _ = utils.generate_test_pair(graphs, seen_classes, classes_dev)
    else:
        id1_dev, id2_dev, graph_list1_dev, graph_list2_dev, y_dev = utils.generate_multiple_pair(graphs, seen_classes, classes_dev)
    print('Dev pair number: ', len(graph_list1_dev))

    # index = np.random.permutation(len(graph_list1_train))
    # # id1_train = [id1_train[idx] for idx in index]
    # # id2_train = [id2_train[idx] for idx in index]
    # graph_list1_train = [graph_list1_train[idx] for idx in index]
    # graph_list2_train = [graph_list2_train[idx] for idx in index]
    # y_train = [y_train[idx] for idx in index]
    
    best_auc = 0.0
    for label_rate in range(args.label_rate, 101, args.pseudo_step):

        record = {}
        record['label_rate'] = label_rate
        record['train_loss'] = []
        record['dev_loss'] = []
        record['dev_auc'] = []

        print(f'label_rate = {label_rate} ...\n')
        model = SimGNN(args, input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):

            print('Epoch: {}'.format(epoch))
            train_loss = train_one_epoch(model, optimizer, graphs, seen_classes, args.batch_size, device)

            # train_loss, train_auc, _= eval_one_epoch(model, graph_list1_train[:bound], graph_list2_train[:bound], y_train[:bound], args.batch_size, device)
            dev_loss, dev_auc, _ = eval_one_epoch(model, graph_list1_dev, graph_list2_dev, y_dev, args.batch_size, device)

            print('\tTrain: loss = {}'.format(train_loss))
            print('\tDev: loss = {}, auc = {}'.format(dev_loss, dev_auc))
            record['dev_loss'].append(dev_loss)
            record['dev_auc'].append(dev_auc)
            record['train_loss'].append(train_loss)

            if dev_auc > best_auc:
                best_auc = dev_auc
                torch.save(model.state_dict(), f'{BestModel_FILE}.pth')
                with open(f'{BestModel_FILE}.txt', 'w') as f:
                    f.write(f'Epoch: {epoch}\n')
                    f.write(f'Label Rate: {label_rate}\n')
                    f.write(f'Best AUC: {best_auc}\n')
                print('Best Model Saved!')

        Record.append(record)
        with open(f'{Record_FILE}.pkl', 'wb') as f:
            pkl.dump(Record, f)
        
        if label_rate == 100:
            break
        # load best model and label the unlabeled data

        model.load_state_dict(torch.load(f'{BestModel_FILE}.pth'))

        id1_, id2_, graph_list1_, graph_list2_, y_, squeeze_unseen, squeeze_seen,row_class = utils.generate_test_pair(graphs, seen_classes, unseen_classes)
        print('Seen graph number: ', len(squeeze_seen), 'Unseen graph number: ', len(squeeze_unseen))
        print('Unseen pair number: ', len(graph_list1_))

        _, _, pred = eval_one_epoch(model, graph_list1_, graph_list2_, y_, args.batch_size * 10, device)
        pred = np.array(pred).reshape(len(squeeze_unseen), len(squeeze_seen))
        dis = np.array(pred)
        arg_max = np.argmax(dis, axis=1)
        assert len(arg_max) == len(squeeze_unseen)
        max_dis_raw = dis[np.arange(len(dis)), arg_max]
        index = np.argsort(max_dis_raw)
        assert len(index) == len(squeeze_unseen)
        tot = len(squeeze_unseen) + len(squeeze_seen)
        bound = int(tot * (label_rate + 10) / 100) - len(squeeze_seen)

        added_id = set()
        for idx in index[::-1][:bound]:
            arg_max_ = arg_max[idx]
            seen_classes[row_class[arg_max_]].append(squeeze_unseen[idx])
            graphs[squeeze_unseen[idx]].y = torch.tensor([row_class[arg_max_]], dtype=torch.long)
            added_id.add(squeeze_unseen[idx])
        
        for i in range(len(unseen_classes)):
            unseen_classes[i] = [item for item in unseen_classes[i] if item not in added_id]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = utils.get_args()
args = parser.parse_args()
utils.seed_all(args.seed)
main_data_dir = args.data_dir
graph_name = args.dataset
if 'CFG' in main_data_dir:
    graph_min = args.graph_size_min
    graph_max = args.graph_size_max
    graph_init_dim = 6
    title = '{}_Min{}_Max{}'.format(graph_name, graph_min, graph_max)
    print(title)

    sub_data_dir = f'{graph_name}_{graph_init_dim}ACFG_min{graph_min}_max{graph_max}'
    cfg_data_dir = os.path.join(main_data_dir, sub_data_dir) if 'ffmpeg' in sub_data_dir else os.path.join(main_data_dir, sub_data_dir, 'acfgSSL_6')
    utils.create_dir_if_not_exists(args.log_path)
    main_log_dir = args.log_path + f'/{graph_name}_Min{graph_min}_Max{graph_max}_InitDims{graph_init_dim}/'
    create_log_str = utils.create_dir_if_not_exists(main_log_dir)
    dataset = dataloader.CFGDataset(cfg_data_dir)

elif 'TUDataset' in main_data_dir:
    title = graph_name
    print(title)

    utils.create_dir_if_not_exists(args.log_path)
    main_log_dir = args.log_path + f'/{graph_name}/'
    create_log_str = utils.create_dir_if_not_exists(main_log_dir)
    dataset = dataloader.CustomizedTUDataset(main_data_dir, graph_name)

else:
    raise Exception('Wrong data dir!')

best_model_dir = main_log_dir + 'BestModels/'
create_BestModel_dir = utils.create_dir_if_not_exists(best_model_dir)
record_dir = main_log_dir + 'Record/'
create_record_dir = utils.create_dir_if_not_exists(record_dir)
result_dir = main_log_dir + 'Result/'
create_result_dir = utils.create_dir_if_not_exists(result_dir)
BestModel_FILE = best_model_dir + args.model
Record_FILE = record_dir + args.model
Result_FILE = result_dir + args.model

if args.model == 'simgnn':
    BestModel_FILE += str(args.histogram)
    Record_FILE += str(args.histogram)
    Result_FILE += str(args.histogram)

# if args.train:
#     pseudo_train(args, BestModel_FILE, Record_FILE, dataset, device)
# if args.test:
    # eval(args, BestModel_FILE, Result_FILE, dataset, device)
pred_dense, std_dense, squeeze_test, squeeze_database, row_class, graph_list1_, graph_list2_ = eval(args, BestModel_FILE, Result_FILE, dataset, device)
# %%
# index = np.argsort(pred_dense, axis=1)
# index = index[:, ::-1]
for select_nm in range(10):
    # select_nm = 0
    # select_nm = 95

    # maxx = 0
    # for i in range(len(pred_dense)):
    #     if graph_list1_[i * pred_dense.shape[1]].x.shape[0] > 15:
    #         continue
    #     ap = 0
    #     add = 0
    #     for j in range(len(std_dense[0])):
    #         if std_dense[i][index[i, j]] == 1:
    #             add += 1
    #             ap += add / (j + 1)
    #     ap /= add
    #     if ap > maxx:
    #         maxx = ap
    #         select_nm = i
    # if select_nm == -1:
    #     min_node = 1000
    #     for i in range(len(pred_dense)):
    #         if graph_list1_[i * pred_dense.shape[1]].x.shape[0] < min_node:
    #             min_node = graph_list1_[i * pred_dense.shape[1]].x.shape[0]
    #             select_nm = i


    # ith = 0
    # cal = [641, 661, 1172, 702, 1138, 653, 852, 717, 798, 1030]
    # simiscore = [0.9898102, 0.98917526, 0.98875344, 0.9882459, 0.98736095, 0.98691493, 0.98654085, 0.98645926, 0.98645335, 0.9858268]
    # cal = []
    # cal = []
    # simiscore = []

    # for i in range(10):
    #     cal.append(index[select_nm][i])
    #     simiscore.append(pred_dense[select_nm][index[select_nm][i]])

    # print(cal)
    # print(simiscore)
    # print(f'The {ith}-th similar to {select_nm} is {iid}')
    # print(f'query {select_nm}\'s node num is {graph_list1_[select_nm * pred_dense.shape[1]].x.shape[0]}, category is {graph_list1_[select_nm * pred_dense.shape[1]].y.item()}')
    # print(f'database {iid}\'s node num is {graph_list2_[iid].x.shape[0]}, category is {graph_list2_[iid].y.item()}')
    # print(f'Similarity score is pred:{pred_dense[select_nm][iid]}, std:{std_dense[select_nm][iid]}')
    # print(graph_list1_[select_nm * pred_dense.shape[1]].edge_index.shape)

    # ap = 0
    # add = 0
    # for j in range(len(std_dense[0])):
    #     if std_dense[select_nm][index[select_nm, j]] == 1:
    #         add += 1
    #         ap += add / (j + 1)
    # print(f'mAP of selected_num:', ap / add)

    import networkx as nx
    import matplotlib.pyplot as plt

    graph = graph_list1_[select_nm * std_dense.shape[1]]
    edge_index = graph.edge_index
    # print(edge_index)
    graph_nx = nx.Graph()
    graph_nx.add_edges_from(edge_index.T.numpy())
    n_node = graph_nx.number_of_nodes()
    # print(n_node)
    if(graph.x is None):
        # node_labels = np.array([1] * graph.x.shape[0])
        n_node = graph.number_of_nodes()
        node_labels = np.array([1] * n_node)
    else:
        Y = torch.argmax(graph.x, dim=1)
        node_labels = Y.numpy()
    S = list(set(node_labels))
    print(S)
    iid = range(len(S))
    M = dict(list(zip(S,iid)))
    print(M)
    node_classes = len(S)

    start_color = np.array([68,2,84], dtype = np.int32)
    mid_color = np.array([50,191,182], dtype = np.int32)
    end_color = np.array([246,249,17], dtype = np.int32)

    add_blue = 140 
    del_else = -20 
    vec1 = mid_color - start_color
    vec2 = end_color - mid_color

    category = graph.y.item()

    graph = nx.Graph()
    graph.add_edges_from(edge_index.T.numpy())

    fig = plt.figure(figsize=(10,10))
    colors = []
    ref = (node_classes-1) / 2
    for i in range(node_classes):

        if(i <= ref):
            color = np.round(start_color + 1.0 * vec1 * i / ref).astype(np.int32)
        else:
            color = np.round(mid_color + 1.0 * vec2 * (i-ref) / (node_classes-1 - ref)).astype(np.int32)
    #         add = int(round(1.0*add_blue*pow((ref - abs(i - ref)) / ref, 2)))
    #         dele = int(round(1.0*del_else*pow((ref - abs(i - ref)) / ref, 4)))
        colors.append("#%02x%02x%02x" % (color[0], color[1], color[2]))
    C = [colors[M[i]] for i in node_labels]
    C = ['Orange'] * n_node
    print(C)
    nx.draw(graph, pos=nx.spring_layout(graph), node_color = C, node_size=1005,linewidths=4, arrows = False)
    # plt.savefig(f'fig/case/{args.dataset}_query_{select_nm}_{category}.pdf')
    # plt.savefig(f'fig/case/reverse/{args.dataset}_query_{select_nm}_{category}.pdf')
    plt.savefig(f'fig/case/sample/{args.dataset}_{select_nm}.pdf')

    # for ith,(iiad, sim) in enumerate(zip(cal, simiscore)):
    #     graph = graph_list2_[iiad]
    #     edge_index = graph.edge_index
    #     if(graph.x is None):
    #         node_labels = np.array([1] * graph.x.shape[0])
    #     else:
    #         Y = torch.argmax(graph.x, dim=1)
    #         node_labels = Y.numpy()
    #     S = list(set(node_labels))
    #     iid = range(len(S))
    #     M = dict(list(zip(S,iid)))
    #     node_classes = len(S)

    #     category = graph.y.item()
    #     graph = nx.Graph()
    #     graph.add_edges_from(edge_index.T.numpy())
    #     fig = plt.figure(figsize=(10,10))
    #     colors = []
    #     ref = (node_classes-1) / 2
    #     for i in range(node_classes):

    #         if(i <= ref):
    #             color = np.round(start_color + 1.0 * vec1 * i / ref).astype(np.int32)
    #         else:
    #             color = np.round(mid_color + 1.0 * vec2 * (i-ref) / (node_classes-1 - ref)).astype(np.int32)
    #     #         add = int(round(1.0*add_blue*pow((ref - abs(i - ref)) / ref, 2)))
    #     #         dele = int(round(1.0*del_else*pow((ref - abs(i - ref)) / ref, 4)))
    #         colors.append("#%02x%02x%02x" % (color[0], color[1], color[2]))
    #     C = [colors[M[i]] for i in node_labels]
    #     nx.draw(graph, pos=nx.spring_layout(graph), node_color = C, node_size=1005,linewidths=4, arrows = False)
    #     # plt.savefig(f'fig/case/{args.dataset}_database_{ith}th_{iiad}_forquery{select_nm}_{category}_{sim}.pdf')
    #     plt.savefig(f'fig/case/reverse/{args.dataset}_database_{ith}th_{iiad}_forquery{select_nm}_{category}_{sim}.pdf')
