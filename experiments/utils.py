import joblib
import os
import torch
import numpy as np
from copy import deepcopy
import random
import json, logging, sys
import math
import logging.config 
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected

def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "..")


def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")


def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
def save_model(model, save_path, emb=None):

    if emb == None:
        state = {
            'state_dict_model'	: model.state_dict(),
            # 'state_dict_predictor'	: linkPredictor.state_dict(),
        }

    else:
        state = {
            'state_dict_model'	: model.state_dict(),
            'emb'	: emb.weight
        }

    torch.save(state, save_path)

def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)
    
def ratio_split_egdes(total_edges, split_ratio_str):
    split_ratio = [float(x) / 10 for x in split_ratio_str.split(":")]
    total_length = len(total_edges)
    train_size = int(split_ratio[0] * total_length)
    test_size = int(split_ratio[2] * total_length)
    valid_size = total_length - train_size - test_size
    random.shuffle(total_edges)
    train_pos = total_edges[:train_size]
    valid_pos = total_edges[train_size:train_size + valid_size]
    test_pos = total_edges[train_size + valid_size:]
    return train_pos, valid_pos, test_pos

def randomsplit(dataset, data_name, split_ratio):
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    total_pos, total_neg = [], []
    node_set = set()
    dir_path = get_root_dir()
    
    for split in ['train', 'test', 'valid']:

        path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

       
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
            total_pos.append((sub, obj))

    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)
    train_pos, valid_pos, test_pos = ratio_split_egdes(total_pos, split_ratio)
    
    for split in ['train', 'test', 'valid']:
        path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)            
            if split == 'train': 
                train_neg.append((sub, obj))
            if split == 'valid': 
                valid_neg.append((sub, obj))
            if split == 'test': 
                test_neg.append((sub, obj))
            total_neg.append((sub, obj))
    
    train_neg, valid_neg, test_neg = ratio_split_egdes(total_neg, split_ratio)
    
    train_pos_tensor = torch.tensor(train_pos)
    train_neg_tensor = torch.tensor(train_neg)
    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)
    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)
    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]
    split_edge['train']['edge'] = train_pos_tensor
    split_edge['train']['edge_neg'] = train_neg_tensor
    # data['train_val'] = train_val
    split_edge['valid']['edge']= valid_pos
    split_edge['valid']['edge_neg'] = valid_neg
    split_edge['test']['edge']  = test_pos
    split_edge['test']['edge_neg']  = test_neg
    return split_edge

def loaddataset(name, split_ratio, use_val_edges_as_input, load=None):
    dataset = Planetoid(root="dataset", name=name)
    name = name.lower()
    split_edge = randomsplit(dataset, name, split_ratio)
    data = dataset[0]
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
 
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())

    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    
    dir_path = get_root_dir()
    feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data.x = feature_embeddings
    data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])
    # Use training + validation edges for inference on test set.
    if use_val_edges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_edge_index = full_edge_index
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_edge_index = edge_index
        data.full_adj_t = data.adj_t
    return data, split_edge

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, stable=False):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            if stable:
                result_argmax = best_result[:, 1].argmax().item()
                result_argmin = best_result[:, 1].argmin().item()
                indices = torch.arange(best_result.size(0))
                indices_to_keep = indices[(indices != result_argmax) & (indices != result_argmin)]
                best_result = best_result[indices_to_keep]
            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list


class Logger_ddi(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]
        self.epoch_num = 10

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, eval_step, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            # argmax = result[:, 1].argmax().item()
            for i in range(result.size(0)):
                if (i+1)%self.epoch_num == 0:

                    print(f'Run {run + 1:02d}:')
                    print(f'Epoch {(i + 1)*eval_step:02d}:')
                    print(f'Train: {result[i, 0]:.2f}')
                    print(f'Valid: {result[i, 1]:.2f}')
                    print(f'Test: {result[i, 2]:.2f}')
        else:
            # result = 100 * torch.tensor(self.results)

            # best_results = []
            
            eval_num = int(len(self.results[0])/self.epoch_num)
            all_results = [[] for _ in range(eval_num)]

            for r in self.results:
                r = 100 * torch.tensor(r)

                for i in range(r.size(0)):
                    if (i+1)%self.epoch_num == 0:

                        train = r[i, 0].item()
                        valid = r[i, 1].item()
                        test = r[i, 2].item()
                
                        all_results[int((i+1)/self.epoch_num)-1].append((train, valid, test))


            for i, best_result in enumerate(all_results):
                best_result = torch.tensor(best_result)


                print(f'All runs:')
                
                epo = (i + 1)*self.epoch_num
                epo = epo*eval_step
                print(f'Epoch {epo:02d}:')


                # r = best_result[:, 0]
                # print(f'Final Train: {r.mean():.2f} ± {r.std():.2f}')

                r = best_result[:, 0]
                best_train_mean = round(r.mean().item(), 2)
                best_train_var = round(r.std().item(), 2)
                print(f'Final Train: {r.mean():.2f} ± {r.std():.2f}')

                r = best_result[:, 1]
                best_valid_mean = round(r.mean().item(), 2)
                best_valid_var = round(r.std().item(), 2)

                best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
                print(f'Final Valid: {r.mean():.2f} ± {r.std():.2f}')


                r = best_result[:, 2]
                best_test_mean = round(r.mean().item(), 2)
                best_test_var = round(r.std().item(), 2)
                print(f'Final Test: {r.mean():.2f} ± {r.std():.2f}')

                mean_list = [best_train_mean, best_valid_mean, best_test_mean]
                var_list = [best_train_var, best_valid_var, best_test_var]


            # return best_valid, best_valid_mean, mean_list, var_list


def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger