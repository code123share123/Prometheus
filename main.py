from itertools import repeat
from pyexpat.errors import XML_ERROR_UNEXPECTED_STATE
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups_vectorized
#from Dec_BiO import *

import random

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import numpy as np
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter 
import copy

import urllib.request
from sklearn.datasets import load_svmlight_file
urllib.request.urlretrieve ("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a", "a9a")
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t","a9a.t")



class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', default=5, type=int, help='the number of agents')#default=5
    parser.add_argument('--network_probability', default=0.5, type=int, help='the edge connectivity probability')
    parser.add_argument('--train_size',default=1000,type=int, help='training dataset size for each agent') #default=1000
    parser.add_argument('--train_size_per_batch',default=2,type=int, help='training dataset size for each agent and each batch')# #default=2
    parser.add_argument('--flag_GT',default=1,type=bool, help='flag of using gradient tracking')#default=1
    parser.add_argument('--epochs', default=100, type=int, help='epoch numbers')#default=100
    parser.add_argument('--seed_num', default=15, type=int, help='seed numbers')#default=10
    parser.add_argument('--val_size', type=int, default=1000, help='validation dataset size for each agent') #default=1000
    parser.add_argument('--val_size_per_batch', type=int, default=2, help='validation dataset size for each agent and each batch') #default=2
    parser.add_argument('--alg', type=str, default='PROMETHEUS', choices=['PROMETHEUS', 'Prox-DSGD','PROMETHEUS-SG', 'PROMETHEUS-dir',]) #default=1000
    parser.add_argument('--hessian_q', type=int, default=10, help='number of steps to approximate hessian')#default=10
    parser.add_argument('--L_g', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    parser.add_argument('-learning_rate_upper_level',  default=0.1, type=int, help='Experiment name')
    parser.add_argument('-learning_rate_lower_level',  default=0.1, type=int, help='Experiment name')
    args, unknown = parser.parse_known_args() 

        
        
    if not args.save_folder:
        args.save_folder = './save_results'
    args.model_name = '{}_m_{}_pc_{}_GT_{}_train_{}_{}_{}_{},val_{}_T_{}_hessianq_{}_lr_u_{}_lr_l_{}'.format(args.alg, args.num_agents, args.network_probability,
                        args.flag_GT,args.train_size, args.train_size_per_batch,args.val_size, args.val_size_per_batch,  
                        args.seed_num, args.epochs, args.hessian_q,args.learning_rate_upper_level,args.learning_rate_lower_level)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    # parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    return args


import torch
from torch.autograd import grad
from torch.nn import functional as F


def Dec_stocbio(params, hparams, val_data_list, args, out_f, reg_f):
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

    # Hessian
    z_list = []
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams, output, reg_f)

    G_gradient = torch.reshape(params, [-1]) - args.L_g * torch.reshape(Gy_gradient, [-1])

    for _ in range(args.hessian_q):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z_list.append(v_0)
    v_Q = v_0 + torch.sum(torch.stack(z_list), dim=0)

    # k =np.random.random_integers(1,args.hessian_q)
    # Jacobian = torch.matmul(G_gradient, v_0)
    # v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
    # v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
    # for i in range(k):
    # z_list.append(v_0)
    # v_Q = v_0+torch.sum(torch.stack(z_list), dim=0)

    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams, retain_graph=True)[0]
    outer_update = -Gyx_gradient

    return outer_update


def gradient_fy(args, labels, params, data, output):
    loss = F.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, params)[0]
    return grad


def gradient_gy(args, labels_cp, params, data, hparams, output, reg_f):
    # For MNIST data-hyper cleaning experiments
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    # For NewsGroup l2reg expriments
    # loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, hparams, loss)
    grad = torch.autograd.grad(loss_regu, params, create_graph=True)[0]
    return grad


def from_sparse(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))



def generate_network(num_agents,network_probability):
    G=nx.erdos_renyi_graph(num_agents,network_probability,seed=1)
    nx.draw_circular(G,with_labels=True)
    plt.savefig('n_{}_pc_{}_network.png'.format(num_agents,network_probability))
    L=nx.laplacian_matrix(G)
    L=L.todense()
    W=np.identity(num_agents)-2/(3*max(np.linalg.eigvals(L)))*L

    return W,L
    
    
    
    
    
    
    
    
# Constant
warm_start = True
val_log_interval = 1

    

cuda = True and torch.cuda.is_available()
default_tensor_str = 'torch.cuda.FloatTensor' if cuda else 'torch.FloatTensor'
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
torch.set_default_tensor_type(default_tensor_str)
    #torch.multiprocessing.set_start_method('forkserver')


    #-------------------------------- Functions------------------------------------------------- 
def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()
def train_loss(params, hparams, data):
    x_mb, y_mb = data
    # print(x_mb.size()) = torch.Size([5657, 130107])
    out = out_f(x_mb,  params)
    return F.cross_entropy(out, y_mb) + reg_f(params, hparams)

def val_loss(opt_params, hparams):
    val_loss = 0
    acc = 0
    for m in range(args.num_agents):
        index = random.randint(0,args.val_size//args.val_size_per_batch-1)
        if args.alg == 'PROMETHEUS':
            out = out_f(xmb_val[m][index],  opt_params[m][:len(parameters[0][m])])
        elif args.alg == 'PROMETHEUS-dir':
            out = out_f(xmb_val[m][index],  opt_params[m][:len(parameters[0][m])])
        elif args.alg == 'PROMETHEUS-SG':
            out = out_f(xmb_val[m][index],  opt_params[m][:len(parameters[0][m])])
        elif args.alg == 'Prox-DSGD':
            out = out_f(xmb_val[m][index],  opt_params[m][:len(parameters[m])])
        
        val_loss+= F.cross_entropy(out,ymb_val[m][index])
 
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc+= pred.eq(ymb_val[m][index].view_as(pred)).sum().item() / len(ymb_val[m][index])

    val_loss = val_loss/args.num_agents
    acc = acc/args.num_agents
    val_losses.append(tonp(val_loss))
    val_accs.append(acc)
    return val_loss

def reg_f(params, l2_reg_params, l1_reg_params=None):
    ones_dxc = torch.ones(params.size())
    r = 0.5 * ((params ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
    if l1_reg_params is not None:
        r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
    return r
def out_f(x, params):
    out = x @ params
    return out

def reg_fs(params, hparams, loss):
    reg = reg_f(params, hparams)
    return torch.mean(loss)+reg

def eval(params, x, y):
    loss = 0.0
    acc = 0.0
    for m in range(args.num_agents):
        out = out_f(x[m],  params[m])
        loss+= F.cross_entropy(out, y[m])
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc+= pred.eq(y[m].view_as(pred)).sum().item() / len(y[m])
    loss = loss/args.num_agents
    acc = acc/args.num_agents
    return loss, acc
    #------------------------------------------------------------------------------------------
      
      
      
      
args = parse_args()
#Set the Network Matrix M
Network,L =  generate_network(args.num_agents,args.network_probability)


    
# load twentynews and preprocess

X, y = load_svmlight_file("a9a")
# x_test,y_test = load_svmlight_file("a9a.t")
# x_train, x_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=val_size_ratio)
y = (y+1)/2
test_size_ratio = 0.2
val_size_ratio = 0.5



X2, x_test, y2, y_test = train_test_split(X, y, stratify=y, test_size=test_size_ratio)
x_train, x_val, y_train, y_val = train_test_split(X2, y2, stratify=y2, test_size=val_size_ratio)
train_samples, n_features = x_train.shape
test_samples, n_features = x_test.shape
val_samples, n_features = x_val.shape
n_classes = np.unique(y_train).shape[0]
# train_samples=5657, val_samples=5657, test_samples=7532, n_features=130107, n_classes=20
print('Dataset a9a, train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
    % (train_samples, val_samples, test_samples, n_features, n_classes))
ys = [frnp(y_train).long(), frnp(y_val).long(), frnp(y_test).long()]
xs = [x_train, x_val, x_test]

if cuda:
    xs = [from_sparse(x).cuda() for x in xs]
else:
    xs = [from_sparse(x) for x in xs]

x_train, x_val, x_test = xs
y_train, y_val, y_test = ys
    
# torch.DataLoader has problems with sparse tensor on GPU    
iterators, train_list, val_list = [], [], []
xmb_train, xmb_val, xmb_test, ymb_train, ymb_val, ymb_test = [], [], [], [], [], []
# For minibatch method, we build the list to store the splited tensor
  
#if args.alg == 'PROMETHEUS' or 'Prox-DSGD' or 'PROMETHEUS-dir' or 'PROMETHEUS-SG':
train_iterator = CustomTensorIterator([x_train, y_train], batch_size=args.train_size_per_batch, shuffle=True, **kwargs)
val_iterator = CustomTensorIterator([x_val, y_val], batch_size=args.val_size_per_batch, shuffle=True, **kwargs)
test_iterator = CustomTensorIterator([x_test, y_test], batch_size=test_samples//args.num_agents, shuffle=True, **kwargs)
for m in range(args.num_agents):
    x_mb,y_mb = [],[]
    for _ in range(args.train_size//args.train_size_per_batch):
        data_temp = next(train_iterator)
        x_mb_temp, y_mb_temp = data_temp
        x_mb.append(x_mb_temp)
        y_mb.append(y_mb_temp)
    xmb_train.append(x_mb)
    ymb_train.append(y_mb)
    

train_iterator_all = CustomTensorIterator([x_train, y_train], batch_size=args.train_size, shuffle=True, **kwargs)
x_train_all=[]
y_train_all=[]
for m in range(args.num_agents):
    data_temp_all= next(train_iterator_all)
    x_mb_all, y_mb_all = data_temp_all
    x_train_all.append(x_mb_all)
    y_train_all.append(y_mb_all)

        
val_list = []
for m in range(args.num_agents):
    x_mb,y_mb = [],[]
    for _ in range(args.val_size//args.val_size_per_batch):
        data_temp = next(val_iterator)
        x_mb_temp, y_mb_temp = data_temp
        x_mb.append(x_mb_temp)
        y_mb.append(y_mb_temp)
    xmb_val.append(x_mb)
    ymb_val.append(y_mb)
    val_list.append([x_mb,y_mb])

val_iterator_all = CustomTensorIterator([x_val, y_val], batch_size=args.val_size, shuffle=True, **kwargs)

val_list_all = []
x_val_all=[]
y_val_all=[]
for m in range(args.num_agents):
    x_mb,y_mb = [],[]
    for _ in range(3):
        data_temp = next(val_iterator_all)
        x_mb_temp, y_mb_temp = data_temp
        x_mb.append(x_mb_temp)
        y_mb.append(y_mb_temp)
    x_val_all.append(x_mb)
    y_val_all.append(y_mb)
    val_list_all.append([x_mb,y_mb])    
    
    
    
for m in range(args.num_agents):
    data_temp = next(test_iterator)
    x_mb, y_mb = data_temp
    xmb_test.append(x_mb)
    ymb_test.append(y_mb)

        # set up another train_iterator & val_iterator to make sure train_list and val_list are full
iterators = []
for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
    iterators.append(repeat([x, y]))
train_iterator, val_iterator = iterators




import copy

# Basic Setting
seed_list_all = [0, 7, 13, 14, 15, 29, 36, 45, 57, 60, 63, 66, 67, 42, 52, 62, 72, 82]
seed_list = seed_list_all[:args.seed_num]
seed_i = 0

total_time = 0
val_losses, val_accs = [], []
loss_acc_time_results = np.zeros((args.epochs + 1, args.seed_num, 4))

for seed in seed_list:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize hyper-parameters and parameters
    hparams = []
    hparams_space = []
    parameters = []
    for m in range(args.num_agents):
        l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
        l1_reg_params = (0. * torch.ones(1)).requires_grad_(True)  # one l1 hp only (best when really low)
        hparams.append(l2_reg_params)
        space_h = torch.ones(n_features).requires_grad_(True)
        hparams_space.append(space_h)
        w = torch.zeros(n_features, n_classes).requires_grad_(True)
        parameters.append(w)

    test_loss, test_acc = eval(parameters, xmb_test, ymb_test)
    loss_acc_time_results[0, seed_i, 0] += test_loss
    loss_acc_time_results[0, seed_i, 1] += test_acc
    loss_acc_time_results[0, seed_i, 2] += 0.0

    for o_step in range(args.epochs):
        outer_lr = args.learning_rate_upper_level
        inner_lr = args.learning_rate_lower_level

        start_time = time.time()
        if args.alg == 'Prox-DSGD':  # --------------------------------Prox-DSGD-------------------------------------------------------
            if o_step == 0:
                inner_grad = [0] * args.num_agents
                outer_grad = [0] * args.num_agents

                # train_index_list = torch.randperm(train_list_len)
                # val_index = torch.randperm(val_list_len)
            for m in range(args.num_agents):

                index = random.randint(0, args.train_size // args.train_size_per_batch - 1)
                loss_train = train_loss(parameters[m], hparams[m], [xmb_train[m][index], ymb_train[m][index]])
                inner_grad[m] = torch.autograd.grad(loss_train, parameters[m])

                outer_index = random.randint(0, args.val_size // args.val_size_per_batch - 2 - 1)
                val_list_Bio = [xmb_val[m][outer_index:outer_index + 3], ymb_val[m][outer_index:outer_index + 3]]
                outer_grad[m] = Dec_stocbio(parameters[m], hparams[m], val_list_Bio, args, out_f, reg_fs)

                if args.num_agents == 1:
                    parameters[m] = parameters[m] - inner_lr * inner_grad[m][0]
                    hparams[0] = hparams[0] - outer_lr * outer_grad[m]
                else:
                    parameters[m] = parameters[m] - inner_lr * inner_grad[m][0]
                    hparams[m] = Network[m, m] * hparams[m]
                    for k in range(0, args.num_agents):
                        if L[m, k] != 0 and k != m:
                            hparams[m] += Network[m, k] * hparams[k]
                    hparams[m] -= outer_lr * outer_grad[m]
                hparams_set = np.ones(len(hparams[m]))
                v_norm = np.sqrt(sum(hparams_set ** 2))
                proj_of_hparams_on_v = (np.dot(hparams[m].tolist(), hparams_set) / v_norm ** 2) * hparams_set
                hparams[m] = torch.tensor(proj_of_hparams_on_v).requires_grad_(True)
            val_loss(parameters, hparams)

        elif args.alg == 'PROMETHEUS-SG':  # ---------------------------------------PROMETHEUS-SG-----------------------------------------------

            if o_step == 0:
                hparams = [list(range(args.num_agents))] * (args.epochs + 1)
                parameters = [list(range(args.num_agents))] * (args.epochs + 1)
                for m in range(args.num_agents):
                    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
                    hparams[0][m] = l2_reg_params
                    w = torch.zeros(n_features, n_classes).requires_grad_(True)
                    parameters[0][m] = w
                v_inner_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                p_outer_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                primal = [torch.zeros(n_features)] * args.num_agents

            for m in range(args.num_agents):

                inner_index = random.randint(0, args.train_size // args.train_size_per_batch - 1)
                loss_train = train_loss(parameters[o_step][m], hparams[o_step][m],
                                        [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                inner_grad = torch.autograd.grad(loss_train, parameters[o_step][m])

                if o_step == 0:
                    v_inner_grad[o_step][m] = inner_grad[0]
                else:
                    loss_train_old = train_loss(parameters[o_step - 1][m], hparams[o_step - 1][m],
                                                [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                    v_inner_grad_result = torch.autograd.grad(loss_train_old, parameters[o_step - 1][m])
                    v_inner_grad[o_step][m] = v_inner_grad_result[0]

                    # Outer gradient

                outer_index = random.randint(0, args.val_size // args.val_size_per_batch - 2 - 1)
                val_list_Bio = [xmb_val[m][outer_index:outer_index + 3], ymb_val[m][outer_index:outer_index + 3]]
                outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f, reg_fs)

                if o_step == 0:
                    p_outer_grad[o_step][m] = outer_grad
                else:
                    p_outer_grad[o_step][m] = Dec_stocbio(parameters[o_step - 1][m], hparams[o_step - 1][m],
                                                          val_list_Bio, args, out_f, reg_fs)

                # Gradient Tracking
                if o_step == 0:
                    primal[m] = p_outer_grad[o_step][m]
                else:

                    if args.num_agents == 1:
                        primal[0] = p_outer_grad[o_step][0]
                    else:
                        if args.flag_GT == 1:
                            primal[m] = Network[m, m] * primal[m]
                            for k in range(args.num_agents):
                                if L[m, k] != 0 and k != m:
                                    primal[m] += Network[m, k] * primal[k]
                            primal[m] += (p_outer_grad[o_step][m] - p_outer_grad[o_step - 1][m])
                        else:
                            primal[m] = p_outer_grad[o_step][m]

                if args.num_agents == 1:
                    # Outer parameter update
                    hparams[o_step + 1][0] = hparams[o_step][0] - outer_lr * primal[0]
                    # Inner parameter update
                    parameters[o_step + 1][0] = parameters[o_step][0] - inner_lr * v_inner_grad[o_step][0]
                else:
                    # Outer parameter update
                    hparams[o_step + 1][m] = Network[m, m] * hparams[o_step][m]
                    for k in range(args.num_agents):
                        if L[m, k] != 0 and k != m:
                            hparams[o_step + 1][m] += Network[m, k] * hparams[o_step][k]
                    hparams[o_step + 1][m] = hparams[o_step + 1][m] - outer_lr * primal[m]
                    # pro
                    hparams_set = np.ones(len(hparams[o_step + 1][m]))
                    v_norm = np.sqrt(sum(hparams_set ** 2))
                    proj_of_hparams_on_v = (np.dot(hparams[o_step + 1][m].tolist(),
                                                   hparams_set) / v_norm ** 2) * hparams_set
                    hparams[o_step + 1][m] = torch.tensor(proj_of_hparams_on_v).requires_grad_(True)
                    # Inner parameter update
                    parameters[o_step + 1][m] = parameters[o_step][m] - inner_lr * v_inner_grad[o_step][m]

            val_loss(parameters[o_step + 1], hparams[o_step + 1])

        elif args.alg == 'PROMETHEUS':  # ---------------------------------------PROMETHEUS----------------------------------------------

            if o_step == 0:
                hparams = [list(range(args.num_agents))] * (args.epochs + 1)
                parameters = [list(range(args.num_agents))] * (args.epochs + 1)
                for m in range(args.num_agents):
                    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
                    hparams[0][m] = l2_reg_params
                    w = torch.zeros(n_features, n_classes).requires_grad_(True)
                    parameters[0][m] = w
                v_inner_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                p_outer_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                primal = [torch.zeros(n_features)] * args.num_agents

            for m in range(args.num_agents):

                q = 5
                # variance reduction
                if o_step % q == 0:
                    loss_train = train_loss(parameters[o_step][m], hparams[o_step][m], [x_train_all[m], y_train_all[m]])
                    inner_grad = torch.autograd.grad(loss_train, parameters[o_step][m])
                    v_inner_grad[o_step][m] = inner_grad[0]
                else:
                    inner_index = random.randint(0, args.train_size // args.train_size_per_batch - 1)
                    loss_train = train_loss(parameters[o_step][m], hparams[o_step][m],
                                            [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                    inner_grad = torch.autograd.grad(loss_train, parameters[o_step][m])
                    loss_train_old = train_loss(parameters[o_step - 1][m], hparams[o_step - 1][m],
                                                [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                    inner_grad_old = torch.autograd.grad(loss_train_old, parameters[o_step - 1][m])
                    v_inner_grad[o_step][m] = inner_grad[0] - inner_grad_old[0] + v_inner_grad[o_step - 1][m]

                    # Outer gradient

                val_list_Bio = [x_val_all[m][0:3], y_val_all[m][0:3]]
                outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f, reg_fs)
                p_outer_grad[o_step][m] = outer_grad
                if o_step % q == 0:
                    val_list_Bio = [x_val_all[m][0:3], y_val_all[m][0:3]]
                    outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f,
                                             reg_fs)
                    p_outer_grad[o_step][m] = outer_grad
                else:
                    outer_index = random.randint(0, args.val_size // args.val_size_per_batch - 2 - 1)
                    val_list_Bio = [xmb_val[m][outer_index:outer_index + 3], ymb_val[m][outer_index:outer_index + 3]]
                    outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f,
                                             reg_fs)
                    outer_grad_old = Dec_stocbio(parameters[o_step - 1][m], hparams[o_step - 1][m], val_list_Bio, args,
                                                 out_f, reg_fs)
                    p_outer_grad[o_step][m] = outer_grad + (p_outer_grad[o_step - 1][m] - outer_grad_old)

                # Gradient Tracking
                if o_step == 0:
                    primal[m] = p_outer_grad[o_step][m]
                else:

                    if args.num_agents == 1:
                        primal[0] = p_outer_grad[o_step][0]
                    else:
                        if args.flag_GT == 1:
                            primal[m] = Network[m, m] * primal[m]
                            for k in range(args.num_agents):
                                if L[m, k] != 0 and k != m:
                                    primal[m] += Network[m, k] * primal[k]
                            primal[m] += (p_outer_grad[o_step][m] - p_outer_grad[o_step - 1][m])
                        else:
                            primal[m] = p_outer_grad[o_step][m]

                if args.num_agents == 1:
                    # Outer parameter update
                    hparams[o_step + 1][0] = hparams[o_step][0] - outer_lr * primal[0]
                    # Inner parameter update
                    parameters[o_step + 1][0] = parameters[o_step][0] - inner_lr * v_inner_grad[o_step][0]
                else:
                    # Outer parameter update
                    hparams_tuide = - primal[m] + hparams[o_step + 1][m]
                    hparams_set = np.ones(len(hparams[o_step + 1][m]))
                    v_norm = np.sqrt(sum(hparams_set ** 2))
                    proj_of_hparams_on_v = (np.dot(hparams_tuide.tolist(), hparams_set) / v_norm ** 2) * hparams_set
                    hparams[o_step + 1][m] = torch.tensor(proj_of_hparams_on_v).requires_grad_(True)
                    # consensus
                    hparams[o_step + 1][m] = Network[m, m] * hparams[o_step][m]
                    for k in range(args.num_agents):
                        if L[m, k] != 0 and k != m:
                            hparams[o_step + 1][m] += Network[m, k] * hparams[o_step][k]

                    hparams[o_step + 1][m] = hparams[o_step + 1][m] - outer_lr * (
                                torch.tensor(proj_of_hparams_on_v).requires_grad_(True) - hparams[o_step + 1][m])

                    # Inner parameter update
                    parameters[o_step + 1][m] = parameters[o_step][m] - inner_lr * v_inner_grad[o_step][m]

            val_loss(parameters[o_step + 1], hparams[o_step + 1])





        elif args.alg == 'PROMETHEUS-dir':  # ---------------------------------------PROMETHEUS-dir----------------------------------------------

            q = 7
            if o_step == 0:
                hparams = [list(range(args.num_agents))] * (args.epochs + 1)
                parameters = [list(range(args.num_agents))] * (args.epochs + 1)
                for m in range(args.num_agents):
                    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
                    hparams[0][m] = l2_reg_params
                    w = torch.zeros(n_features, n_classes).requires_grad_(True)
                    parameters[0][m] = w
                v_inner_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                p_outer_grad = [list(range(args.num_agents))] * (args.epochs + 1)
                primal = [torch.zeros(n_features)] * args.num_agents

            for m in range(args.num_agents):

                # variance reduction inner gradient
                if o_step % q == 0:
                    loss_train = train_loss(parameters[o_step][m], hparams[o_step][m], [x_train_all[m], y_train_all[m]])
                    inner_grad = torch.autograd.grad(loss_train, parameters[o_step][m])
                    v_inner_grad[o_step][m] = inner_grad[0]
                else:
                    inner_index = random.randint(0, args.train_size // args.train_size_per_batch - 1)
                    loss_train = train_loss(parameters[o_step][m], hparams[o_step][m],
                                            [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                    inner_grad = torch.autograd.grad(loss_train, parameters[o_step][m])
                    loss_train_old = train_loss(parameters[o_step - 1][m], hparams[o_step - 1][m],
                                                [xmb_train[m][inner_index], ymb_train[m][inner_index]])
                    inner_grad_old = torch.autograd.grad(loss_train_old, parameters[o_step - 1][m])
                    v_inner_grad[o_step][m] = inner_grad[0] + (v_inner_grad[o_step - 1][m] - inner_grad_old[0])

                    # variance reduction Outer gradient

                if o_step % q == 0:
                    val_list_Bio = [x_val_all[m][0:3], y_val_all[m][0:3]]
                    outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f,
                                             reg_fs)
                    p_outer_grad[o_step][m] = outer_grad
                else:
                    outer_index = random.randint(0, args.val_size // args.val_size_per_batch - 2 - 1)
                    val_list_Bio = [xmb_val[m][outer_index:outer_index + 3], ymb_val[m][outer_index:outer_index + 3]]
                    outer_grad = Dec_stocbio(parameters[o_step][m], hparams[o_step][m], val_list_Bio, args, out_f,
                                             reg_fs)
                    outer_grad_old = Dec_stocbio(parameters[o_step - 1][m], hparams[o_step - 1][m], val_list_Bio, args,
                                                 out_f, reg_fs)
                    p_outer_grad[o_step][m] = outer_grad + (p_outer_grad[o_step - 1][m] - outer_grad_old)

                    # Gradient Tracking
                if o_step == 0:
                    primal[m] = p_outer_grad[o_step][m]
                else:

                    if args.num_agents == 1:
                        primal[0] = p_outer_grad[o_step][0]
                    else:
                        if args.flag_GT == 1:
                            primal[m] = Network[m, m] * primal[m]
                            for k in range(args.num_agents):
                                if L[m, k] != 0 and k != m:
                                    primal[m] += Network[m, k] * primal[k]
                            primal[m] += (p_outer_grad[o_step][m] - p_outer_grad[o_step - 1][m])
                        else:
                            primal[m] = p_outer_grad[o_step][m]

                if args.num_agents == 1:
                    # Outer parameter update
                    hparams[o_step + 1][0] = hparams[o_step][0] - outer_lr * primal[0]
                    # Inner parameter update
                    parameters[o_step + 1][0] = parameters[o_step][0] - inner_lr * v_inner_grad[o_step][0]

                else:
                    # Outer parameter update
                    hparams[o_step + 1][m] = Network[m, m] * hparams[o_step][m]
                    for k in range(args.num_agents):
                        if L[m, k] != 0 and k != m:
                            hparams[o_step + 1][m] += Network[m, k] * hparams[o_step][k]
                    hparams[o_step + 1][m] = hparams[o_step + 1][m] - outer_lr * primal[m]
                    # pro
                hparams_set = np.ones(len(hparams[o_step + 1][m]))
                v_norm = np.sqrt(sum(hparams_set ** 2))
                proj_of_hparams_on_v = (np.dot(hparams[o_step + 1][m].tolist(),
                                               hparams_set) / v_norm ** 2) * hparams_set
                hparams[o_step + 1][m] = torch.tensor(proj_of_hparams_on_v).requires_grad_(True)
                # Inner parameter update
                parameters[o_step + 1][m] = parameters[o_step][m] - inner_lr * v_inner_grad[o_step][m]

            val_loss(parameters[o_step + 1], hparams[o_step + 1])

        iter_time = time.time() - start_time
        total_time += iter_time
        if o_step % val_log_interval == 0:
            if args.alg == 'Prox-DSGD':
                test_loss, test_acc = eval(parameters[:len(parameters)], xmb_test, ymb_test)
            elif args.alg == 'PROMETHEUS':
                test_loss, test_acc = eval(parameters[o_step + 1], xmb_test, ymb_test)
            elif args.alg == 'PROMETHEUS-SG':
                test_loss, test_acc = eval(parameters[o_step + 1], xmb_test, ymb_test)
            elif args.alg == 'PROMETHEUS-dir':
                test_loss, test_acc = eval(parameters[o_step + 1], xmb_test, ymb_test)
            loss_acc_time_results[o_step + 1, seed_i, 0] += test_loss
            loss_acc_time_results[o_step + 1, seed_i, 1] += test_acc
            loss_acc_time_results[o_step + 1, seed_i, 2] += total_time
            print('o_step={} ({:.2e}s) '.format(o_step, iter_time))
            print('          Test loss: {:.4e}, Test Acc: {:.2f}%'.format(test_loss, 100 * test_acc))

    seed_i += 1

file_name = 'results.npy'
file_addr = os.path.join(args.save_folder, file_name)
with open(file_addr, 'wb') as f:
    np.save(f, loss_acc_time_results)

print(loss_acc_time_results)

print('HPO ended in {:.2e} seconds\n'.format(total_time))










