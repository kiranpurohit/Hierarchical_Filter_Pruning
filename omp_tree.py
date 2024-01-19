from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import pdb
import os, sys
import numpy as np
import torch.nn.functional as F
import copy
import math
import time
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
from collections import OrderedDict
from typing import Dict, Callable
from Util import *
import resnet

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)

class args:
    gpu='0'
    batch_size=128
    step_ft=300
    ft_lr=1e-3
    ratio=0.5
    workers=2
    model=None
    checkpoint=''


op_dict = {}
ip_dict = {}
outputs_out = []
outputs_in = []
outputs_org = []

def main():
    global args, iters
    global file
    #args = parser.parse_args()

    args.gpu = [int(i) for i in args.gpu.split(',')]
    torch.cuda.set_device(args.gpu[0] if args.gpu else None)
    torch.backends.cudnn.benchmark = True
    L_cls_f = nn.CrossEntropyLoss().cuda()

    # Dataset Loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./cifar100', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(), normalize])),
        batch_size=400, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Model Initialize and Loading
    model = resnet.resnet56()
    model = model.to(device)

    #print(model)

    args.checkpoint = './models/cifar100_resnet56.pth'

    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
    model.load_state_dict(checkpoint['net'])

    model.eval()
    original_model = copy.deepcopy(model)

    loss, init_acc = validate(val_loader, model, L_cls_f, '')
    print('\nOriginal performance. Acc: {:2.2f}%'.format(init_acc))


    # 1. Initialization process
    layer_names = get_layer_names(model)

    #expected_flops(copy.deepcopy(model), layer_names[1:], num_params, num_flops, args.ratio)

    ##################################################
    add_1x1_convs(model, layer_names[1:])


    print(model)

    num_params = get_params(model)
    num_flops = get_flops(model)

    #print("\n Number of initial parameters and flops = ", num_params, num_flops)
    
    ##########################################################
    ##########################################################

    print('== 1. Initialization fine-tuning stage. ')
    model_opt = torch.optim.SGD(
        model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    for epochs in range(0):
        fine_tuning(model, original_model, train_loader,
                    val_loader, L_cls_f, model_opt, False)
        loss, acc = validate(val_loader, model, L_cls_f, '* ')
        print("[Init {:02d}] Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(epochs+1, loss, acc, 100 - (get_params(model)/num_params*100), 100 - (get_flops(model)/num_flops*100)))

    # 2. Pruning process, from top to bottom

    start_pruning = time.time()

    print('\n== 2. Pruning stage. ')
    K=0
    pruning_ratio = args.ratio
    pruning_num = 0
    for i in range(1,len(layer_names)):
        layer_name = layer_names[i]
        layer_name = layer_name.split('.')[:-1]
        layer = model
        for i in range(len(layer_name)-1):
            layer = layer._modules[layer_name[i]]
        conv3x3 = layer._modules[layer_name[-1]][1]

        pruning_num += int(round(conv3x3.out_channels * pruning_ratio))
        pruning_num += int(round(conv3x3.in_channels * pruning_ratio))

    param_reduction = 0
    print("Total filters needed to be removed = ",pruning_num)
    while param_reduction < 69:
        model,flag,k = prune(model, original_model, layer_names, train_loader, val_loader, L_cls_f)
 
        ##########################################################
        ##########################################################
        shape_of_layers = []
        for layers in model.modules():
            if isinstance(layers, torch.nn.modules.conv.Conv2d):
                shape_of_layers.append(layers.weight.shape)
                print(layers.weight.shape)

        ##########################################################
        ##########################################################

        if flag != -1:
            K += k

        print(model)
        print(f"Total filters pruned till now = {K}")

        loss, acc = validate(val_loader, model, L_cls_f, '* ')

        print("[Pruning done: {:2.2f}%]. Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(K/pruning_num*100, loss, acc, 100 - (get_params(model)/num_params*100), 100 - (get_flops(model)/num_flops*100)))

        param_reduction = 100 - (get_params(model)/num_params*100)

    pruning_time = time.time()-start_pruning

    print('Saving pruned model..')
    state = {
        'net': model.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/omp_cifar10_resnet56_68_tree.pth')


    # 3. Final Fine-tuning stage

    start_finetune = time.time()

    print('\n==3. Final fine-tuning stage after pruning.')
    best_acc = 0
    model_opt = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    for epochs in range(args.step_ft):
        adjust_learning_rate(model_opt, epochs, args.step_ft)
        fine_tuning(model, original_model, train_loader,
                    val_loader, L_cls_f, model_opt)
        loss, acc = validate(val_loader, model, L_cls_f, '* ')
        if acc > best_acc:
            best_acc = acc

        print("[Fine-tune {:03d}] Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%  Best: {:2.2f}%".format(epochs+1, loss, acc,
                                                                                                                          100 - (get_params(model)/num_params*100), 100 - (get_flops(model)/num_flops*100), best_acc))

    print("\n[Final] Baseline: {:2.2f}%. After Pruning: {:2.2f}%. || Diff: {:2.2f}%  Param: {:2.2f}%  Flop: {:2.2f}%".format(
        init_acc, best_acc, init_acc - best_acc, 100 - (get_params(model)
        / num_params*100), 100 - (get_flops(model)/num_flops*100)))

    finetuning_time = time.time()-start_finetune

    print("Time taken for pruning = ",pruning_time)
    print("Time taken for fine tuning = ",finetuning_time)

    total_time = pruning_time + finetuning_time
    print("Total time taken = ",total_time)

def func_out(module,input,output):
  outputs_out.append(output)
  return None

def func_in(module,input,output):
  outputs_in.append(output)
  return None

def func_org(module,input,output):
  outputs_org.append(output)
  return None

def add_hooks_out(module):
  if type(module) is nn.BatchNorm2d:
    handle = module.register_forward_hook(func_out)

def add_hooks_in(module):
  if type(module) is nn.BatchNorm2d:
    handle = module.register_forward_hook(func_in)

def add_hooks_org(module):
  if type(module) is nn.BatchNorm2d:
    handle = module.register_forward_hook(func_org)

def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)

def prune(model, original_model, layer_names, train_loader, val_loader, L_cls_f):
    best_model = model
    model_copy = copy.deepcopy(model)
    best_idx = -1
    flag = -1
    k = 0

    model.eval()
    model_copy.eval()
    original_model.eval()

    error_out = [0]*30
    error_in = [0]*30

    for _, (input, target) in enumerate(val_loader):

        outputs_out.clear()
        outputs_in.clear()
        outputs_org.clear()
        torch.cuda.empty_cache()

        input = input.cuda()

        original_model.apply(add_hooks_org)

        original_model(input)

        remove_all_forward_hooks(original_model)

        req_org_tensor = outputs_org[-1]
        
        outputs_org.clear()
        torch.cuda.empty_cache()

        for i in range(1,len(layer_names)):

            layer_name = layer_names[i]

            model_copy1 = copy.deepcopy(model_copy)
            model_copy2 = copy.deepcopy(model_copy)

            model_out,pruned_out,k_out = pruning_output_channel(model_copy1, original_model, layer_name)
            model_in,pruned_in,k_in = pruning_input_channel(model_copy2, original_model, layer_name)

            layer_name = layer_name.split('.')[:-1]

            if pruned_out:
                model_out.apply(add_hooks_out)
                model_out(input)
                remove_all_forward_hooks(model_out)

                req_tensor = outputs_out[-1]
                outputs_out.clear()
                torch.cuda.empty_cache()
                
                error1_out = torch.norm(req_org_tensor - req_tensor).item()/torch.norm(req_org_tensor).item()
                #print(f'Layer{i} lst[ele]:{torch.norm(req_tensor).item()} lst[0]:{torch.norm(req_org_tensor).item()} rel_error:{error1_out}')
                error_out[i-1] += error1_out
            else:
                error_out[i-1] = float("inf")

            if pruned_in:
                model_in.apply(add_hooks_in)
                model_in(input)
                remove_all_forward_hooks(model_in)

                req_tensor = outputs_in[-1]
                outputs_in.clear()
                torch.cuda.empty_cache()

                error1_in = torch.norm(req_org_tensor - req_tensor).item()/torch.norm(req_org_tensor).item()
                #print(f'Layer{i} lst[ele]:{torch.norm(req_tensor).item()} lst[0]:{torch.norm(req_org_tensor).item()} rel_error:{error1_in}')
                error_in[i-1] += error1_in
            else:
                error_in[i-1] = float("inf")


    # print(error_in)
    # print(error_out)

    idx_out = np.argmin(error_out)
    idx_in = np.argmin(error_in)

    model_copy_new = copy.deepcopy(model_copy)

    if error_out[idx_out] > error_in[idx_in]:
        layer_name = layer_names[idx_in+1]
        model_in,pruned_in,k_in = pruning_input_channel(model_copy_new, original_model, layer_name)
        best_model = model_in
        best_idx = idx_in+1
        flag = 0
        k = k_in
    else:
        layer_name = layer_names[idx_out+1]
        model_out,pruned_out,k_out = pruning_output_channel(model_copy_new, original_model, layer_name)
        best_model = model_out
        best_idx = idx_out+1
        flag = 1
        k = k_out


    # One epoch of fine-tuning
    model_opt = torch.optim.SGD(
        best_model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
    fine_tuning(best_model, original_model, train_loader,
                val_loader, L_cls_f, model_opt)
    if flag == 1:
      print(f"\n Channel chosen: output channel of Layer {best_idx}")
      # Delete the dictionary of output channel of that layer which contains S, S' and lambda_list, as for the next iteration we have to calculate again these as this layer has been pruned and we have to calculate new values for next iteration. Other layer values will remain same as no change has been done there.
      del op_dict[layer_names[best_idx]]
    elif flag == 0:
      print(f"\n Channel chosen: input channel of Layer {best_idx}")
      del ip_dict[layer_names[best_idx]]
    elif flag == -1:
      print("\n No pruning took place")

    return(best_model,flag,k)

def pruning_output_channel(model, original_model, layer_name):

    global args

    old_name = layer_name

    layer_name = layer_name.split('.')[:-1]
    pruned = False
    num = 1
    for xx in range(num):

        layer = model
        for i in range(len(layer_name)-1):
            layer = layer._modules[layer_name[i]]
        conv3x3 = layer._modules[layer_name[-1]][1]
        conv1x1 = layer._modules[layer_name[-1]][2]
        mid_channel = conv3x3.out_channels
        pruning_ratio = args.ratio
        #pruning_num = int(round(conv3x3.out_channels * pruning_ratio))
        pruning_num = 5

        if pruning_num==0:
            break

        left = mid_channel - pruning_num
        if left <= 0:
          op_dict[old_name] = None
          break

        new_conv3x3 = nn.Conv2d(in_channels=conv3x3.in_channels, out_channels=conv3x3.out_channels - pruning_num, kernel_size=conv3x3.kernel_size,
                                stride=conv3x3.stride, padding=conv3x3.padding, bias=conv3x3.bias).cuda()

        new_conv1x1 = nn.Conv2d(in_channels=conv1x1.in_channels - pruning_num, out_channels=conv1x1.out_channels, kernel_size=(1, 1),
                                stride=conv1x1.stride, padding=(0, 0), bias=conv3x3.bias).cuda()

        # LRF calculation
        # print(conv1x1.weight.data.shape)
        # print(conv3x3.weight.data.shape)
        conv3x3_weights = conv3x3.weight.data.view(conv3x3.weight.data.shape[0], -1)

        if old_name not in op_dict:
            #######################################################################################
            cv2 = normalize(conv3x3_weights.cpu(), axis = 1)
            # print(cv2.shape)
            res = cv2

            S = []
            proj = []
            xi = []
            lamb = []

            while np.count_nonzero(lamb)< left:

                total = np.arange(mid_channel)
                S_das = list(set(total) - set(S))
                # print("\nsize of selected set S = ",len(S))
                # print(S)
                # print("size of pruning set S' = ", len(S_das))
                # print(S_das,"\n")


                lambda_list = []
                res_norm = []
                resd = []
                proj = [float('-inf') for i in range(cv2.shape[0])]
                xi = [float('-inf') for i in range(cv2.shape[0])]

                for i in S_das:
                    for j in range(cv2.shape[0]):
                        proj[j] = np.dot(res[j,:],cv2[i,:]) #projection

                    xi[i] = np.sum(np.absolute(proj))

                ind = np.argmax(xi) #argmax

                # print("xi = ",xi,"\n")

                # print("selected index = ",ind)

                S.append(ind)

                if len(S) == 1:
                    A = cv2[ind,:]
                    #S means indices, and A means the values of f:,l at those indices
                else:
                    A = np.vstack([A,cv2[ind,:]]) #Accumulating desired indices

                for j in range(cv2.shape[0]):
                    if len(S) == 1:
                        A = A.reshape(1,-1)
                        #LSQ
                        lambdaopt = lstsq(A.dot(A.T) + np.identity(A.shape[0]),A.dot(cv2[j,:].reshape(-1,1)))[0]
                        res = (cv2[j,:].reshape(-1,1) - np.dot(A.T,lambdaopt.reshape(-1,1))).reshape(1,-1)
                        resd.append(np.squeeze(res))
                        lamb = lambdaopt
                        lambda_list.append(np.squeeze(lamb.reshape(1,-1)))

                    else:
                        lambdaopt = lstsq(A.dot(A.T) + np.identity(A.shape[0]),A.dot(cv2[j,:].reshape(-1,1)))[0]
                        #Want positive lambda
                        res = (cv2[j,:].reshape(-1,1) - np.dot(A.T,lambdaopt.reshape(-1,1))).reshape(1,-1)
                        resd.append(np.squeeze(res))
                        lamb = lambdaopt
                        lambda_list.append(np.squeeze(lamb.reshape(1,-1)))

                lambda_list = np.array(lambda_list,dtype='float32')
                # print("lambda = ",lambda_list.shape)

                res = np.array(resd)
                # print("res = ",res.shape)

            total = np.arange(mid_channel)
            S = np.sort(S).tolist()
            S_das = list(set(total) - set(S))

            op_dict[old_name] = (S,S_das,lambda_list)

        else:
            S = op_dict[old_name][0]
            S_das = op_dict[old_name][1]
            lambda_list = op_dict[old_name][2]
        # print("\nsize of selected set S = ",len(S))
        # print(S)
        # print("size of pruning set S' = ", len(S_das))
        # print(S_das,"\n")

        #################################################################################################

        if len(S)==1:
            lambda_id = torch.from_numpy(lambda_list[S_das])
            lambda_id = torch.unsqueeze(lambda_id, 1)

        else:
            lambda_id = torch.from_numpy(lambda_list[S_das,:])

        # Copy the weight values of selected filters original convolution to new convolution

        new_conv3x3.weight.data[:, :, :, :] = conv3x3.weight.data[S, :, :, :]
        new_conv1x1.weight.data[:, :, :, :] = conv1x1.weight.data[:, S, :, :]

        # Weights Compensation

        compen_weight = conv1x1.weight.data[:, S_das, :, :]

        compen_weight = torch.matmul(compen_weight.view(compen_weight.shape[0], compen_weight.shape[1]), lambda_id.cuda())
        compen_weight = compen_weight.view(compen_weight.shape[0], compen_weight.shape[1], 1, 1)

        new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

        #print("\n\n\nLayer Structure before pruning = ", conv3x3.in_channels, conv3x3.out_channels)

        layer._modules[layer_name[-1]][1] = new_conv3x3
        layer._modules[layer_name[-1]][2] = new_conv1x1

        pruned = True

    return(model,pruned,pruning_num)




def pruning_input_channel(model, original_model, layer_name):

    global args

    old_name = layer_name

    layer_name = layer_name.split('.')[:-1]

    pruned = False
    num = 1
    for xx in range(num):

        layer = model
        for i in range(len(layer_name)-1):
            layer = layer._modules[layer_name[i]]
        conv1x1 = layer._modules[layer_name[-1]][0]
        conv3x3 = layer._modules[layer_name[-1]][1]
        mid_channel = conv1x1.out_channels
        pruning_ratio = args.ratio
        #pruning_num = int(round(conv3x3.in_channels * pruning_ratio))

        pruning_num = 5

        if pruning_num==0:
            break

        left = mid_channel - pruning_num
        if left <= 0:
          ip_dict[old_name] = None
          break

        new_conv1x1 = nn.Conv2d(in_channels=conv1x1.in_channels, out_channels=conv1x1.out_channels - pruning_num, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0), bias=conv3x3.bias).cuda()
        new_conv3x3 = nn.Conv2d(in_channels=conv3x3.in_channels - pruning_num, out_channels=conv3x3.out_channels, kernel_size=conv3x3.kernel_size,
                                stride=conv3x3.stride, padding=conv3x3.padding, bias=conv3x3.bias).cuda()

        # LRF calculation
        # print(conv1x1.weight.data.shape)
        # print(conv3x3.weight.data.shape)
        conv3x3_weights = conv3x3.weight.data.transpose(0, 1).contiguous().view(conv3x3.weight.data.shape[1], -1)

        if old_name not in ip_dict:
            ##########################################################################################################
            cv2 = normalize(conv3x3_weights.cpu(), axis = 1)
            # print(cv2.shape)

            res = cv2

            S = []
            proj = []
            xi = []
            lamb = []

            while np.count_nonzero(lamb)< left:

                total = np.arange(mid_channel)
                S_das = list(set(total) - set(S))
                # print("\nsize of selected set S = ",len(S))
                # print(S)
                # print("size of pruning set S' = ", len(S_das))
                # print(S_das,"\n")


                lambda_list = []
                res_norm = []
                resd = []
                proj = [float('-inf') for i in range(cv2.shape[0])]
                xi = [float('-inf') for i in range(cv2.shape[0])]
                for i in S_das:
                    for j in range(cv2.shape[0]):
                        proj[j] = np.dot(res[j,:],cv2[i,:]) #projection

                    xi[i] = np.sum(np.absolute(proj))

                ind = np.argmax(xi) #argmax

                # print("xi = ",xi,"\n")

                # print("selected index = ",ind)

                S.append(ind)

                if len(S) == 1:
                    A = cv2[ind,:]
                    #S means indices, and A means the values of f:,l at those indices
                else:
                    A = np.vstack([A,cv2[ind,:]]) #Accumulating desired indices

                for j in range(cv2.shape[0]):
                    if len(S) == 1:
                        A = A.reshape(1,-1)
                        #LSQ
                        lambdaopt = lstsq(A.dot(A.T) + np.identity(A.shape[0]),A.dot(cv2[j,:].reshape(-1,1)))[0]
                        res = (cv2[j,:].reshape(-1,1) - np.dot(A.T,lambdaopt.reshape(-1,1))).reshape(1,-1)
                        resd.append(np.squeeze(res))
                        lamb = lambdaopt
                        lambda_list.append(np.squeeze(lamb.reshape(1,-1)))

                    else:
                        lambdaopt = lstsq(A.dot(A.T) + np.identity(A.shape[0]),A.dot(cv2[j,:].reshape(-1,1)))[0]
                        #Want positive lambda
                        res = (cv2[j,:].reshape(-1,1) - np.dot(A.T,lambdaopt.reshape(-1,1))).reshape(1,-1)
                        resd.append(np.squeeze(res))
                        lamb = lambdaopt
                        lambda_list.append(np.squeeze(lamb.reshape(1,-1)))

                lambda_list = np.array(lambda_list,dtype='float32').T
                # print("lambda = ",lambda_list.shape)

                res = np.array(resd)
                # print("res = ",res.shape)

            total = np.arange(mid_channel)
            S = np.sort(S).tolist()
            S_das = list(set(total) - set(S))

            ip_dict[old_name] = (S,S_das,lambda_list)
        else:
            S = ip_dict[old_name][0]
            S_das = ip_dict[old_name][1]
            lambda_list = ip_dict[old_name][2]
        # print("\nsize of selected set S = ",len(S))
        # print(S)
        # print("size of pruning set S' = ", len(S_das))
        # print(S_das,"\n")

        ##########################################################################################################
        if len(S)==1:
            lambda_id = torch.from_numpy(lambda_list[S_das])
            lambda_id = torch.unsqueeze(lambda_id, 1)

        else:
            lambda_id = torch.from_numpy(lambda_list[:,S_das])

        # Copy the weight values of original convolution to new convolution
        # except the channel with the lowest approximation error

        new_conv1x1.weight.data[:, :, :, :] = conv1x1.weight.data[S, :, :, :]
        new_conv3x3.weight.data[:, :, :, :] = conv3x3.weight.data[:, S, :, :]


        # Weights Compensation

        compen_weight = conv1x1.weight.data[S_das, :, :, :]

        compen_weight = torch.matmul(lambda_id.view(-1,compen_weight.shape[0]).cuda(),compen_weight.view(compen_weight.shape[0], compen_weight.shape[1]))
        compen_weight = compen_weight.view(compen_weight.shape[0], compen_weight.shape[1], 1, 1)

        new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

        #print("\n\n\nLayer Structure before pruning = ", conv3x3.in_channels, conv3x3.out_channels)

        layer._modules[layer_name[-1]][0] = new_conv1x1
        layer._modules[layer_name[-1]][1] = new_conv3x3

        pruned = True

    return(model,pruned,pruning_num)



def distillation_loss(y_logit, t_logit, T=2):
    return F.kl_div(F.log_softmax(y_logit/T, 1), F.softmax(t_logit/T, 1), reduction='sum')/y_logit.size(0)


def fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, use_distill=True):

    global args
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.cuda(non_blocking=True)
        model_opt.zero_grad()
        z = model(input)
        z_ori = original_model(input)
        L = L_cls_f(z, target)
        if use_distill:
            L += distillation_loss(z, z_ori)
        L.backward()
        model_opt.step()
    model.eval()


def adjust_learning_rate(optimizer, epoch, total):
    lr = 0.01
    if epoch > total/2:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, L_cls_f, prefix='', print=False):
    global args

    loss = 0
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input = input.to(device)
            z = model(input)
            L_cls = L_cls_f(z, target)
            loss += L_cls.item()

            _, predicted = torch.max(z.data, 1)
            total += input.size(0)
            correct += (predicted == target).sum().item()

    if print:
        print('== {} Loss : {:.5f}. Acc : {:2.2f}%'.format(
            prefix, loss/len(val_loader), correct/total*100))

    return loss/len(val_loader), correct/total*100

if __name__ == '__main__':
    main()
