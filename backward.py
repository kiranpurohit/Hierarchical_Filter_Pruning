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
from scipy.linalg import lstsq 
from sklearn.preprocessing import normalize
import time

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
    ratio=0.50
    workers=2
    model=None
    checkpoint=''
    
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--step_ft', type=int, default=300)
# parser.add_argument('--ft_lr', type=float, default=1e-3)
# parser.add_argument('--ratio', type=float, default=0.5)
# parser.add_argument('--workers', type=int, default=4)
# parser.add_argument('--arch', type=str, default='resnet32')
# parser.add_argument('--model', type=str, default=None)


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
        batch_size=100, shuffle=False,
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


    ##################################################
    # 1. Initialization process
    layer_names = get_layer_names(model)

    #expected_flops(copy.deepcopy(model), layer_names[1:], num_params, num_flops, args.ratio)

    ##################################################
    add_1x1_convs(model, layer_names[1:])
    ##################################################

    num_params = get_params(model)
    num_flops = get_flops(model)

    print('== 1. Initialization fine-tuning stage. ')
    model_opt = torch.optim.SGD(
        model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    for epochs in range(0):
        fine_tuning(model, original_model, train_loader,
                    val_loader, L_cls_f, model_opt, False)
        loss, acc = validate(val_loader, model, L_cls_f, '* ')

        print("[Init {:02d}] Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(epochs+1, loss, acc,
                                                                                                     100-get_params(model)/num_params*100, 100-get_flops(model)/num_flops*100))

    # 2. Pruning process, from top to bottom

    start_pruning = time.time()
    
    print('\n== 2. Pruning stage. ')
    for i in range(1, len(layer_names)):
        index = len(layer_names)-i
        model = pruning_output_channel(model, original_model, layer_names[index], train_loader, val_loader, L_cls_f)
        model = pruning_input_channel(model, original_model, layer_names[index], train_loader, val_loader, L_cls_f)

        #print(model) 
                
        loss, acc = validate(val_loader, model, L_cls_f, '* ')
        print("[Pruning {:02d}]. Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%".format(index, loss, acc, 100-get_params(model)/num_params*100, 100-get_flops(model)/num_flops*100))
 
    pruning_time = time.time()-start_pruning                                                                                                        

    print('Saving pruned model..')
    state = {
        'net': model.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/backward_cifar10_resnet56_50.pth')


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

        print("[Fine-tune {:03d}] Loss: {:.3f}. Acc: {:2.2f}%. || Param: {:2.2f}%  Flop: {:2.2f}%  Best: {:2.2f}%".format(epochs+1, loss, acc, 100-get_params(model)/num_params*100, 100-get_flops(model)/num_flops*100, best_acc))

    print("\n[Final] Baseline: {:2.2f}%. After Pruning: {:2.2f}%. || Diff: {:2.2f}%  Param: {:2.2f}%  Flop: {:2.2f}%".format(init_acc, best_acc, init_acc - best_acc, 100-get_params(model)/ num_params*100, 100-get_flops(model)/num_flops*100))
    
    finetuning_time = time.time()-start_finetune

    print("Time taken for pruning = ",pruning_time)
    print("Time taken for fine tuning = ",finetuning_time)

    total_time = pruning_time + finetuning_time
    print("Total time taken = ",total_time)

def pruning_output_channel(model, original_model, layer_name, train_loader, val_loader, L_cls_f):

    global args
    
    # Calculate the number of channels to prune
    old_name = layer_name
    
    layer_name = layer_name.split('.')[:-1]
    
    layer = model
    for i in range(len(layer_name)-1):
        layer = layer._modules[layer_name[i]]
    conv3x3 = layer._modules[layer_name[-1]][1]
    pruning_ratio = args.ratio
    pruning_num = int(round(conv3x3.out_channels * pruning_ratio))

    layer = model
    for i in range(len(layer_name)-1):
        layer = layer._modules[layer_name[i]]
    conv3x3 = layer._modules[layer_name[-1]][1]
    conv1x1 = layer._modules[layer_name[-1]][2]
    mid_channel = conv3x3.out_channels
    new_conv3x3 = nn.Conv2d(in_channels=conv3x3.in_channels, out_channels=conv3x3.out_channels - pruning_num, kernel_size=conv3x3.kernel_size,
                            stride=conv3x3.stride, padding=conv3x3.padding, bias=conv3x3.bias).cuda()

    new_conv1x1 = nn.Conv2d(in_channels=conv1x1.in_channels - pruning_num, out_channels=conv1x1.out_channels, kernel_size=(1, 1),
                            stride=conv1x1.stride, padding=(0, 0), bias=conv3x3.bias).cuda()

    # LRF calculation
    conv3x3_weights = conv3x3.weight.data.view(conv3x3.weight.data.shape[0], -1)
    #print("output channel weights shape =",conv3x3_weights.shape)
    #######################################################################################
    #cv2 = normalize(conv3x3_weights.cpu(), axis = 1)

    cv2 = conv3x3_weights
    b = cv2.T
    A = copy.deepcopy(b)

    num = [i for i in range(0,cv2.shape[0])]
    
    S_das = []

    while A.shape[1]>(mid_channel - pruning_num):

        G = torch.inverse(torch.matmul(torch.t(A),A))

        u_k = torch.tensor([float('inf') for i in range(A.shape[1])])            
        
        for i in range(A.shape[1]):
            # \gamma_k
            l_k = G[i,i]
            # for calculating g_k and d_k
            # if i=0 row
            if i==0:
                g_k = G[i+1:,i].view(-1,1)
                d_k = torch.matmul(A[:,i+1:],g_k) + A[:,i].view(-1,1)*l_k 
            # if i=last row
            elif i==A.shape[1]:
                g_k = G[:i,i].view(-1,1)
                d_k = torch.matmul(A[:,:i],g_k) + A[:,i].view(-1,1)*l_k
            # else not first or last row
            else:
                g_k = torch.cat([G[:i,i].view(-1,1), G[i+1:,i].view(-1,1)],0)
                d_k = torch.matmul(torch.cat((A[:,:i],A[:,i+1:]),1),g_k) + A[:,i].view(-1,1)*l_k 
     
            u_k[i] = torch.nansum(torch.square(torch.matmul(torch.t(d_k),b)))/l_k

 
        ind = torch.argmin(u_k)
        #print("index = ",ind)
        S_das.append(num[ind])
        num.pop(ind)

        # updating filter weights matrix by removing indth col
        # if first col
        if ind==0:
            A = A[:,ind+1:]
        # if last col
        elif ind==A.shape[1]:
            A = A[:,:ind]
        else:
            A = torch.cat([A[:,:ind],A[:,ind+1:]], 1)


    A=A.cpu().numpy()
    b=b.cpu().numpy()
    lambda_id = lstsq(A, b)
    lambda_id = lambda_id[0]


    lambda_id = torch.from_numpy(lambda_id).T
    #print("lambda_id = ", lambda_id.shape)
    #################################################################################################

    total = np.arange(mid_channel)
    S_das = np.sort(S_das).tolist()
    S = list(set(total) - set(S_das))
    print("S = ",S)

    # Copy the weight values of selected filters original convolution to new convolution

    new_conv3x3.weight.data[:, :, :, :] = conv3x3.weight.data[S, :, :, :]
    new_conv1x1.weight.data[:, :, :, :] = conv1x1.weight.data[:, S, :, :]

    # Weights Compensation

    compen_weight = conv1x1.weight.data[:, S_das, :, :]

    lambda_id = lambda_id[S_das,:]

    compen_weight = torch.matmul(compen_weight.view(compen_weight.shape[0], compen_weight.shape[1]), lambda_id.cuda())
    compen_weight = compen_weight.view(compen_weight.shape[0], compen_weight.shape[1], 1, 1)

    new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

    layer._modules[layer_name[-1]][1] = new_conv3x3
    layer._modules[layer_name[-1]][2] = new_conv1x1

    print(layer)       
 
    # One epoch of fine-tuning
    model_opt = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
    fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt)

    return model


def pruning_input_channel(model, original_model, layer_name, train_loader, val_loader, L_cls_f):

    global args

    old_name = layer_name

    layer_name = layer_name.split('.')[:-1]
    layer = model
    for i in range(len(layer_name)-1):
        layer = layer._modules[layer_name[i]]
    conv3x3 = layer._modules[layer_name[-1]][1]

    # Calculate the number of channels to prune
    pruning_ratio = args.ratio
    pruning_num = int(round(conv3x3.in_channels * pruning_ratio))

    layer = model
    for i in range(len(layer_name)-1):
        layer = layer._modules[layer_name[i]]
    conv1x1 = layer._modules[layer_name[-1]][0]
    conv3x3 = layer._modules[layer_name[-1]][1]
    mid_channel = conv1x1.out_channels

    new_conv1x1 = nn.Conv2d(in_channels=conv1x1.in_channels, out_channels=conv1x1.out_channels - pruning_num, kernel_size=(1, 1),
                            stride=(1, 1), padding=(0, 0), bias=conv3x3.bias).cuda()
    new_conv3x3 = nn.Conv2d(in_channels=conv3x3.in_channels - pruning_num, out_channels=conv3x3.out_channels, kernel_size=conv3x3.kernel_size,
                            stride=conv3x3.stride, padding=conv3x3.padding, bias=conv3x3.bias).cuda()

    # LRF calculation
    conv3x3_weights = conv3x3.weight.data.transpose(0, 1).contiguous().view(conv3x3.weight.data.shape[1], -1)

    ##########################################################################################################
    #cv2 = normalize(conv3x3_weights.cpu(), axis = 1)
 
    cv2 = conv3x3_weights
    b = cv2.T
    A = copy.deepcopy(b)

    num = [i for i in range(0,cv2.shape[0])]
    
    S_das = []

    while A.shape[1]>(mid_channel - pruning_num):

        G = torch.inverse(torch.matmul(torch.t(A),A))

        u_k = torch.tensor([float('inf') for i in range(A.shape[1])])            
        
        for i in range(A.shape[1]):
            # \gamma_k
            l_k = G[i,i]
            # for calculating g_k and d_k
            # if i=0 row
            if i==0:
                g_k = G[i+1:,i].view(-1,1)
                d_k = torch.matmul(A[:,i+1:],g_k) + A[:,i].view(-1,1)*l_k 
            # if i=last row
            elif i==A.shape[1]:
                g_k = G[:i,i].view(-1,1)
                d_k = torch.matmul(A[:,:i],g_k) + A[:,i].view(-1,1)*l_k
            # else not first or last row
            else:
                g_k = torch.cat([G[:i,i].view(-1,1), G[i+1:,i].view(-1,1)],0)
                d_k = torch.matmul(torch.cat((A[:,:i],A[:,i+1:]),1),g_k) + A[:,i].view(-1,1)*l_k 
            #print("g_k shape = ", g_k.shape, "d_K shape = ", d_k.shape, "b shape:", b.shape)         
            u_k[i] = torch.nansum(torch.square(torch.matmul(torch.t(d_k),b)))/l_k

        ind = torch.argmin(u_k)
        #print("index = ",ind)
        S_das.append(num[ind])
        num.pop(ind)

        # updating filter weights matrix by removing indth col
        # if first col
        if ind==0:
            A = A[:,ind+1:]
        # if last col
        elif ind==A.shape[1]:
            A = A[:,:ind]
        else:
            A = torch.cat([A[:,:ind],A[:,ind+1:]], 1)


    A=A.cpu().numpy()
    b=b.cpu().numpy()
    lambda_id = lstsq(A, b)
    lambda_id = lambda_id[0]


    lambda_id = torch.from_numpy(lambda_id).T

    ##########################################################################################################

    total = np.arange(mid_channel)
    S_das = np.sort(S_das).tolist()
    S = list(set(total) - set(S_das))
    print("S = ",S)

    # Copy the weight values of original convolution to new convolution
    # except the channel with the lowest approximation error

    new_conv1x1.weight.data[:, :, :, :] = conv1x1.weight.data[S, :, :, :]
    new_conv3x3.weight.data[:, :, :, :] = conv3x3.weight.data[:, S, :, :]

    # Weights Compensation

    compen_weight = conv1x1.weight.data[S_das, :, :, :]

    lambda_id = lambda_id[S_das,:]

    compen_weight = torch.matmul(lambda_id.view(-1,compen_weight.shape[0]).cuda(),compen_weight.view(compen_weight.shape[0], compen_weight.shape[1]))
    compen_weight = compen_weight.view(compen_weight.shape[0], compen_weight.shape[1], 1, 1)

    new_conv1x1.weight.data = new_conv1x1.weight.data + compen_weight

    layer._modules[layer_name[-1]][0] = new_conv1x1
    layer._modules[layer_name[-1]][1] = new_conv3x3

    print(layer)

    # One epoch of fine-tuning
    model_opt = torch.optim.SGD(
        model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=1e-4)
    fine_tuning(model, original_model, train_loader,
                val_loader, L_cls_f, model_opt)

    return model


def distillation_loss(y_logit, t_logit, T=2):
    return F.kl_div(F.log_softmax(y_logit/T, 1), F.softmax(t_logit/T, 1), reduction='sum')/y_logit.size(0)


def fine_tuning(model, original_model, train_loader, val_loader, L_cls_f, model_opt, use_distill=True):

    global args
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        input = input.cuda()
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
            input = input.cuda()
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

