from __future__ import division
import torch
import torch.nn as nn
from thop import profile
import copy
def check(dd):
	print(dd)

def get_layer_names(model):
	layer_names = []
	for name, p in model.named_parameters():
		if len(p.size())==4 and p.size(3)>1:
			layer_names.append(name)
	return layer_names
	

def add_1x1_convs(model, layer_names):

	for layer_name in layer_names:
		layer_name = layer_name.split('.')[:-1]
		
		layer = model
		for steps in layer_name[:-1]:
                    layer = layer._modules[steps]

		conv3x3 = layer._modules[layer_name[-1]]
		conv1x1_1 = nn.Conv2d(in_channels = conv3x3.in_channels, out_channels = conv3x3.in_channels, kernel_size = (1,1), \
			padding = (0,0), bias = False).cuda()
		conv1x1_1.weight.data.copy_(torch.eye(conv3x3.in_channels).unsqueeze(2).unsqueeze(2))
		conv1x1_2 = nn.Conv2d(in_channels = conv3x3.out_channels, out_channels = conv3x3.out_channels, kernel_size = (1,1), \
			padding = (0,0), bias = False).cuda()
		conv1x1_2.weight.data.copy_(torch.eye(conv3x3.out_channels).unsqueeze(2).unsqueeze(2))
		layer._modules[layer_name[-1]] = nn.Sequential(conv1x1_1, conv3x3, conv1x1_2)

def get_flops(model):
	input = torch.randn(1, 3, 32, 32).cuda()
	temp_model = copy.deepcopy(model)
	flops, params = profile(temp_model, inputs=(input, ))
	return flops

def get_params(model):
    num_param=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        num_param += nn
    return num_param

def expected_flops(model, layer_names, num_params, num_flops, ratio):

	for layer_name in layer_names:
		layer_name = layer_name.split('.')[:-1]
		
		layer = model
		for steps in layer_name[:-1]:
			layer = layer._modules[steps]

		conv3x3 = layer._modules[layer_name[-1]]

		pruning_num_out = int(round(conv3x3.out_channels * ratio))
		#pruning_num_out = min(conv3x3.out_channels-1, pruning_num_out)

		pruning_num_in = int(round(conv3x3.in_channels * ratio))
		#pruning_num_in = min(conv3x3.in_channels-1, pruning_num_in)

		new_conv3x3 = nn.Conv2d(in_channels = conv3x3.in_channels - pruning_num_in, out_channels = conv3x3.out_channels - pruning_num_out, kernel_size = conv3x3.kernel_size, \
				stride = conv3x3.stride, padding = conv3x3.padding, bias = conv3x3.bias).cuda()
		
		conv1x1_1 = nn.Conv2d(in_channels = conv3x3.in_channels, out_channels = conv3x3.in_channels  - pruning_num_in, kernel_size = (1,1), \
			padding = (0,0), bias = False).cuda()
		conv1x1_2 = nn.Conv2d(in_channels = conv3x3.out_channels - pruning_num_out, out_channels = conv3x3.out_channels, kernel_size = (1,1), \
			padding = (0,0), bias = False).cuda()

		layer._modules[layer_name[-1]] = nn.Sequential(conv1x1_1, new_conv3x3, conv1x1_2)	

	num_params_after_prune = get_params(model)
	num_flops_after_prune = get_flops(model)

	print("Expected Param: {:2.2f}%  Flop: {:2.2f}%\n".format(num_params_after_prune/num_params*100, num_flops_after_prune/num_flops*100))
