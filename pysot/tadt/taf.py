#20190513 by zikun
import numpy as np
import torch
from pysot.tadt.taf_net import Regress_Net
import math
import torch.nn as nn
from torch.optim import SGD


torch.backends.cudnn.benchmark=True

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def taf_model(features, filter_sizes, device):
    '''
    function: select target aware feature
    args:
        filter_sizes - [batch, channel, height, width]
    '''
    num_feaure_groups = len(features) # 共几组特征
    feature_weights = []
    channel_num = [300, 1024, 1500] # 要提取的各个layer的channel_num
    print('channel_num', channel_num)
    nz_num = 0
    nz_num_min = 256

    for i in range(num_feaure_groups):
        #对layer4的第一帧不用训练 因为其全部特征要保留
        if i == 0 or i == 2 or i == 1:
            feature, filter_size = features[i], filter_sizes[i]
            reg = Regress_Net(filter_size).to(device)
            feature_size = torch.tensor(feature.shape).numpy()

            output_sigma = filter_size[-2:]* 0.1
            gauss_label = generate_gauss_label(feature_size[-2:], output_sigma).to(device) # 得到高斯标签

            if i == 0:
                max_epochs = 300
                lr_set = 1e-3
                momentum_set = 0.98
                weight_decay_set = 0.0001
            elif i == 1:
                max_epochs = 300
                lr_set = 1e-3
                momentum_set = 0.93
                weight_decay_set = 0.0001
            else:
                max_epochs = 300
                lr_set = 4.92e-5
                momentum_set = 0.9
                weight_decay_set = 0.0001
            # first train the network with mse_loss
            objective = nn.MSELoss()
            # optim = SGD(reg.parameters(),lr = 5e-7,momentum = 0.9,weight_decay = 0.0005)
            optim = SGD(reg.parameters(), lr=lr_set, momentum=momentum_set, weight_decay=weight_decay_set)  # 设定优化器
            train_reg(reg, optim, feature, objective, gauss_label, device, max_epochs) # 训练mse_loss
            reg_weights = reg.conv.weight.data

            weight = torch.sum(reg_weights, dim = (0,2,3)) # 对得到的卷积核梯度信息进行全局平均池化

            # The value ot the parameters equals to the sum of the gradients in all BP processes.
            # And we found that using the converged parameters is more stable
            sorted_cap, indices = torch.sort(torch.sum(reg_weights, dim = (0,2,3)),descending = True) # 排序

            feature_weight = torch.zeros(len(indices))
            feature_weight[indices[sorted_cap > 0]] = 1 # 将权重大于0的feature_weight置于1
            feature_weight[indices[channel_num[i]:]] = 0
            ###print("target active channel number:",sum(feature_weight))
            feature_weight[indices[:channel_num[i]]] = 1  # 新添加-对layer2/3来说只提取256个channel，对layer4来说全部提取
            nz_num = nz_num + torch.sum(feature_weight)

            # In case，there　are two less features, we set a minmum feature number.
            # If the total number is less than the minimum number, then select more from conv4_3
            if i == 1 and nz_num < nz_num_min:
                added_indices = indices[torch.sum(feature_weight).to(torch.long): (
                            torch.sum(feature_weight) + nz_num_min - nz_num).to(torch.long)]
                feature_weight[added_indices] = 1

        else:
            feature_weight = torch.full((1024,1), 1) # layer4层输出的通道数
            feature_weight = torch.squeeze(feature_weight)
        #print("Num_selected_feature:",torch.sum(feature_weight))
        # we perform scale sensitive feature selection on the conv41 feaure, as it retains more spatial information
        '''
        if i == 0:
            temp_feature_weight = taf_rank_model(feature, filter_size, device)
            feature_weight = feature_weight * temp_feature_weight # 一共139个channel
        '''
        feature_weights.append(feature_weight)
    return feature_weights


def train_reg(model, optim, input, objective, gauss_label, device, epochs = 100):
    """
    function: train the regression net and regression loss
    """
    for i in range(epochs):
        input = input.to(device)
        predict = model(input).view(1,-1) # 注意它加了padding保证输出和输入尺寸一致，均为45 x 45

        gauss_label = gauss_label.view(1,-1) # 将高斯标签一维化处理
        loss = objective(predict, gauss_label)+ 0.0001 * torch.sum(model.conv.weight.data**2)  # 求取MSELOSS
        ####print('loss',i,':',loss)
        if loss<0.0001:
            break

        if hasattr(optim,'module'):
            optim.module.zero_grad()
            loss.backward()
            optim.module.step()
        else:
            optim.zero_grad() # 将梯度信息清零
            loss.backward(retain_graph=True) # 执行反向传播
            optim.step() # 更新参数信息


# 生成高斯标签
def generate_gauss_label(size, sigma, center = (0, 0), end_pad=(0, 0)):
    """
    function: generate gauss label for L2 loss
    """
    shift_x = torch.arange(-(size[1] - 1) / 2, (size[1] + 1) / 2 + end_pad[1])
    shift_y = torch.arange(-(size[0] - 1) / 2, (size[0] + 1) / 2 + end_pad[0])

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

    alpha = 0.2
    gauss_label = torch.exp(-1*alpha*(
                        (shift_y-center[0])**2/(sigma[0]**2) +
                        (shift_x-center[1])**2/(sigma[1]**2)
                        ))

    return gauss_label



if __name__ == '__main__':
    from feature_utils_v2 import resize_tensor

    input = torch.rand(1, 1, 3, 3)
    print(input)
    resized_input = resize_tensor(input, [7, 7])
    print(resized_input)
