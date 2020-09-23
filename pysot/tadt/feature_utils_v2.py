
#### from image_loader import default_image_loader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize(tensor):
    return ((tensor.cpu().squeeze(0).numpy()+np.array([0.485,0.456,0.406]))*np.array([0.229,0.224,0.225])).transpose((1,2,0)).clip(0,1)

def get_subwindow_feature(model, image, location, input_sz, layer_name = None, visualize = False, name = None):
    """
    function: extracts the spetialed deep features of the input images with the given model
    args:
        model - deep network to extract features
        img - image to be processed
        location - the positions and sizes of the sub-windows, can contain several locations, [x1, y1, w, h]
        input_sz - the size of the image patch to be fed to model
        layer_name -
    """
    # 从原图image根据location(search_window位置信息)crop得到search_window，并将其resize成input_sz大小
    subwindow = get_subwindow(location, image, input_sz, visualize)
    visualize = False
    #if visualize:
    #    tensor_show(subwindow, 10, normalize = False)
    subwindow = (torch.unsqueeze(subwindow, 0)).to(device)
    features = model(subwindow, layer_name)
    if visualize:
        feature = torch.cat(features, dim = 1) # 将2组特征连接成 1 x 1024 x 45 x 45
        print('feature.shape:',feature.shape)
        heatmap = torch.sum(torch.squeeze(feature),dim = 0) # 1 x 1024 x 45 x 45 -> 1024 x 45 x 45 ->  45 x 45
        heatmap = heatmap/torch.max(heatmap)
        print('heatmap:\n',heatmap.shape)
        tensor_show(heatmap, 2, normalize = False, feature = True, name = name) # 将heatmap显示出来
    return features,subwindow

def get_subwindow(location, image, input_sz, visualize = False):
    """
    args:
        location - subwindow location [x1, y1, w, h]
        image - <class 'numpy.ndarray'>
        input_sz - the size of the input of vgg model, [width,height]
        visualize - whether to visualize the subwindow
    """
    size = np.array(location[2:4])
    position = np.array(location[0:2]) + size/2
    height, width = image.shape[0:2]


    x_index = (np.floor(position[0] + np.arange(1, size[0]+1) - np.ceil(size[0]/2))).astype(int)
    y_index = (np.floor(position[1] + np.arange(1, size[1]+1) - np.ceil(size[1]/2))).astype(int)

    #crop
    y_index = clamp(y_index, 0, height - 1)

    x_index = clamp(x_index, 0, width - 1)

    x_index, y_index = np.meshgrid(x_index, y_index)

    image = image[y_index, x_index, :]

    # 从原始image中crop出search_window
    image = cv2.resize(image, tuple(input_sz), interpolation=cv2.INTER_LINEAR)

    if visualize:

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print('show_image')
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input_image', image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


    #image transforms for matconvnet vgg model
    image = torch.tensor(image.transpose(2,0,1), dtype=torch.float)- 128 # 去中心化处理

    return image

def clamp(index, lower, upper):

    for idx in range(len(index)):
        if index[idx] > upper:
            index[idx] = upper
        if index[idx] < lower:
            index[idx] = lower
    return index
# 从CNN得到的features根据target_size得到target的features 保存到patch_features，把target的位置信息（corner形式）保存到patch_locations
def generate_patch_feature(target_size, srch_window_location, features):

    patch_features = []
    patch_locations = []

    for feature in features:
        feature_size = torch.tensor(feature.shape[-2:]).numpy()

        center = round_python2(feature_size/2)-1

        patch_size = np.floor(target_size * feature_size / srch_window_location[2:4] / 2) * 2 + 1

        patch_loc = np.append(center - np.floor(patch_size/2), center + np.floor(patch_size/2)).astype(int)

        patch_feature = feature[:,:,patch_loc[0]:patch_loc[2]+1,patch_loc[1]:patch_loc[3]+1]
        patch_features.append(patch_feature)
        patch_locations.append(patch_loc)
    visualize = False
    if visualize:
        patch_feature = torch.cat(patch_features, dim = 1) # 将2层特征(1 x 512 x 11 x 11)连接成(1 x 1024 x 11 x 11)
        print('patch_feature.shape:',patch_feature.shape)
        heatmap = torch.sum(torch.squeeze(patch_feature),dim = 0) # 将1024个特征压缩成1个特征来显示热图，这样做行么？？？？1 x 1024 x 11 x 11 -> 1024 x 11 x 11 -> 11 x 11
        heatmap = heatmap/torch.max(heatmap) # 归一化处理
        print('heatmap:\n',heatmap.shape)
        tensor_show(heatmap, 2, normalize = False, feature = True, name = 'patch_feature')
    return patch_features, patch_locations

# 逢5进1
def round_python2(temp_array):

    re_array = []
    for i in temp_array:
        if i % 1 == 0.5:
            re_array.append(np.ceil(i))
        else:
            re_array.append(np.round(i))
    return np.array(re_array)

def features_selection(input_features, feature_weights, mode = 'reduction'):

    assert(mode in ['pca', 'sa', 'pca_sa', 'reduction']), "mode need to be 'pca' or 'sa' or 'pca_sa' or 'reduction'"
    target_features = []
    num_channels = 0
    if mode == 'reduction':
        for i in range(len(input_features)):
            patch_feature = input_features[i]
            feature_weight = feature_weights[i]
            patch_feature[:, feature_weight < 1, :, :] = 0

            target_features.append(patch_feature)

            num_channels = num_channels + torch.sum(feature_weight)

    else:
        assert(mode == 'reduction'), "only 'reduction' is available for now!"
    return target_features

def feature_selection(input_feature, feature_weight, mode = 'reduction'):
    #TODO: pca, sa, pca_sa
    assert(mode in ['pca', 'sa', 'pca_sa', 'reduction']), "mode need to be 'pca' or 'sa' or 'pca_sa' or 'reduction'"
    return input_feature[:,feature_weight>0,:,:]



def resize_tensor(input_tensor, size, mode = 'bilinear',align_corners = False):
    """
    function: calculate the resize of torch.Tensor
    args:
        size: tuple(height,width)
    """
    # 使用nn.functional.interpolate对input_tensor进行插值处理，resize到size大小
    return nn.functional.interpolate(input_tensor, size, mode = mode, align_corners = align_corners)


def tensor_show(tensor, time = 20,bbox = None, normalize = True, feature = False, name = None):

    fig,ax = plt.subplots(1,1) # 创建 1 x 1 的窗口
    if normalize and not feature:
        ax.imshow(unnormalize(tensor))
    elif not normalize and not feature:
        ax.imshow(torch.squeeze(tensor).to('cpu').numpy().transpose((1,2,0)))
    elif feature:
        ax.imshow(torch.squeeze(tensor).to('cpu').numpy())
    if bbox is not None:
        ax.add_patch(Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],fill=False,color='g')) # 画矩形框
    plt.ion() # 打开交互式模式-可以同时打开多个窗口
    plt.title(str(name))
    mngr = plt.get_current_fig_manager()
    #mngr.window.setGeometry(100,100,800,500)
    str2 = './images/' + str(name) + '.jpg'
    plt.savefig(str2)
    plt.pause(time) # 显示time秒数
    plt.clf() # clear figure 清楚所有轴，但是窗口打开，这样他可以重复使用
    plt.close() # close a figure window
