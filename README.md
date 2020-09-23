# Ta-SiamRPN++

This is the repo for paper "Tracking Algorithm for Siamese Network Based on Target-Aware Feature Selection" [[paper](http://www.opticsjournal.net/Articles/abstract?aid=OJ9e66208a818c6dbf)].

## Introduction

Tracking algorithms implemented in Siamese networks utilize an offline training network to extract features from a target object for matching and tracking. The offline-trained deep features are less efficient for distinguishing targets with arbitrary forms from the background. Therefore, we proposed a tracking algorithm for a Siamese network based on target-aware feature selection. First, the cropped template and detection frames were sent to a feature extraction network based on ResNet50 to extract the shallow, middle and deep features of the target and search regions. Second, in the target-aware module, a regression loss function was formulated for target-aware features and an importance scale for each convolution kernel was obtained based on backpropagated gradients. Then, the convolution kernels with large importance scales were activated to select target-aware features. Finally, the selected features were inputted into the SiamRPN module for target-background classification and the bounding box regression was applied to obtain an accurate bounding box of the target. Results of experiments on OTB2015 and VOT2018 datasets confirm that the proposed algorithm can achieve robust tracking of the target.

## Environment

This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 1.1, CUDA 10, RTX 1080Ti GPU

## Model Zoo

The corresponding offline-trained models are availabe at [PySOT Model Zoo](MODEL_ZOO.md).


## Get Started

### Installation

 - Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).
 - Add DROL to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/ta-siamrpn++:$PYTHONPATH
```

### Download models

Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth to the corresponding directory in experiment.

### Test tracker

```bash
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

### Eval tracker

assume still in experiments/siamrpn_r50_l234_dwxcorr_8gpu
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

## Others

 - Since this repo is a grown-up modification of [PySOT](https://github.com/STVIR/pysot), we recommend to refer to PySOT for more technical issues.


## References

- Chen Zhiwang,Zhang Zhongxin,Song Juan,Luo Hongfu,Peng Yong. [Tracking Algorithm for Siamese Network Based on Target-Aware Feature Selection](http://www.opticsjournal.net/Articles/abstract?aid=OJ9e66208a818c6dbf). Acta Optica Sinica, 2020, 40(9): 0915003
- Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan.[SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks](https://arxiv.org/abs/1812.11703). IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- Xin Li, Chao Ma, Baoyuan Wu, Zhenyu He, Ming-Hsuan Yang.[Target-Aware Deep Tracking](https://arxiv.org/abs/1904.01772). IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

## Ackowledgement

- [pysot](https://github.com/STVIR/pysot)
- [TADT](https://github.com/ZikunZhou/TADT-python)

## Contact

Email: ZZXin00016@163.com


















