 ---

<div align="center">    
 
# Adaptive feature selection-based feature reconstruction network for few-shot learning

Jie Ren, Yaohui An, Tao Lei, Junpo Yang, Wenyue Zhang, Zicheng Pan, Yi Liao, Yongsheng Gao, Changming Sun, Weichuan Zhang

</div>

## Abstract
Few-shot learning (FSL) aims to accurately classify samples of different categories using extremely limited training data. In this work, we thoroughly analyze the fact that existing FSL methods ignore the differences, even significant differences, in feature representations extracted by different base backbones from each input image, which in turn affect classification performance. Therefore, a novel automatic feature selection (AFS) module is designed which has the capability to consistently obtain high-quality feature representations from each input image across different datasets and integrate quantized local and global features extracted from different base backbones using adaptive weights. Furthermore, the designed AFS module has the capability to effectively highlight the target feature information, suppress the influence of background noise, and improve the quality of feature representations. Then a novel AFS-based feature reconstruction AFS-FR network is proposed for performing different FSL tasks. Extensive experiments conducted on five benchmark datasets (i.e., CUB200-2011, Stanford Dog, Mini-ImageNet, Tiered-ImageNet, and Aircraft) demonstrate the effectiveness and superiority of the proposed AFS-FR method over state-of-the-art approaches. Especially in the Tiered-ImageNet dataset, the classification accuracies of the proposed AFS-FR method under the 5-way 1-shot and 5-way 5-shot experimental settings are 86.30±0.13 and 94.84±0.06 respectively, which achieve about 4% and 5% improvement over the best performance indicators in the comparison methods respectively.


## Datasets/Pre-trained Model Preparation Guidelines

Follow the guidelines provided in the [FRN](https://github.com/Tsingularity/FRN) to download and prepare these datasets. The ViT pre-trained model (vit-base-patch16-224-in21k.pth) can be downloaded from [here](https://huggingface.co/google/vit-base-patch16-224-in21k).

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [FRN](https://github.com/Tsingularity/FRN)

