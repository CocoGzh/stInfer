# stInfer
stInfer: Spatially Transcription Expression Inference from H&E Histology Images 

Background: 
The rapid development of spatial transcriptomics (ST) technology has enabled the measurement of transcript abundance while obtaining the spatial location of cells. However, the high cost has hindered the implementation of spatial omics in large-scale organizations. In addition, widely used ST platforms, such as Visium, can only measure the spots expression in low-resolution, which limits their usefulness in studying detailed structures. 
Results: 
Here, we propose stInfer, a method that can infer gene expression and enhance resolution from Hematoxylin and Eosin-stained (H&E) histology images. Specifically, the spot in an H&E image can be segmented into patches. Then a pre-trained visual model is applied to encode these patch images into feature vectors. Finally, the expression of the target spot can be predicted by nearest neighbor weighting. Comprehensive evaluation on breast cancer datasets demonstrates the effectiveness of the proposed method.
Conclusions: 
In summary, stInfer is a powerful tool that can infer gene expression and improve spatial resolution from H&E images. It holds great promise for being widely applied to complex ST data to bring novel insights into structural compositions and microenvironments.

<img src="Figure 1.jpg" width="800px"></img>

## Getting started
### data preparation

Please put all data in the data folder.

The Old ST dataset can be obtained from https://github.com/almaan/her2st. 

10X ST dataset can be obtained from GSE195665. 

Marker genes have been saved in the data folder.


### model preparation

Please put all model checkpoint in the timm_model folder.

vgg16:
https://huggingface.co/timm/vgg16.tv_in1k/resolve/main/pytorch_model.bin 

densenet121
https://huggingface.co/timm/densenet121.ra_in1k/resolve/main/pytorch_model.bin 

resnet18
https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth

resnet50
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth

resnext50d
https://huggingface.co/timm/resnext50d_32x4d.bt_in1k/resolve/main/pytorch_model.bin


### a simple tutorial
It can be seen in 1OldST.ipynb

## Contact

guozhenhao17@mails.ucas.ac.cn

guozhenhao@tongji.edu.cn

chenzhanheng@mails.ucas.ac.cn

## Citation

TODO
