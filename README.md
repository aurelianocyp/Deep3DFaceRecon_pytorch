For the original tensorflow implementation, check this [repo](https://github.com/microsoft/Deep3DFaceReconstruction).


## Installation

别用太高级的显卡，A5000不行

1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
conda env create -f environment.yml
source activate deep3d_pytorch
```

2. Install Nvdiffrast library:
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
```

3. Install Arcface Pytorch:
```
cd ..    # ./Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```
## Inference with a pre-trained model

### Prepare prerequisite models
1. download "01_MorphableModel.mat". Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── BFM
    │
    └─── 01_MorphableModel.mat
    │
    └─── Exp_Pca.bin
    |
    └─── ...
```
2. We provide a model trained on a combination of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 
[LFW](http://vis-www.cs.umass.edu/lfw/), [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
[IJB-A](https://www.nist.gov/programs-projects/face-challenges), [LS3D-W](https://www.adrianbulat.com/face-alignment), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. Download the pre-trained model using this [link (google drive)](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing) and organize the directory into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── mymodel
        │
        └─── epoch_20.pth

```

### Test with custom images
即此文件树种放置的是要拿来reconstruct的图片。他已经放好了一些参考图片，我们直接跑python test.py --name=<model_name>那个命令就行，在下面
To reconstruct 3d faces from test images, organize the test image folder as follows:
```
Deep3DFaceRecon_pytorch
│
└─── images
    │
    └─── *.jpg/*.png
    |
    └─── detections
        |
	└─── *.txt
```
The \*.jpg/\*.png files are test images. The \*.txt files are detected 5 facial landmarks with a shape of 5x2, and have the same name as the corresponding images. Check [./datasets/examples](datasets/examples) for a reference.

Then, run the test script:
```
# get reconstruction results of your custom images
python test.py --name=mymodel--epoch=20 --img_folder=images

# get reconstruction results of example images
python test.py --name=mymodel --epoch=20 --img_folder=./datasets/examples
```
当出现ninja: build stopped: subcommand failed.时代表没有OpenGL环境，可以使用--use_opengl False来避免这个报错
如果报没有detections文件夹，从next3d里的数据准备里面去获取
Results will be saved into ./checkpoints/<model_name>/results/<folder_to_test_images>, which contain the following files:

可能运行完成后过一会输出文件夹才能显示出来，等一下吧
| \*.png | A combination of cropped input image, reconstructed image, and visualization of projected landmarks.
|:----|:-----------|
| \*.obj | Reconstructed 3d face mesh with predicted color (texture+illumination) in the world coordinate space. Best viewed in Meshlab. |
| \*.mat | Predicted 257-dimensional coefficients and 68 projected 2d facial landmarks. Best viewed in Matlab.

## Training a model from scratch
### Prepare prerequisite models
1. We rely on [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to extract identity features for loss computation. Download the pre-trained model from Arcface using this [link](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3). By default, we use the resnet50 backbone ([ms1mv3_arcface_r50_fp16](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)), organize the download files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── recog_model
        │
        └─── ms1mv3_arcface_r50_fp16
	    |
	    └─── backbone.pth
```
2. We initialize R-Net using the weights trained on [ImageNet](https://image-net.org/). Download the weights provided by PyTorch using this [link](https://download.pytorch.org/models/resnet50-0676ba61.pth), and organize the file as the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── init_model
        │
        └─── resnet50-0676ba61.pth
```
3. We provide a landmark detector (tensorflow model) to extract 68 facial landmarks for loss computation. The detector is trained on [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [LFW](http://vis-www.cs.umass.edu/lfw/), and [LS3D-W](https://www.adrianbulat.com/face-alignment) datasets. Download the trained model using this [link (google drive)](https://drive.google.com/file/d/1Jl1yy2v7lIJLTRVIpgg2wvxYITI8Dkmw/view?usp=sharing) and organize the file as follows:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── lm_model
        │
        └─── 68lm_detector.pb
```
### Data preparation
1. To train a model with custom images，5 facial landmarks of each image are needed in advance for an image pre-alignment process. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn) to detect these landmarks. Then, organize all files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── datasets
    │
    └─── <folder_to_training_images>
        │
        └─── *.png/*.jpg
	|
	└─── detections
            |
	    └─── *.txt
```
The \*.txt files contain 5 facial landmarks with a shape of 5x2, and should have the same name with their corresponding images.

2. Generate 68 landmarks and skin attention mask for images using the following script:
```
# preprocess training images
python data_preparation.py --img_folder <folder_to_training_images>

# alternatively, you can preprocess multiple image folders simultaneously
python data_preparation.py --img_folder <folder_to_training_images1> <folder_to_training_images2> <folder_to_training_images3>

# preprocess validation images
python data_preparation.py --img_folder <folder_to_validation_images> --mode=val
```
The script will generate files of landmarks and skin masks, and save them into ./datasets/<folder_to_training_images>. In addition, it also generates a file containing the path of all training data into ./datalist which will then be used in the training script.



By default, the script uses a batchsize of 32 and will train the model with 20 epochs. For reference, the pre-trained model in this repo is trained with the default setting on a image collection of 300k images. A single iteration takes 0.8~0.9s on a single Tesla M40 GPU. The total training process takes around two days.

To use a trained model, see [Inference](https://github.com/sicxu/Deep3DFaceRecon_pytorch#inference-with-a-pre-trained-model) section.


