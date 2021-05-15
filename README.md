# tfvaegan
Custom AwA2 semantic attribute training for tfvaegan

This project is based on [Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification (ECCV 2020)](https://github.com/akshitac8/tfvaegan). I apply some custom modification on top of it.

## Installation
```shell
conda update conda
conda create --name tfvaegan python=3.6
conda activate tfvaegan

pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install h5py sklearn pandas

git clone https://github.com/e96031413/tfvaegan
cd tfvaegan/zero-shot-images/data/

wget  http://datasets.d2.mpi-inf.mpg.de/xian/cvpr18xian.zip
unzip cvpr18xian.zip && rm cvpr18xian.zip

wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
unzip xlsa17.zip && rm xlsa17.zip

move all the folders inside {cvpr18xian/data/*} and {xlsa17/data/*} to tfvaegan/zero-shot-images/data/
```

## Zero-Shot Image Classification

make sure you have execute ```conda activate tfvaegan```

```shell
cd zero-shot-images
python image-scripts/run_cub_tfvaegan.py
python image_scripts/run_awa_tfvaegan.py   # There are 5 different version of semantic attributes that can be trained with this script.
python image_scripts/run_flo_tfvaegan.py
python image_scripts/run_sun_tfvaegan.py
```

## 5 different version of semantic attributes for AWA2 dataset

**att:** continuous value ( 0.1, 0.3, 0.7..... 0.9) ( 50 class x 85 attribute)

**binaryAtt:** binary value (0 or 1)  ( 50 class x 85 attribute)

For the other 3 attributes, I use their class information only. I encode the classes info to binary, bit, and one-hot, respectively.( 50 class )

**bit_encoding:** 0,0,0,0,0 for class 1, 0,0,0,0,1 for class 2, .......... 1,0,0,0,2 for class 50 ( 50 class x 5 attribute)

**label_encoding:** 1 for class 1, 2 for class 2,.............. 50 for class 50 ( 50 class x 1 attribute)

**one_hot_encoding:** Since there are 50 classes in AWA2 dataset, each class can be represented by 50 different numbers. ( 50 class x 50 attribute)
(1,0,0,0,0,0,0..........0 for class 1, and
0,1,0,0,0,0,0,0,0............0 for class 2, etc.)

You can view the encoding file in .mat and .csv file [here](https://github.com/e96031413/tfvaegan/tree/main/zero-shot-images/data/AWA2).

Note: When training the model, we use .mat file only. ( csv file is only for human to understand the encoding structure.)

## How to create your own semantic attribute?
You can use the notebook [here](https://github.com/e96031413/tfvaegan/blob/main/zero-shot-images/data/AWA2/awa_create_custom_attribute.ipynb) to create the attribute.

* Step 1
Define your attribute with pandas (For example: one-hot encoding or label encoding)
* Step 2
Export the dataframe to csv file
* Step 3
Load the csv file with Octave or Matlab to check the structure of the csv file, the structure should look similar to the one provided by the dataset. Modify it manually if you find the structure wrong.
* Step 4
Save the opened csv file as .mat file.

Supposed that you have followed my step1 and step 2, and open the octave:
In the terminal:
```matlab
csvread('fileName.csv')

% manually rename the ans variable with att variable in the bottom left panel.

att(2:end,:)  % select all except the first row according to your structure

% save att as mat file
att = [ 1:1; 2:2; 3:3; ............50:50];

save myfile.mat att -v7   % use flag -v7 for scipy.io mat file compatibility.

% Now you can load myfile.mat as your custom attribute.
```
For custom attribute code, please see line 9 in [zero-shot-images/config.py](https://github.com/e96031413/tfvaegan/blob/main/zero-shot-images/config.py#L9), and line 40-54 in [zero-shot-images/util.py](https://github.com/e96031413/tfvaegan/blob/main/zero-shot-images/util.py#L40-L54).

## How to create your own visual feature?
Use [resnet101_feature_extractor.py](https://github.com/e96031413/tfvaegan/blob/main/zero-shot-images/resnet101_feature_extractor.py) to extract the image.

The script is used for my own dataset only, if you want to apply this code to your own dataset, just modify the loading images part.

## Experiment result

The experiment result shows that among these 5 types of attribute, the provided continuous and binary attributes work smoothly.

However, when it comes to the self-made attribute, only one_hot_encoding attribute can be successfully trained with the corresponding attribute size(50). But the performance is not good. Thus I only provide attribute methods that work in the following table.

|                 | binaryATT | continuous | one-hot |
|:---------------:|:---------:|:----------:|---------|
|  ZSL Unseen ACC |   62.5%   |    71.4%   | 17.9%   |
|  GZSL Seen ACC  |   76.2%   |    75.4%   | 78.8%   |
| GZSL Unseen ACC |   47.4%   |    59.0%   | 5.5%    |
|      GZSH H     |   58.5%   |    66.3%   | 10.4%   |

**I assume that the reason why bit_encoding and label_encoding can't be trained is that both of them are with small attribute size.(5 and 1, respectively)**
**For one_hot_encoding, it is 50 and can be trained successfully.**


## Citation:
```
@inproceedings{narayan2020latent,
	title={Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification},
	author={Narayan, Sanath and Gupta, Akshita and Khan, Fahad Shahbaz and Snoek, Cees GM and Shao, Ling},
	booktitle={ECCV},
	year={2020}
}
```
