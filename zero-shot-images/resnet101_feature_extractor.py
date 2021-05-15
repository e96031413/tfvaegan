import torch
import torchvision
import torchvision.models as models
import json
import pandas as pd
import numpy as np
import torch.nn as nn
import scipy.io as io
from torch.utils.data import Dataset
from torch.utils import data
import os
from PIL import Image
from tqdm import tqdm

# data_dir = "./test"
data_dir = "../XXison/datasets_for_ma/"

dataset_info = json.load(open("../XXison/dataset.json", "r"))
df = pd.DataFrame.from_dict(dataset_info, orient="index")
df['file_name'] = df.index
df["file_name"] = data_dir + df["file_name"].astype(str)   #新增圖片完整路徑(full path)

orig_df = pd.DataFrame.from_dict(dataset_info, orient="index") #載入只有圖片檔名的data(後續可複製一份df代替，就不用再次載入)
orig_df['file_name'] = orig_df.index              

print("Load dataset.json successfully!!!")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

def get_vector(image):
    # Create a PyTorch tensor with the transformed image
    t_img = transforms(image)
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 2048
    my_embedding = torch.zeros(2048)

    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    with torch.no_grad():                               # <-- no_grad context
        model(t_img.unsqueeze(0).cuda())                       # <-- unsqueeze
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding.cuda()

model = models.resnet101(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()


if torch.cuda.is_available():
    gpus = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if len(device_ids)>1:
        model = nn.DataParallel(model, device_ids = device_ids).cuda()
    else:
        model = model.cuda()
else:
    print("Use CPU only, check your environment!")

feat = {}
feature_list = []
file_path_list = []
label_list = []

files = df["file_name"].tolist()          #之前用os.walk去訪歷資料夾，但是百萬張圖片的速度太慢，透過之前預先建好的json檔案，直接取得路徑

for file_path in tqdm(files[0:10]):                    
    if (is_image(file_path)):             #假如路徑是圖片
        image = Image.open(file_path)     #開圖檔
        pic_vector = get_vector(image)    #用ResNet101萃取圖片的特徵向量
#         print(pic_vector.is_cuda)       #檢查是否用GPU
        file_path_list.append(file_path)                 #把該張圖片的絕對路徑加到list保存
        feature_list.append(pic_vector.data.tolist())    #把該張圖片萃取的特徵加到list保存      
        
        #如果當前萃取之圖片的檔名與原始json檔中的檔名相符，取得該檔名的class值，並加到list保存
        label_list.append(orig_df.loc[orig_df['file_name'] == file_path.split("/")[-1] ]['class'].values[0])       
        
for index, value in enumerate(label_list):
    if value == 'good':
        label_list[index] = 1
    elif value == 'missing':
        label_list[index] = 2
    elif value == 'shift':
        label_list[index] = 3
    elif value == 'stand':
        label_list[index] = 4
    elif value == 'broke':
        label_list[index] = 5
    elif value == 'short':
        label_list[index] = 6

# https://stackoverflow.com/a/11885718/13369757 (單維度dim轉置)
# https://stackoverflow.com/a/7464816 (處理char字串呈現問題)

feat['features'] = np.array(feature_list).astype('float32').T
feat['image_files'] = np.array(file_path_list, dtype=np.object)[None].T
feat['labels'] = np.array(label_list).astype('float32')[None].T


io.savemat('res101.mat',feat)
