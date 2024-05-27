import numpy as np
from scipy.io import loadmat 
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2



# dataset       
class Dataset(torch.utils.data.Dataset):  
       def __init__(self, data, label, indice, std, patch_size=11):
              #super(data_loader, self).__init__()
              
              self.data = data
              self.label = label
              self.patch_size = patch_size
              self.indice = indice
              self.std = std

              
              
       def __len__(self):
              return len(self.indice) # location길이 만큼 
       
       
       def __getitem__(self, i):
              # center pixel
              # print(f"i={i}")
              z, x, y = self.indice[i] # 중간 output = 학습 x, y좌표
              # print(f"self.indice={np.shape(self.indice)}")
              x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
              x2, y2 = x1 + self.patch_size, y1 + self.patch_size
              
              # patch, 최종 output
              patch = self.data[z, x1:x2, y1:y2]
              label = self.label[z,x,y]

              #Add Gaussian noise with predefined std
              patch += self.std * np.random.normal(0, 1, size=patch.shape)

              patch = np.asarray(np.copy(patch).transpose((2,0,1)), dtype="float32")
              label = np.asarray(np.copy(label), dtype="int64") 
              # print(f'patch_data = {patch.shape}, center_label = {np.shape(label)}')
              # print(f"patch={patch}, center_label={label}")

              return patch, label-2

def set_datasets(data, label, patch_size=11): 
       #bk elimination
       hp=patch_size // 2
       indices = np.where(label!=0) # (3, 60931), 각 인덱스마다 3차원 좌표
       indices = [(x,y,z) for x,y,z in zip(*indices)] # (25324, 3), 60931개가 각각 3차원으로 구성
       
       valid_indices = []
       for indice in indices:
               z, x, y = indice 
               x_start, x_end = x - patch_size // 2 , x + patch_size // 2
               y_start, y_end = y - patch_size // 2 , y + patch_size // 2
               
               # 범위 설정
               if x_start >= 0 and x_end < data[z].shape[0] and y_start >= 0 and y_end < data[z].shape[1]:
                     valid_indices.append(indice)
       
       return valid_indices


def calibration(data, dr, wr):
        dr = np.mean(dr, axis=0)
        wr = np.mean(wr, axis=0)
        calibration = (data - dr) / (wr - dr)
        return calibration

# def collate_fn(batchDummy): 
#        x = [torch.LongTensor(batch['x'])for batch in batchDummy] 
#        x = torch.nn.utils.rnn.pad_sequence(x, batch_first = True) 
#        return {'x' : x}