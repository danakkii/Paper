import numpy as np
from scipy.io import loadmat 
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Torchdata
from torchsummary import summary

# feature extraction
class input_conv_layer(nn.Module): # k(oc, k, k, ic) = 16*3*3*nbands, s =1
       def __init__(self, ic, oc, k, s):
              super(input_conv_layer, self).__init__()
               
              self.conv1 = torch.nn.Conv2d(ic, oc, kernel_size = k, stride = s, bias=True)
           
       def forward(self, x):
              out = self.conv1(x)
              return out

class frist_dense_block(nn.Module): # k(oc, k, k, ic), ic= (16+(n-1)g), n = 6, s = 1, relu, dropout=10%
       def __init__(self, oc, oc2, k, s, oc3, k2, g): 
              super(frist_dense_block, self).__init__() 
              
              self.layer1 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer2 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc+g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc+g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer3 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc+2*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc+2*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer4 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc+3*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc+3*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer5 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc+4*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc+4*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer6 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(oc+5*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc+5*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              
       def forward(self, x):
              out = self.layer1(x)
              out = torch.cat([x, out], 1) # concatenation 
              out2 = self.layer2(out) 
              out = torch.cat([out,out2], 1)
              out2 = self.layer3(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer4(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer5(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer6(out) 
              out = torch.cat([out, out2], 1)
              return out
              

class transition_layer(nn.Module): # k = 1, k = 16+6, s=1, relu, dropout=10%
       def __init__(self, t_ic, t_oc, k2, s):
              super(transition_layer, self).__init__()
              
              self.layer1 = torch.nn.Sequential( 
                     torch.nn.BatchNorm2d(t_ic),
                     torch.nn.ReLU(),
                     torch.nn.Conv2d(t_ic, t_oc, kernel_size=k2, stride=s),
                     torch.nn.Dropout2d(p=0.1),
                     torch.nn.AvgPool2d(2, stride=2) 
              )

       def forward(self, x):
              out = self.layer1(x)
              return out
       
       
class second_dense_block(nn.Module): # relu, global average pooling(output 1*1*k), k= k/2+16*g, fully connected of nclasses layers with softmax
       def __init__(self, t_oc, k, s, oc2, oc3, k2, g): 
              super(second_dense_block, self).__init__() 

              self.layer1 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer2 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer3 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+2*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+2*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer4 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+3*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+3*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer5 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+4*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+4*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer6 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+5*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+5*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer7 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+6*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+6*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer8 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+7*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+7*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer9 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+8*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+8*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer10 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+9*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+9*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer11 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+10*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+10*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer12 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+11*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+11*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer13 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+12*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+12*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer14 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+13*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+13*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer15 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+14*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+14*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
              
              self.layer16 = torch.nn.Sequential( 
              torch.nn.BatchNorm2d(t_oc+15*g), #k1
              torch.nn.ReLU(),
              torch.nn.Conv2d(t_oc+15*g, oc2, kernel_size=k2, stride=s), 
              torch.nn.Dropout2d(p=0.1),
              torch.nn.BatchNorm2d(oc2), #k2
              torch.nn.ReLU(),
              torch.nn.Conv2d(oc2, oc3, kernel_size=k, stride=s, padding=1), 
              torch.nn.Dropout2d(p=0.1))
    
       def forward(self, x): 
              out = self.layer1(x) 
              out = torch.cat([x, out], 1)
              out2 = self.layer2(out) 
              out = torch.cat([out,out2], 1)
              out2 = self.layer3(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer4(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer5(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer6(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer7(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer8(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer9(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer10(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer11(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer12(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer13(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer14(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer15(out) 
              out = torch.cat([out, out2], 1)
              out2 = self.layer16(out) 
              out = torch.cat([out, out2], 1)
              return out


class classification_layer(nn.Module): # relu, global_average_pooling with output(1*1*k3), fully connected of n layers with softmax

       def __init__(self, c_oc):
              super(classification_layer, self).__init__()
              
              self.layer1 = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(c_oc),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(c_oc, 2, bias=True),
              #       torch.nn.Softmax()                     
              )
         
       def forward(self, x):
              out = self.layer1(x)
              return out

class DDCNN(nn.Module):
       def __init__(self, ic, oc, oc2, k, s, oc3, k2, g, t_ic, t_oc, c_oc):
              super(DDCNN, self).__init__()
              
              self.conv1 = input_conv_layer(ic, oc, k, s) 
              self.dense1 = frist_dense_block(oc,oc2,k,s,oc3,k2,g)
              self.trans1 = transition_layer(t_ic,t_oc, k2, s)
              self.dense2 = second_dense_block(t_oc, k, s, oc2, oc3, k2, g)
              self.classification1 = classification_layer(c_oc)
              
       def forward(self, x):
              out = self.conv1(x)
              out = self.trans1(self.dense1(out))
              out = self.dense2(out)
              out = self.classification1(out)
              return out


if __name__ =="__main__":
       ic = 200 
       oc = 16
       g = 32
       oc2 = 128
       k = 3 
       s = 1
       oc3 = 32
       k2 = 1
       t_ic = oc+6*g
       t_oc = int(t_ic/2)
       c_oc = t_oc  + 16*g
       
       model = DDCNN(ic, oc, oc2, k, s, oc3, k2, g, t_ic, t_oc, c_oc)
       summary(model, (ic, 11, 11))