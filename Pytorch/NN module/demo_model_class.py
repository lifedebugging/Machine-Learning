#Create model class
import torch
import torch.nn as nn

class Model(nn.Module):    #clss needs to inherit  from nn.Module class to use its functionality

    def __init__(self,num_features):
        super().__init__()   #inherit form parent class, nn.module ke constructor ko invoke kar rahe
        self.linear = nn.Linear(num_features,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,features):
        out = self.linear(features)
        out = self.sigmoid(out)

        return out 
    
