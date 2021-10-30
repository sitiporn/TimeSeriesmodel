import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union, cast
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.module):
    
    def __init__(self,input_size:int,hidden_size:int,bias:bool=True):
        #params
        self.in_dim = input_size
        self.h_dim  = hidden_size
        self.num_layers = None  
        self.bias = None  
        self.batch_first = None  
        self.dropout = None  
        self.bidirectional = None   
        self.proj_size = None

        # (h,d)
        self.wf = torch.rand(self.h_dim,self.in_dim)
        self.wi = torch.rand(self.h_dim,self.in_dim)
        self.wo = torch.rand(self.h_dim,self.in_dim)
        self.wc = torch.rand(self.h_dim,self.in_dim)

 
        # (h,h)
        self.uf = torch.rand(self.h_dim,self.h_dim)
        self.ui = torch.rand(self.h_dim,self.h_dim)
        self.uo = torch.rand(self.h_dim,self.h_dim)
        self.uc = torch.rand(self.h_dim,self.h_dim)
        
        
        # (h,1)

        self.bf = torch.rand(self.h_dim,1) 
        self.bi = torch.rand(self.h_dim,1) 
        self.bo = torch.rand(self.h_dim,1) 
        self.bc = torch.rand(self.h_dim,1) 
    
        self.sigmoid = nn.Sigmoid() 
        self.tanh =  torch.tanh()        

   
   def foward(self,input:Tensor,hx:Optional[Tuple[Tensor,Tensor]] = None) -> Turple[Tensor,Tensor]
       


       # Todo: 
       """
       ft = sigmoid(wt*xt + uf *h_prev + b_f)
       it = sigmoid(wi*xt + ui *h_prev + b_i)
       ot = sigmoid(wo*xt + uo *h_prev + b_o)
       c_hat = tanh(wc*xt + uc * h_prev + bc)
       ct = ft * ct_prev + it + c_hat
       ht = ot * tanh(ct) 
       
       xt shape: (seq_len,batch_size,input_size)  
       ot shape: (seq_len,

       LSTM(input_dim,out_dim)
      
       hx : [h0,c0] 


       """
       h_prev = h0
       c_prev = c0

       for t in range(seq_len):
           xt = input[t,:,:] 
           
           # (h,batch_size) < -  (h,d) (d,batch_size) +... + (h,1)
           it = self.sigmoid((self.wi @ xt.T) + (self.ui @ h_prev) + self.b_i)
           ft = self.sigmoid((self.wf @ xt.T) + (self.uf @ h_prev) + self.b_f)
           ot = self.sigmoid((self.wo @ xt.T) + (self.uo @ h_prev) + self.b_o) 
           c_hat = self.tanh((self.wc @ xt.T) + (self.uc @ h_prev) + self.b_c) 
           ct =  (ft * c_prev) + (it* ) 
           #it = (Wi @ ) + (Ui @ h_prev) + b_i
