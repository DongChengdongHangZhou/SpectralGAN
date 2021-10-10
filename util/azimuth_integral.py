import numpy as np
import torch


def GetPSD1D(psd2D):
    psd2D = 0.5*psd2D + 0.5
    psd2D[:,:,128,128] = 1   # we must ensure the center of the spectrum is 1, to stablize the training.
    batch_size = psd2D.shape[0]
    psd2D = torch.exp(psd2D*16)-1
    Y, X = np.ogrid[0:256, 0:256]
    r    = np.hypot(X - 128, Y - 128).astype(np.int)
    r = torch.from_numpy(r).cuda(0)
    result = torch.zeros((batch_size,128)).cuda(0)
    for i in range(batch_size):
        for j in range(128):
            get = torch.eq(r,j)
            result[i][j] = (psd2D[i]*get).sum()
    psd1D = (result.t()/result.t()[0]).t().cuda(0) # every element is devided by the value of the first column. result[i][j]/result[i][0]
    return torch.reshape(psd1D,(batch_size,2,8,8))      

def Azimuthloss(tensor1,tensor2):
    loss = torch.relu(((tensor1-tensor2)*(tensor1-tensor2)).sum())*200
    return loss        

def powerloss(tensor1,tensor2):
    loss = ((tensor1-tensor2)*(tensor1-tensor2)).sum()
    return loss

def spectralloss(tensor_real,tensor_fake):
    loss =(-1/128) * ((tensor_real*torch.log(tensor_fake)).sum() + ((1.01-tensor_real)*torch.log(1.01-tensor_fake)).sum())
    return loss
