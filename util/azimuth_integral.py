import numpy as np
import torch


def GetPSD1D(psd2D):
    psd2D = 0.5*psd2D + 0.5
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

def crossEntropyFourierAzimuth(tensor1,tensor2):

    '''
    you must rewrite your own crossEntropyLoss since
    the pytorch version of crossEntropyLoss is
    (-p(x)*log(q(x))).sum()
    but the crossEntropyLoss applied in this paper is
    (-p(x)*log(q(x))-(1-p(x))*log(1-q(x))).sum()
    for the crossEntropyLoss, the sequence of tensor1
    and tensor2 cannot be changed
    '''
    loss = torch.relu(((tensor1-tensor2)*(tensor1-tensor2)).sum()-0.25)
    return loss        

def powerloss(tensor,upper,bottom):
    loss = ((tensor-upper)*torch.relu(tensor-upper)).sum()+((bottom-tensor)*torch.relu(bottom-tensor)).sum()
    return loss