import torch.nn as nn
import kornia
import torch

## Modified by Alejandro Salgueiro Dorado for a Master Thesis Project, Wageningen University, 24.11.2023


def MSEloss(pred_y,batch_y):
    return nn.MSELoss(pred_y,batch_y)
    
def MAELoss(pred_y,batch_y):
    return nn.L1Loss(pred_y,batch_y)

def WeightLoss(pred_y,batch_y):
    """
    Inspired by the weighted MSE loss for radar reflectivity from:
    Wu, D.; Wu, L.; Zhang, T.; Zhang, W.; Huang, J.; Wang, X. Short-Term Rainfall Prediction Based on Radar Echo Using an Improved Self-Attention PredRNN Deep Learning Model. Atmosphere 2022, 13, 1963. https://doi.org/10.3390/atmos13121963
    """
    if pred_y <= 0.285:
        weight = 1
    elif 0.285 < pred_y <= 0.428:
        weight = 2
    elif pred_y > 0.428:
        weight = 3

    loss = weight * nn.MSELoss(pred_y,batch_y)
    return loss

class WeightLoss():
    """
    Inspired by the weighted MSE loss for radar reflectivity from:
    Wu, D.; Wu, L.; Zhang, T.; Zhang, W.; Huang, J.; Wang, X. Short-Term Rainfall Prediction Based on Radar Echo Using an Improved Self-Attention PredRNN Deep Learning Model. Atmosphere 2022, 13, 1963. https://doi.org/10.3390/atmos13121963
    """
    def __init__(self,thresh1 = 0.285, thresh2 = 0.428):
        self.thresh1 = thresh1
        self.thresh2 = thresh2


    def __call__(self,pred_y,batch_y):

        if pred_y <= self.thresh1:
            weight = 1
        elif self.thresh1 < pred_y <= self.thresh2:
            weight = 2
        elif pred_y > self.thresh2:
            weight = 3

        loss = weight * nn.MSELoss(pred_y,batch_y)
        return loss


class MSCELoss():
    """
    Inspired by the edge preserving MSE loss from:
    Pandey, R.K., Saha, N., Karmakar, S., Ramakrishnan, A.G. (2018). MSCE: An Edge-Preserving Robust Loss Function for Improving Super-Resolution Algorithms. In: Cheng, L., Leung, A., Ozawa, S. (eds) Neural Information Processing. ICONIP 2018. Lecture Notes in Computer Science(), vol 11306. Springer, Cham. https://doi.org/10.1007/978-3-030-04224-0_49 
    """

    def __call__(self,pred_y,batch_y):
        
        loss = nn.MSELoss()
        
        dim = pred_y.shape[3]
        dim2 = pred_y.shape[4]

        input = torch.empty((0,12,1,dim,dim2))
        input = input.to(device="cuda")
        target = torch.empty((0,12,1,dim,dim2))
        target = target.to(device="cuda")

        # Loop to concatenate the edge pictures by sequences of 12
        for i in range(pred_y.shape[0]):
            edge_input = kornia.filters.canny(pred_y[i])[:][1]
            edge_input = edge_input[None,:]
            input = torch.cat((input,edge_input),dim=0)   
            
            edge_target = kornia.filters.canny(batch_y[i])[:][1]
            edge_target = edge_target[None,:]
            target = torch.cat((target,edge_target),dim=0)

        weight = 0.85

        # Loss is a sum of weighted MSE loss and edge preservating loss
        loss_ = weight * loss(pred_y,batch_y) + (1-weight) * loss(input,target)

        return loss_
    


loss_maps = {
    'mseloss': nn.MSELoss,
    'maeloss': nn.L1Loss,
    'weightloss': WeightLoss,
    'msceloss': MSCELoss
}