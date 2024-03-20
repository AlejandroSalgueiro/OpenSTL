import torch.nn as nn
import kornia
import torch
import torchvision

## Modified by Alejandro Salgueiro Dorado for a Master Thesis Project, Wageningen University, 24.11.2023


class WeightLoss(nn.Module):
    """
    Inspired by the weighted MSE loss for radar reflectivity from:
    Wu, D.; Wu, L.; Zhang, T.; Zhang, W.; Huang, J.; Wang, X. Short-Term Rainfall Prediction Based on Radar Echo Using an Improved Self-Attention PredRNN Deep Learning Model. Atmosphere 2022, 13, 1963. https://doi.org/10.3390/atmos13121963
    """
    def __init__(self,thresh1 = 20, thresh2 = 30):
        super().__init__()
        self.thresh1 = thresh1
        self.thresh2 = thresh2


    def forward(self,pred_y,batch_y):
        
        def weighted_mse_loss(input, target, weight):
                return (weight * (input - target) ** 2).mean()

        weights_tensor = torch.ones_like(batch_y)

        weights_tensor[(batch_y*70 < self.thresh1)] = 1
        weights_tensor[(batch_y*70 >= self.thresh1) & (batch_y*70 < self.thresh2)] = 2
        weights_tensor[(batch_y*70 >= self.thresh2)] = 3

        loss = weighted_mse_loss(pred_y,batch_y,weights_tensor)
        
        return loss


class MSCELoss(nn.Module):
    """
    Inspired by the edge preserving MSE loss from:
    Pandey, R.K., Saha, N., Karmakar, S., Ramakrishnan, A.G. (2018). MSCE: An Edge-Preserving Robust Loss Function for Improving Super-Resolution Algorithms. In: Cheng, L., Leung, A., Ozawa, S. (eds) Neural Information Processing. ICONIP 2018. Lecture Notes in Computer Science(), vol 11306. Springer, Cham. https://doi.org/10.1007/978-3-030-04224-0_49 
    """
    def __init__(self,weight=0.85):
         super().__init__()
         self.weight = weight


    def forward(self,pred_y,batch_y):
        
        mseloss = nn.MSELoss()
        
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

        # Loss is a sum of weighted MSE loss and edge preservating loss
        loss = self.weight * mseloss(pred_y,batch_y) + (1-self.weight) * mseloss(input,target)

        return loss
    
class WeightMSCELoss(nn.Module):
     '''
     Combination of Weighted loss and MSCELoss
     '''
     def __init__(self,weight=0.85,weightloss = WeightLoss(), mseloss = nn.MSELoss()):
          super().__init__()
          self.weight = weight
          self.weightloss = WeightLoss()
          self.mseloss = nn.MSELoss()

     def forward(self,pred_y,batch_y):

        dim = pred_y.shape[3]
        dim2 = pred_y.shape[4]

        input = torch.empty((0,12,1,dim,dim2))
        if torch.cuda.is_available():
            input = input.to(device="cuda")
        target = torch.empty((0,12,1,dim,dim2))
        if torch.cuda.is_available():
            target = target.to(device="cuda")

        # Loop to concatenate the edge pictures by sequences of 12
        for i in range(pred_y.shape[0]):
            edge_input = kornia.filters.canny(pred_y[i])[:][1]
            edge_input = edge_input[None,:]
            input = torch.cat((input,edge_input),dim=0)   
            
            edge_target = kornia.filters.canny(batch_y[i])[:][1]
            edge_target = edge_target[None,:]
            target = torch.cat((target,edge_target),dim=0)

        # Loss is a sum of weighted MSE loss and edge preservating loss
        loss = self.weight * self.weightloss(pred_y,batch_y) + (1-self.weight) * self.mseloss(input,target)

        return loss

class VGGPerceptualLoss(nn.Module):
    '''
    copied from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49 VGG perceputal loss
    author: Alper Ahmetoglu
    This loss compares the prediction and target by passing them through another neural network, 
    here VGG16 and passing them through another loss, here mse. Perceptual losses are usually 
    used to recreate realistic images 
    '''
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features[:4].eval().cuda())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inputs, targets, feature_layers=[0, 1, 2, 3], style_layers=[]):
        losses = 0
        for input,target in zip(inputs,targets):
            if input.shape[1] != 3:
                input = input.repeat(1, 3, 1, 1)
                target = target.repeat(1, 3, 1, 1)
            if self.resize:
                input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
                target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            loss = 0.0
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                if i in feature_layers:
                    loss += torch.nn.functional.mse_loss(x, y)
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.mse_loss(gram_x, gram_y)
            losses += loss
        return losses/inputs.shape[0]

loss_maps = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'weight': WeightLoss,
    'msce': MSCELoss,
    'weightmsce': WeightMSCELoss,
    'vggperceptual': VGGPerceptualLoss
}