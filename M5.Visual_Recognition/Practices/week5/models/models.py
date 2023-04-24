import torch
import torch.nn as nn
import torchvision
# To avoid error with GLBIC_2.32 version run: pip install fasttext==0.9.2
import fasttext

    
class EmbeddingNetImage(nn.Module):
    def __init__(self, weights, network_image, dim_out_fc):   # dim_out_fc = 'as_image' or 'as_text'
        super(EmbeddingNetImage, self).__init__()
        
        self.network_image = network_image
        
        if network_image == 'fasterRCNN':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.model = nn.Sequential(*list(self.faster_rcnn.backbone.children())[:-1])
            in_features = 3840
            
            # for name, param in self.model.named_parameters():
            #     if 'fc' not in name:
            #         param.requires_grad = False
                
        elif network_image == 'RESNET50':
            self.model = torchvision.models.resnet50(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            
        elif network_image == 'RESNET101':
            self.model = torchvision.models.resnet101(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        
        self.activation = nn.ReLU()
        self.fc = nn.Linear(in_features, dim_out_fc)
    
    def init_weights(self):
        # Linear
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_in', nonlinearity='relu')
                
    def forward_resnet(self, x):
        output = self.model(x)
        return output
        
    def forward_faster(self, x):
        output = self.model(x)
        tensor_list = []
        for key, value in output.items():
            tensor_list.append(value.reshape(value.shape[0], value.shape[1], -1).max(dim=-1)[0])

        output = torch.cat(tensor_list, dim=1)
        return output   
    
    def forward(self, x):
        if self.network_image == 'fasterRCNN':
            output = self.forward_faster(x)
        else:
            output = self.forward_resnet(x)
        
        output = self.activation(output)
        output = self.fc(output)
        output = output / output.pow(2).sum(dim=-1, keepdim=True).sqrt()
        
        return output 
    


class EmbeddingNetText(nn.Module):
    def __init__(self, weights, device, network_text='FastText', dim_out_fc = 'as_image'):  # type = 'FastText' or 'BERT'
        super(EmbeddingNetText, self).__init__()
        self.device = device
        self.network_text = network_text
        
        if network_text == 'FastText':
            
            if dim_out_fc < 1500:
                self.fc = nn.Sequential(nn.Linear(300, 512), 
                                    nn.PReLU(), 
                                    nn.Linear(512, dim_out_fc))
            else:
                self.fc = nn.Sequential(nn.Linear(300, 1024), 
                                    nn.PReLU(), 
                                    nn.Linear(1024, dim_out_fc))
                
        elif network_text == 'BERT':
            if dim_out_fc < 1500:
                self.fc = nn.Sequential(nn.Linear(768, dim_out_fc))
            else:
                self.fc = nn.Sequential(nn.Linear(768, 1024), 
                                    nn.PReLU(), 
                                    nn.Linear(1024, dim_out_fc))
        self.activation = nn.ReLU()
    
    def init_weights(self):
        # Linear
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        

    def forward(self, x):
        
        output = self.activation(x)
        output = self.fc(output)
        output = output / output.pow(2).sum(dim=-1, keepdim=True).sqrt()
        

        return output
    

    
    

class TripletNetIm2Text(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetIm2Text, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_image(x1)
        output2 = self.embedding_net_text(x2)
        output3 = self.embedding_net_text(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)
    
    def get_embedding_text(self, x):
        return self.embedding_net_text(x)
    
    
    
class TripletNetText2Img(nn.Module):
    def __init__(self, embedding_net_image, embedding_net_text):
        super(TripletNetText2Img, self).__init__()
        self.embedding_net_image = embedding_net_image
        self.embedding_net_text = embedding_net_text

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net_text(x1)
        output2 = self.embedding_net_image(x2)
        output3 = self.embedding_net_image(x3)
        return output1, output2, output3

    def get_embedding_image(self, x):
        return self.embedding_net_image(x)
    
    def get_embedding_text(self, x):
        return self.embedding_net_text(x)