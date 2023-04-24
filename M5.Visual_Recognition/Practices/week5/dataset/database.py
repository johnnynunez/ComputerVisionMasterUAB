
from PIL import Image
from torch.utils.data import Dataset
import json
import torch
import numpy as np


class ImageDatabase(Dataset):
    def __init__(self, ann_file, img_dir, num_samples=1000, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        # Keep only the first 2000 images
        self.images = self.images[:num_samples]
        # Create a list of 'id' of the self.images
        self.images_list_id = [self.images[i]['id'] for i in range(len(self.images))]
        
        self.annotations_an = self.annotations['annotations']
        # Delete the annotations that have 'image_id' not in the self.images_list_id
        self.annotations_an = [self.annotations_an[i] for i in range(len(self.annotations_an)) if self.annotations_an[i]['image_id'] in self.images_list_id]
        
        # Create a dictionary with the image id as key and the annotation caption id
        # Each image can have multiple captions id
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.images_list_id:
                continue
            else:
                if img_id not in self.img2ann:
                    self.img2ann[img_id] = [self.annotations_an[i]['id']]
                else:
                    self.img2ann[img_id].append(self.annotations_an[i]['id'])      
                
  
    def __len__(self):
        return len(self.images)
    
    def getCaptionsId_fromImageIdx(self, index):
        img_id = self.images[index]['id']
        return self.img2ann[img_id]
    
    
    def getImageIdx_fromId(self, img_id):
        return self.images_list_id.index(img_id)
        

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        img_id = self.images[index]['id']

        return image, img_id
    
    
    
class TextDatabase(Dataset):
    def __init__(self, ann_file, img_dir, ann_file_bert, network_text, num_samples=1000, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.network_text = network_text
        
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        # Keep only the first 2000 images
        self.images = self.images[:num_samples]
        # Create a list of 'id' of the self.images
        self.images_list_id = [self.images[i]['id'] for i in range(len(self.images))]
        
        self.annotations = self.annotations['annotations']
        # Delete the annotations that have 'image_id' not in the self.images_list_id. Also keep the index of the annotations that have 'image_id' in the self.images_list_id
        self.annotations_an = [self.annotations[i] for i in range(len(self.annotations)) if self.annotations[i]['image_id'] in self.images_list_id]
        self.annotations_an_idx = [i for i in range(len(self.annotations)) if self.annotations[i]['image_id'] in self.images_list_id]
        
        print('Loading the embeddings dict:')
        with open(ann_file_bert, 'rb') as f:
            self.embeddings = np.load(f)
        self.embeddings = self.embeddings[self.annotations_an_idx]
        
        
        # Create a dictionary with the caption id as key and the images id that have this caption
        # Each image can have multiple annotations
        self.capt2img = {}
        for i in range(len(self.annotations_an)):
            caption_id = self.annotations_an[i]['id']
            if caption_id not in self.capt2img:
                self.capt2img[caption_id] = [self.annotations_an[i]['image_id']]
            else:
                self.capt2img[caption_id].append(self.annotations_an[i]['image_id'])     
        
    def __len__(self):
        return len(self.annotations_an)
    
    def getImageId_fromCaptionIdx(self, index):
        caption_id = self.annotations_an[index]['id']
        return self.capt2img[caption_id]
    
    def getImageId_fromCaptionId(self, caption_id):
        return self.capt2img[caption_id]

    def getCaptionIdx_fromId(self, caption_id):
        return [i for i in range(len(self.annotations_an)) if self.annotations_an[i]['id'] == caption_id][0]
    
    def getstrCaption_fromCaptionIdx(self, index): # FOR BERT
        caption = self.annotations_an[index]['caption']
        
        return caption

    def __getitem__(self, index):
        
        caption = self.annotations_an[index]['caption']
        caption_id = self.annotations_an[index]['id']
        
        
        embedding = torch.tensor(self.embeddings[index], dtype=torch.float32)

        return embedding, caption_id