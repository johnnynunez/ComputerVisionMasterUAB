
import random
from PIL import Image
from torch.utils.data import Dataset
import json
import random
import torch
import ujson
import numpy as np


class TripletIm2Text(Dataset):
    def __init__(self, ann_file, img_dir, ann_file_bert, network_text, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.network_text = network_text
        
   
        print('Loading the embeddings dict...')
        with open(ann_file_bert, 'rb') as f:
            self.embeddings = np.load(f)
        print('Done!')


        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.annotations_an = self.annotations['annotations']

        # Create a dictionary with the image id as key and the annotation index
        # Each image can have multiple annotations
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [i]
            else:
                self.img2ann[img_id].append(i)       
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.images[index]['file_name']
        img_id = self.images[index]['id']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Choose randomly one captions for the image
        idx_pos = random.choice(self.img2ann[img_id])
        assert self.annotations_an[idx_pos]['image_id'] == img_id
        positive_caption_id = self.annotations_an[idx_pos]['id']
        positive_caption = self.annotations_an[idx_pos]['caption']
        
        
        # Choose randomly one caption that is not the same as the positive caption
        negative_caption_id = positive_caption_id
        while negative_caption_id == positive_caption_id:
            neg_ann_idx = random.choice(range(len(self.annotations_an)))
            neg_ann = self.annotations_an[neg_ann_idx]
            if neg_ann_idx in self.img2ann[img_id]:
                continue
            negative_caption_id = neg_ann['id'] 
            
        negative_caption = neg_ann['caption']
        
        # positive_caption = torch.tensor(self.embeddings[idx_pos]['caption'])
        # assert self.embeddings[idx_pos]['id'] == positive_caption_id
        # negative_caption = torch.tensor(self.embeddings[neg_ann_idx]['caption'])
        # assert self.embeddings[neg_ann_idx]['id'] == negative_caption_id
        
        positive_caption = torch.tensor(self.embeddings[idx_pos], dtype=torch.float32)
        negative_caption = torch.tensor(self.embeddings[neg_ann_idx], dtype=torch.float32)
            
        return (image, positive_caption, negative_caption), []
    
    
    
class TripletText2Im(Dataset):
    def __init__(self, ann_file, img_dir, ann_file_bert, network_text, num_images=20000, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.network_text = network_text
        

        print('Loading the embeddings dict...')
        with open(ann_file_bert, 'rb') as f:
            self.embeddings = np.load(f)
   
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']               # 80.000
        self.annotations_an = self.annotations['annotations']  # 400.000
 
                
        # Create a dictionary with the image id as key and the annotation index
        # Each image can have multiple annotations
        self.img2ann = {}
        for i in range(len(self.annotations_an)):
            img_id = self.annotations_an[i]['image_id']
            if img_id not in self.img2ann:
                self.img2ann[img_id] = [self.annotations_an[i]['id']]
            else:
                self.img2ann[img_id].append(self.annotations_an[i]['id']) 
        
        self.img2idx = {}
        # Also add in this dictionary the index of the image in self.images
        for i in range(len(self.images)):
            img_id = self.images[i]['id']
            if img_id not in self.img2idx:
                self.img2idx[img_id] = [i]
            else:
                print('Image repeated')
                self.img2idx[img_id].append(i)
           
        
    def __len__(self):
        return len(self.annotations_an)

    def __getitem__(self, index):
        caption_id = self.annotations_an[index]['id']
        caption = self.annotations_an[index]['caption']
        
        # Positive image
        pos_img_id = self.annotations_an[index]['image_id']
        pos_img_idx = self.img2idx[pos_img_id][0]
        img_path = self.img_dir + '/' + self.images[pos_img_idx]['file_name']
        positive_image = Image.open(img_path).convert('RGB')
        
        # Negative image
        neg_img_id = pos_img_id
        while neg_img_id == pos_img_id:
            neg_img = random.choice(self.images)
            neg_img_id = neg_img['id']
            if caption_id in self.img2ann[neg_img_id]:
                continue
            
        neg_img_idx = self.img2idx[neg_img_id][0]
        neg_img_path = self.img_dir + '/' + self.images[neg_img_idx]['file_name']
        negative_image = Image.open(neg_img_path).convert('RGB')
        
        if self.transform is not None:
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
            
  
        # anchor_embedding = torch.tensor(self.embeddings[index]['caption'])
        # assert self.embeddings[index]['id'] == caption_id
        caption = torch.tensor(self.embeddings[index], dtype=torch.float32)
            
        
        return (caption, positive_image, negative_image), []
    
    
