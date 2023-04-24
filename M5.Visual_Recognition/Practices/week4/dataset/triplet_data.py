import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import cv2
from tqdm import tqdm

import json
import random
import sys


from pycocotools.coco import COCO


class TripletMITDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mit_dataset, split_name='train'):
        self.mit_dataset = mit_dataset

        self.train = split_name == 'train'

        self.transform = self.mit_dataset.transform

        if self.train:
            self.train_labels = self.mit_dataset.targets
            self.train_data = self.mit_dataset.samples
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mit_dataset.targets
            self.test_data = self.mit_dataset.samples
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.open(img1[0])
        img2 = Image.open(img2[0])
        img3 = Image.open(img3[0])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mit_dataset)




class TripletCOCODataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, coco_dataset, obj_img_dict, dataset_path, split_name='train', dict_negative_img=None, transform=None):

        self.coco = coco_dataset.coco
        self.obj_img_dict = obj_img_dict[split_name]
        self.transform = transform
        self.dataset_path = dataset_path
        self.dict_negative_img = dict_negative_img

        # Get ID's of all images
        self.imgs_ids = self.coco.getImgIds()
            

        # Create a list of all image IDs that contain at least one object from obj_img_dict
        self.obj_img_ids = []
        for obj in self.obj_img_dict:
            cat_ids = self.coco.getCatIds(catNms=[obj])
            img_ids = self.coco.getImgIds(catIds=cat_ids)
            self.obj_img_ids.extend(img_ids)
        self.obj_img_ids = list(set(self.obj_img_ids))

        # Create a list of all image IDs that do not contain any object from obj_img_dict
        self.non_obj_img_ids = list(set(self.imgs_ids) - set(self.obj_img_ids))
        
        
        # If dict_negative_img is None, create a dictionary with all id images as keys and for each key a list of all images that do not contain any object of the same category
        if self.dict_negative_img is None:
            print("dict_negative_img is not found")
            print("Creating dict_negative_img")
            self.dict_negative_img = {}
            for img_id in tqdm(self.obj_img_ids):
                self.dict_negative_img[img_id] = self.process_img_id(img_id)
            
            size = sys.getsizeof(self.dict_negative_img)
            print("The size of the dictionary is {} bytes".format(size))
            
            path = f'/ghome/group03/mcv/datasets/COCO/{split_name}_dict_negative_img_low.json'
            
            with open(path, 'w') as fp:
                json.dump(self.dict_negative_img, fp)
            
            print("dict_negative_img is created and saved to {}".format(path))
        
                
                
    def process_img_id(self, img_id):
        img_ann_ids = self.coco.getAnnIds(imgIds=img_id)
        img_anns = self.coco.loadAnns(img_ann_ids)
        img_cat_ids = list(set([ann['category_id'] for ann in img_anns]))

        negative_img_id = [item for item in self.obj_img_ids if item != img_id and all(
            cat_id not in self.obj_img_dict for cat_id in img_cat_ids)]
        
        try:
            negative_img_id = random.sample(negative_img_id, 1000)
        except:
            negative_img_id = negative_img_id
            print(f'Image {img_id} has less than 500 negative images')
        
        return negative_img_id
       

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def resize_bounding_boxes(self, boxes, image_size, target_size):
        """
        Resize bounding boxes based on the size of the image.

        Args:
            boxes (list): List of bounding box coordinates [x1, y1, x2, y2].
            image_size (tuple): Original size of the image (width, height).
            target_size (tuple): Target size of the image (width, height).

        Returns:
            list: List of resized bounding box coordinates [x1, y1, x2, x2].
        """
        resized_boxes = []
        ratio_x = target_size[0] / image_size[0]
        ratio_y = target_size[1] / image_size[1]
        for box in boxes:
            x = box[0] * ratio_x
            y = box[1] * ratio_y
            width = box[2] * ratio_x
            height = box[3] * ratio_y
            resized_boxes.append([x, y, x + width, y + height])
        return resized_boxes

    def __getitem__(self, index):
        # Choose anchor image
        while True:
            # Choose anchor image
            anchor_img_id = self.obj_img_ids[index % len(self.obj_img_ids)]
            anchor_img = self.coco.loadImgs(anchor_img_id)[0]
            anchor_ann_ids = self.coco.getAnnIds(imgIds=anchor_img_id)  # Get the id of the instances
            anchor_anns = self.coco.loadAnns(anchor_ann_ids)
            anchor_cat_ids = list(set([ann['category_id'] for ann in anchor_anns]))
            anchor_cat_ids_str = [str(cat) for cat in anchor_cat_ids]

            if not anchor_cat_ids_str:
                index += 1
                continue
            else:
                break
            
        # Choose positive image that contains at least one object from the same class as the anchor
        positive_img_id = anchor_img_id
        while positive_img_id == anchor_img_id:
            rand_cat = random.choice(anchor_cat_ids_str)
            possible_positive_imgs = self.intersection(self.obj_img_dict[rand_cat], self.obj_img_ids)
            if possible_positive_imgs == []:
                continue
            positive_img_id = random.choice(self.obj_img_dict[rand_cat])

        positive_img = self.coco.loadImgs(positive_img_id)[0]
        positive_ann_ids = self.coco.getAnnIds(imgIds=positive_img_id)  # Get the id of the instances
        positive_anns = self.coco.loadAnns(positive_ann_ids)

        # # Choose negative image that does not contain any object from the same class as the anchor
        negative_anns = []
        while negative_anns == []:
            negative_img_id = random.choice(self.dict_negative_img[str(anchor_img_id)])
            negative_img = self.coco.loadImgs(negative_img_id)[0]
            negative_ann_ids = self.coco.getAnnIds(imgIds=negative_img_id)  # Get the id of the instances
            negative_anns = self.coco.loadAnns(negative_ann_ids)
        

        # Load anchor, positive, and negative images
        anchor_img_path = os.path.join(self.dataset_path, anchor_img['file_name'])
        positive_img_path = os.path.join(self.dataset_path, positive_img['file_name'])
        negative_img_path = os.path.join(self.dataset_path, negative_img['file_name'])

        anchor_img = Image.open(anchor_img_path).convert('RGB')
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')


        # Apply transformations to images, if provided
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        

        return (anchor_img, positive_img, negative_img), []

    def __len__(self):
        return len(self.obj_img_ids)
    
    

class TripletCOCODatasetFast(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, coco_dataset, obj_img_dict, dataset_path, split_name='train', dict_negative_img=None, transform=None):
        self.dict_negative_img = dict_negative_img
        self.obj_img_dict = obj_img_dict
        self.transform = transform
        self.dataset_path = dataset_path
        
        # Obtain labels
        self.objs = {}
        self.labelTrain = self.obj_img_dict[split_name]
        
        # Get objects per image
        for obj in self.labelTrain.keys():
            for image in self.labelTrain[obj]:
                if image in self.objs.keys():
                    self.objs[image].append(obj)
                else:
                    self.objs[image] = [obj]
        
        # Remove images that do not have any object
        aux = 0
        while aux < len(self.trainImages):
            image1 = self.trainImages[aux]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.trainImages[aux]
            else:
                aux += 1
        
    
    def has_common_object(self, objs1, objs2):
        
        for obj in objs1:
            if obj in objs2:
                return True
        return False
    
    def __getitem__(self, index):
        # Get anchor image
        img1name = self.trainImages[index]
        
        # Get random positive image
        img1value = int(img1name[:-4].split("_")[2])
        img1objs = self.objs[img1value]
        
        positiveImgValue = img1value
        while positiveImgValue == img1value:
            # Get random obj
            sharingObj = np.random.choice(img1objs)
            # Get random image 
            positiveImgValue = np.random.choice(self.labelTrain[sharingObj])
        img2name = "COCO_train2014_{:012d}.jpg".format(positiveImgValue)
        
        
        # Get random negative image
        while True:
            # Get random image
            img3name = np.random.choice(self.trainImages)
            img3value = int(img3name[:-4].split("_")[2])
            img3objs = self.objs[img3value] 
            
            if not self.has_common_object(img3objs, img1objs):
                break
                  
        # Read images
        img1 = cv2.imread(self.trainImagesFolder +'/' + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.trainImagesFolder + '/' + img2name)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.imread(self.trainImagesFolder + '/' + img3name)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        
        # Transform
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.trainImages)
    

    
class TripletCOCORetrieval(Dataset):
    """
    Dataset for retrieval
    """

    def __init__(self, databaseImagesFolder, obj_img_dict, 
                 transform, split_name, allLabels  = None):
        
        self.labelDatabase = obj_img_dict[split_name]
        self.transform = transform
        self.databaseImagesFolder = databaseImagesFolder + '/'
        self.databaseImages = os.listdir(databaseImagesFolder)
        
        # Obtain labels
        self.objs = {}
        
        # Get objects per image
        for obj in self.labelDatabase.keys():
            for image in self.labelDatabase[obj]:
                if image in self.objs.keys():
                    self.objs[image].append(obj)
                else:
                    self.objs[image] = [obj]
        
        # Remove images that do not have any object
        aux = 0
        while aux < len(self.databaseImages):
            image1 = self.databaseImages[aux]
            image1Num = int(image1[:-4].split("_")[2])
            
            if not(image1Num in self.objs.keys()):
                del self.databaseImages[aux]
            else:
                aux += 1
        
        if not(allLabels is None):
            # Get every object in the image
            coco=COCO(allLabels)
            
            # Obtain labels
            self.objs = {}
            for image in self.databaseImages:
                imageId = int(image[:-4].split('_')[-1])
                ann_ids = coco.getAnnIds(imgIds=[imageId])
                anns = coco.loadAnns(ann_ids)
                annId = []
                for ann in anns:
                    annId.append(str(ann["category_id"]))
                if len(annId)>0:
                    self.objs[imageId]=annId

    def __getitem__(self, index):
        # Get image
        img1name = self.databaseImages[index]
        img1 = cv2.imread(self.databaseImagesFolder + img1name)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Transform
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, []

    def getObjs(self, index):
        # Get image name
        img1name = self.databaseImages[index]

        # Get objs
        img1value = int(img1name[:-4].split("_")[2])
        
        img1objs = self.objs[img1value]
        
        return img1objs
        
    def __len__(self):
        return len(self.databaseImages)
