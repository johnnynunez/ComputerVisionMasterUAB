import json
import os

import numpy as np
import torch
from tqdm import tqdm

import fasttext

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    torch.cuda.amp.GradScaler()
elif torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("cpu")
else:
    print("CPU is available")
    device = torch.device("cpu")


# Load the FastText model
model = fasttext.load_model('/ghome/group03/M5-Project/week5/utils/text/fasttext_wiki.en.bin')



env_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = '/ghome/group03/mcv/datasets/COCO'

train_path = os.path.join(dataset_path, 'captions_train2014.json')
val_path = os.path.join(dataset_path, 'captions_val2014.json')


with open(train_path, "r") as file:
    data = json.load(file)

output_annotations = []
output_numpy = []

for annotation in tqdm(data["annotations"]):
    caption = annotation['caption']
    id = annotation['id']
    image_id = annotation['image_id']
    
    caption = caption.replace('.', '').replace(',','').lower().split()
    
    output = [model[word] for word in caption]
    
    output = np.mean(output, axis=0)
    
    output_numpy.append(output)

    
output_numpy = np.array(output_numpy)


    
# Save the numpy array 
with open(os.path.join(dataset_path,"encoded_captions_train2014_fasttext.npy"), "wb") as file:
    np.save(file, output_numpy)


# ------ Validation set ------

with open(val_path, "r") as file:
    data = json.load(file)

# Extract the captions
output_annotations = []
output_numpy = []

for annotation in tqdm(data["annotations"]):
    caption = annotation['caption']
    id = annotation['id']
    image_id = annotation['image_id']
    
    caption = caption.replace('.', '').replace(',','').lower().split()
    
    output = [model[word] for word in caption]
    
    output = np.mean(output, axis=0)
    
    output_numpy.append(output)

    
output_numpy = np.array(output_numpy)


# Save the numpy array 
with open(os.path.join(dataset_path, "encoded_captions_val2014_fasttext.npy"), "wb") as file:
    np.save(file, output_numpy)