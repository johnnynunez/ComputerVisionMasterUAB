import json
import os

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

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


def encode_caption(tokenizer, model, caption):
    inputs = tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Install the transformers library if not installed
# !pip install transformers

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

# Load the JSON data

env_path = os.path.dirname(os.path.abspath(__file__))
# get path of current file
# dataset_path = '../../../../datasets/COCO'
dataset_path = '/ghome/group03/mcv/datasets/COCO'

train_path = os.path.join(dataset_path, 'captions_train2014.json')
val_path = os.path.join(dataset_path, 'captions_val2014.json')
with open(train_path, "r") as file:
    data = json.load(file)
# Extract the captions
output_annotations = []

for annotation in tqdm(data["annotations"]):
    caption = annotation['caption']
    id = annotation['id']
    image_id = annotation['image_id']

    inputs = tokenizer(caption, return_tensors='pt').to(device)
    outputs = model(**inputs)

    logits = outputs.last_hidden_state[0, 0, :].to("cpu").squeeze().detach().numpy()

    output_annotations.append({
        'caption': logits.tolist(),
        'id': id,
        'image_id': image_id,
    })

# json.dumps({'annotations': output_annotations})
with open("encoded_captions_train2014_bert.json", "w") as file:
    json.dump(output_annotations, file)


# ------ Validation set ------

with open(val_path, "r") as file:
    data = json.load(file)

# Extract the captions
output_annotations = []

for annotation in tqdm(data["annotations"]):
    caption = annotation['caption']
    id = annotation['id']
    image_id = annotation['image_id']

    inputs = tokenizer(caption, return_tensors='pt').to(device)
    outputs = model(**inputs)

    logits = outputs.last_hidden_state[0, 0, :].to("cpu").squeeze().detach().numpy()

    output_annotations.append({
        'caption': logits.tolist(),
        'id': id,
        'image_id': image_id,
    })

# json.dumps({'annotations': output_annotations})
with open("encoded_captions_val2014_bert.json", "w") as file:
    json.dump(output_annotations, file)