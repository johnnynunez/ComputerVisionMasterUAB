import json
import os.path

import fasttext
import numpy as np

def preprocess_caption(caption):
    # Replace special characters and convert to lowercase
    return caption.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")

def one_hot_encode_caption(model, caption):
    return model.get_sentence_vector(caption)

env_path = os.path.dirname(os.path.abspath(__file__))
# get path of current file
dataset_path = '../../../datasets/COCO'
# Load the FastText model
model = fasttext.load_model("./fasttext_wiki.en.bin")

train_path = os.path.join(dataset_path, 'captions_train2014.json')
# Load the JSON data
with open(train_path, "r") as file:
    data = json.load(file)

# Extract the captions
captions = [entry["caption"] for entry in data["annotations"]]

# Preprocess and one-hot encode the captions
encoded_captions = [one_hot_encode_caption(model, preprocess_caption(caption)) for caption in captions]

# Save the one-hot encoded captions to a new file
with open("encoded_captions_train2014.npy", "wb") as file:
    np.save(file, np.array(encoded_captions))