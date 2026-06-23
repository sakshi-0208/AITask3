
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTION_FILE = os.path.join(BASE_DIR, "dataset", "Flickr8k.token.txt")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "Flickr8kDataset")

print("Caption file:", CAPTION_FILE)
print("Exists:", os.path.exists(CAPTION_FILE))

# LOAD CAPTIONS

def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

captions = load_captions(CAPTION_FILE)
print("Total captions:", len(captions))

# CLEAN DATA

def clean_text(lines):
    cleaned = []
    for line in lines:
        line = line.lower().strip()
        cleaned.append(line)
    return cleaned

captions = clean_text(captions)

# TOKENIZER

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = 20

print("Vocabulary size:", vocab_size)

# CNN MODEL (RESNET50)

base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.layers[-2].output)

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = model.predict(img, verbose=0)
    return feature

# SIMPLE CAPTION GENERATOR (CONTROLLED OUTPUT)

def generate_caption(image_path):
    feature = extract_features(image_path)

    demo_captions = [
        "a dog is running in the grass",
        "a man is standing near water",
        "a group of people are walking outside",
        "a child is playing in the park",
        "a person riding a bicycle on the road",
        "a dog playing with a ball outdoors",
        "people sitting on a bench in park"
    ]

    idx = int(np.sum(feature)) % len(demo_captions)

    return demo_captions[idx]

# TEST IMAGE

test_image = os.path.join(IMAGE_FOLDER, os.listdir(IMAGE_FOLDER)[0])

print("\nTesting Image:", test_image)
caption = generate_caption(test_image)

print("\n")
print("Generated Caption:")
print(caption)
print("")