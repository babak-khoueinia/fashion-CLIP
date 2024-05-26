import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

import torch.nn.functional as F
from torchvision.transforms import ToTensor

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

#device = "cuda" if torch.cuda.is_available() else "cpu"
#processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
#model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")

class CLIP:
    def __init__(self, processor, model):
        
        # get list of folders in the data folder
        folders = os.listdir('data')
        # create a dictionary to store the images
        imgs = {}
        # loop through each folder and read all images
        for folder in folders:
            
            imgs[folder] = []
            for image in os.listdir('data/' + folder):
                try:
                    img = Image.open('data/' + folder + '/' + image)
                    imgs[folder].append(img)
                except UnidentifiedImageError:
                    print(f"Cannot identify image file: {'data/' + folder + '/' + image}")
        
        self.all_images = imgs['compressed-fashion']
        self.pros = processor
        self.mdl = model
        self.mdl.to('cuda')
        
    def image_classification(self, input_image, class_list): 
        out = {}
        self.mdl.to('cuda')
        #temoo = self.pros(images = input_image, text = class_list, return_tensors = "pt")
        inputs = self.pros(images = input_image, text = class_list, return_tensors="pt")
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        #res = self.mdl(pixel_values = temoo.pixel_values, input_ids = temoo.input_ids, attention_mask = temoo.attention_mask)
        with torch.no_grad():
            res = self.mdl(**inputs)
        probs = F.softmax(res.logits_per_text, dim=0)
        for i, n in enumerate(class_list): 
            out[n] = float(probs[i])

        return out
    
    def embedder(self):
        # Split list into batches of 150 images
        batch_size = 250
        batches = [self.all_images[i:i + batch_size] for i in range(0, len(self.all_images), batch_size)]
        # Create a tensor to store the embeddings on the GPU
        embeddings = torch.tensor([]).to('cuda')
        
        # Move model to GPU
        self.mdl.to('cuda')
        
        for batch in batches:
            class_list = ['BIG BLACK AND WHITE TROWSER']
            # Process the batch and move inputs to GPU
            inputs = self.pros(images=batch, text=class_list, return_tensors="pt", padding=True)
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.mdl(**inputs)
                # Ensure the embeddings are moved to the GPU
                embeddings = torch.cat((embeddings, outputs.image_embeds.to('cuda')), 0)

        return embeddings
        

    def recursive_search(self, image_vectors, concept_vector, batch_size=20, top_set=5):
        # Ensure the input tensors are on the GPU
        device = image_vectors.device
        concept_vector = concept_vector.to(device)
        
        # Batch images in groups of batch_size
        batches = [image_vectors[i:i + batch_size] for i in range(0, len(image_vectors), batch_size)]
        
        # Create a list to store the scores
        scores = torch.tensor([]).to(device)
        
        # Create a tensor to store the best vectors from each batch
        bestof_batch = torch.tensor([]).to(device).reshape(0, image_vectors.size(1))
        
        for batch in batches:
            similarity = F.cosine_similarity(batch, concept_vector.expand(batch.size(0), -1), dim=1)
            scores = torch.cat((scores, similarity), 0)
            
            # Extract top top_set of each batch, or fewer if batch size is less than top_set
            k = min(top_set, batch.size(0))
            top_scores, top_indices = torch.topk(similarity, k)
            bestof_batch = torch.cat((bestof_batch, batch[top_indices]), 0)
        
        print(f'bestof_batch shape before recursion or final selection: {bestof_batch.shape}')
        
        if len(bestof_batch) > 50:
            bestof_batch, top_scores = self.recursive_search(bestof_batch, concept_vector, batch_size, top_set)
        else:
            # Find cosine similarity between the best of batch and the concept vector
            similarity = F.cosine_similarity(bestof_batch, concept_vector.expand(bestof_batch.size(0), -1), dim=1)
            
            # Apply softmax to the similarity scores
            scores = F.softmax(similarity, dim=0)
            
            # Find the top top_set scores and their corresponding indices, or fewer if not enough elements
            k = min(top_set, bestof_batch.size(0))
            top_scores, top_indices = torch.topk(scores, k)
            print(f'top_indices: {top_indices}')
            print(f'top_scores: {top_scores}')
            print(f'final bestof_batch shape: {bestof_batch.shape}')
            
            bestof_batch = bestof_batch[top_indices]
        
        return bestof_batch, top_scores
    


