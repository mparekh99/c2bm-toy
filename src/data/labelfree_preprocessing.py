from env import CACHE
import os
import torch
import json
import pandas as pd
from random import sample
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from typing import Dict, List, Union
from scipy.stats import pearsonr

from transformers import CLIPModel, CLIPProcessor

def load_robot_clip_model(device='cpu'):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return model, processor

def encode_image_clip(model, processor, images, device):
    with torch.no_grad():
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(device)
            emb = model.get_image_features(pixel_values=pixel_values)
        else:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.detach().cpu().numpy()

def encode_text_clip(model, processor, texts, device):
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.detach().cpu().numpy()

def transform_concepts_to_binary(c_embeddings):

    cluster_assignments = np.zeros_like(c_embeddings)
 
    # Iterate over each dimension
    for i in range(c_embeddings.shape[1]):
        #print('Concept ', i)
        # Reshape the dimension to be a 2D array (n, 1)
        concept = c_embeddings[:, i].reshape(-1, 1)
        
        # Apply k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(concept)
        
        # Get the cluster labels
        labels = kmeans.labels_
        
        # Calculate the mean of each cluster
        cluster_means = [concept[labels == j].mean() for j in range(2)]
        
        # Determine which cluster has the lower mean and which has the higher mean
        if cluster_means[0] < cluster_means[1]:
            cluster_assignments[:, i] = labels
        else:
            cluster_assignments[:, i] = 1 - labels
            
    c = torch.tensor(cluster_assignments)

    return c

def _generate_img_embeddings_and_assign_concepts_maxpool(dataset,
                                                          concepts,
                                                          clip_model,
                                                          clip_processor,
                                                          input_encoder,
                                                          batch_size,
                                                          device):
    # encode concepts once
    concepts_embeddings = encode_text_clip(clip_model, clip_processor, concepts, device)

    images_resnet_embeddings = []
    images_c_similarities = []
    y = []

    # process one episode at a time
    for idx in tqdm(range(len(dataset))):
        row = dataset.rows[idx]
        
        # load all frames for this episode
        frames = []
        for frame_path in row['frames']:
            image = Image.open(frame_path).convert("RGB")
            image = dataset.transform(image)
            frames.append(image)
        
        # process in mini-batches to avoid OOM
        frame_tensor = torch.stack(frames, dim=0)  # [num_frames, C, H, W]
        
        all_clip_embs = []
        all_resnet_embs = []
        
        for i in range(0, len(frames), batch_size):
            batch = frame_tensor[i:i+batch_size].to(device)
            
            # CLIP embeddings
            clip_emb = encode_image_clip(clip_model, clip_processor, batch, device)
            all_clip_embs.append(clip_emb)
            
            # ResNet embeddings
            with torch.no_grad():
                resnet_emb = input_encoder(batch).squeeze().detach().cpu().numpy()
                if resnet_emb.ndim == 1:
                    resnet_emb = resnet_emb.reshape(1, -1)
            all_resnet_embs.append(resnet_emb)
        
        # concatenate all frame embeddings
        all_clip_embs = np.concatenate(all_clip_embs, axis=0)  # [num_frames, clip_dim]
        all_resnet_embs = np.concatenate(all_resnet_embs, axis=0)  # [num_frames, resnet_dim]
        
        # compute similarity per frame
        frame_similarities = metrics.pairwise.cosine_similarity(
            all_clip_embs, concepts_embeddings
        )  # [num_frames, num_concepts]
        
        # max pool across frames
        episode_similarity = frame_similarities.max(axis=0)  # [num_concepts]
        
        # mean pool resnet embeddings for X
        episode_resnet = all_resnet_embs.mean(axis=0)  # [resnet_dim]
        
        images_c_similarities.append(episode_similarity)
        images_resnet_embeddings.append(episode_resnet)
        y.append(np.float32(row['label']))

    # binarize concept scores
    c = transform_concepts_to_binary(torch.tensor(np.array(images_c_similarities)))
    dataset.c = c
    dataset.y = torch.tensor(y).unsqueeze(1)
    dataset.X = torch.tensor(np.array(images_resnet_embeddings))
    return dataset

def concepts_analysis(c, y, concepts, images_c_similarities):

    sum_c = c.sum(dim=1)

    # boxplot of sum of concepts by Pneumothorax status
    data = np.column_stack((c.numpy(), y.numpy(), sum_c))
    column_names = concepts + ["label", "sum_c"]
    df = pd.DataFrame(data, columns = column_names)
    sns.boxplot(x='label', y='sum_c', data=df)
    plt.title('Distribution of concepts by Pneumothorax Status')
    plt.savefig('boxplot_concepts_pneumonia.png')
    plt.clf()

    
   # Plots of the distribution of sum_c for label == 0 and label == 1
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot for label == 0
    subset_0 = df[df['label'] == 0]
    axs[0].hist(subset_0['sum_c'], bins=20, alpha=0.7, color='blue', density=True)
    axs[0].set_xlabel('Sum of Concepts (sum_c)')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Distribution of sum_c for Label 0')
    axs[0].legend(['Label 0'])

    # Plot for label == 1
    subset_1 = df[df['label'] == 1]
    axs[1].hist(subset_1['sum_c'], bins=20, alpha=0.7, color='orange', density=True)
    axs[1].set_xlabel('Sum of Concepts (sum_c)')
    axs[1].set_title('Distribution of sum_c for Label 1')
    axs[1].legend(['Label 1'])

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot
    plt.savefig('distribution_sum_c_for_labels_0_and_1.png')

    # contingency tables and correlation between concepts and Pneumothorax status
    contingency_tables = {}
    for concept in concepts:
        contingency_table = pd.crosstab(df[concept], df['label'], rownames=[concept], colnames=['label'], margins=True)
        contingency_tables[concept] = contingency_table
        print(contingency_table)

    correlations = {}
    with open('contingency_and_correlations.txt', 'w') as file:
        for concept in concepts:
            contingency_table = pd.crosstab(df[concept], df['label'], rownames=[concept], colnames=['label'], margins=True)
            contingency_tables[concept] = contingency_table
            
            corr, _ = pearsonr(df[concept], df['label'])
            correlations[concept] = corr
            
            file.write(f"Contingency Table for {concept}:\n{contingency_table}\n\n")
            file.write(f"Correlation for {concept}: {corr:.4f}\n\n")

    plt.clf()



    return None

def generate_img_embeddings_and_assign_concepts(dataset_name, dataset, concepts,
                                                clip_model: CLIPModel,
                                                clip_processor: CLIPProcessor,
                                                batch_size=128, device='cpu'):
    from torchvision.models import resnet18, ResNet18_Weights
    input_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    modules = list(input_encoder.children())[:-1]
    input_encoder = nn.Sequential(*modules)
    input_encoder.to(device)
    input_encoder.eval()

    for split, data in dataset.data.items():
        print(f"Processing split: {split}", flush=True)
        data = _generate_img_embeddings_and_assign_concepts_maxpool(
            data, concepts, clip_model, clip_processor,
            input_encoder, batch_size, device
        )
        dataset.data[split] = data

    dataset.c_info = {'names': concepts, 'cardinality': [2] * len(concepts)}
    return dataset