import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
source_folder = "G:/Master Material/Biometrics/assignment5/Pre_processed_dataset"
dest_folder = "G:/Master Material/Biometrics/assignment5/Face_Embeddings"
similarity_csv = "G:/Master Material/Biometrics/assignment5/similarity_matrix.csv"

# Create the destination folder if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Get a list of all jpg images in the source folder
image_files = [img_file for img_file in os.listdir(source_folder) if img_file.endswith('.jpg')]

# Dictionary to store embeddings with filenames (without .jpg)
embeddings_dict = {}

# Use tqdm for progress bar
with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
    for img_file in image_files:
        img_path = os.path.join(source_folder, img_file)
        
        try:
            # Extract embeddings using Facenet512 model
            result = DeepFace.represent(img_path, model_name='Facenet512', enforce_detection=False)
            embeddings = result[0]['embedding']
            
            # Save embeddings as a .npy file
            embedding_filename = img_file.replace('.jpg', '.npy')
            embedding_path = os.path.join(dest_folder, embedding_filename)
            np.save(embedding_path, embeddings)

            # Store embeddings in dictionary with the filename stripped of '.jpg'
            filename_no_ext = img_file.replace('.jpg', '')
            embeddings_dict[filename_no_ext] = embeddings
        
        except Exception as e:
            print(f"Failed to process {img_file}: {e}")

        # Update the progress bar
        pbar.update(1)

# Create a similarity matrix
print("Computing similarity matrix...")
filenames = list(embeddings_dict.keys())  # Get filenames without .jpg
embeddings = list(embeddings_dict.values())

# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Normalize match scores by multiplying by 100
normalized_scores = similarity_matrix * 100

# Convert to DataFrame and save as CSV
similarity_df = pd.DataFrame(normalized_scores, index=filenames, columns=filenames)
similarity_df.to_csv(similarity_csv, index=True)
print(f"Similarity matrix saved to {similarity_csv}")
