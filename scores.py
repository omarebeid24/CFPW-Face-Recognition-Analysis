import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract subject ID from the image label (assuming "subjectID_imageID.jpg")
def extract_subID(label):
    return label.split('_')[0]

# Function to separate genuine and impostor scores
def extract_genuine_impostor(full_matrix, labels):
    genuine_scores = []
    impostor_scores = []

    for i in range(len(labels)):
        subID_i = extract_subID(labels[i])

        for j in range(i + 1, len(labels)):
            subID_j = extract_subID(labels[j])

            if subID_i == subID_j:
                # Same subject -> Genuine score
                genuine_scores.append(full_matrix[i, j])
            else:
                # Different subjects -> Impostor score
                impostor_scores.append(full_matrix[i, j])

    return genuine_scores, impostor_scores

# Path to the Face_Embeddings folder (for label extraction)
embeddings_folder = 'G:/Master Material/Biometrics/assignment5/Face_Embeddings'

# Extract labels from the filenames in the Face_Embeddings folder
labels = [os.path.splitext(f)[0] for f in os.listdir(embeddings_folder) if f.endswith('.npy')]
print("Labels from embeddings folder:", labels)

# Load the similarity matrix
similarity_matrix = pd.read_csv('G:/Master Material/Biometrics/assignment5/similarity_matrix.csv', index_col=0)
matrix_headers = similarity_matrix.columns.tolist()
print("Headers from similarity matrix:", matrix_headers)

# Ensure labels match the similarity matrix columns
if set(labels) != set(matrix_headers):
    missing_in_labels = set(matrix_headers) - set(labels)
    missing_in_headers = set(labels) - set(matrix_headers)
    raise ValueError(f"Labels and headers mismatch!\nMissing in labels: {missing_in_labels}\nMissing in headers: {missing_in_headers}")

# Convert similarity matrix to a NumPy array for easier indexing
full_matrix = similarity_matrix.values

# Extract genuine and impostor scores
genuine_scores, impostor_scores = extract_genuine_impostor(full_matrix, labels)

# Save scores to text files
np.savetxt("genuine_scores.txt", genuine_scores, fmt="%.4f")
np.savetxt("impostor_scores.txt", impostor_scores, fmt="%.4f")

print(f"Genuine Scores Range: Min = {min(genuine_scores)}, Max = {max(genuine_scores)}")
print(f"Impostor Scores Range: Min = {min(impostor_scores)}, Max = {max(impostor_scores)}")
print(f"Extracted {len(genuine_scores)} genuine scores and {len(impostor_scores)} impostor scores.")

# Plot the score distributions with normalization
plt.figure(figsize=(10, 6))
plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine Scores',
color='blue', density=True)
plt.hist(impostor_scores, bins=50, alpha=0.6, label='Impostor Scores',
color='red', density=True)
plt.title('Genuine and Impostor Score Distribution')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.show()

