import numpy as np

# Load scores from text files
genuine_scores = np.loadtxt('genuine_scores.txt')
impostor_scores = np.loadtxt('impostor_scores.txt')

# Get the indices of the top 3 highest impostor scores
top_impostor_indices = np.argsort(impostor_scores)[-3:][::-1]  # Sort in descending order and get top 3

# Get the indices of the bottom 3 lowest genuine scores
bottom_genuine_indices = np.argsort(genuine_scores)[:3]  # Sort in ascending order and get bottom 3

# Print the top 3 impostor scores
print("Top 3 Highest-Scoring Impostors:")
for idx in top_impostor_indices:
    print(f"Index: {idx}, Score: {impostor_scores[idx]:.2f}")

# Print the bottom 3 genuine scores
print("\nBottom 3 Lowest-Scoring Genuines:")
for idx in bottom_genuine_indices:
    print(f"Index: {idx}, Score: {genuine_scores[idx]:.2f}")
