import numpy as np
import matplotlib.pyplot as plt

# Load the genuine and impostor scores from text files
genuine_scores = np.loadtxt('genuine_scores.txt')  # Ensure the file names are correct
impostor_scores = np.loadtxt('impostor_scores.txt')

# Define thresholds spanning the range of scores
thresholds = np.linspace(
    min(min(genuine_scores), min(impostor_scores)),
    max(max(genuine_scores), max(impostor_scores)),
    100
)

# Initialize FMR and FNMR lists
FMRs = []
FNMRs = []

# Calculate FMR and FNMR for each threshold
for threshold in thresholds:
    FMR = np.sum(impostor_scores >= threshold) / len(impostor_scores)  # False Match Rate
    FMRs.append(FMR)
    FNMR = np.sum(genuine_scores < threshold) / len(genuine_scores)  # False Non-Match Rate
    FNMRs.append(FNMR)

# Convert FMR and FNMR lists to NumPy arrays
FMRs = np.array(FMRs)
FNMRs = np.array(FNMRs)

# Find the optimal threshold where FMR and FNMR are closest
min_diff_index = np.argmin(np.abs(FMRs - FNMRs))
optimal_threshold = thresholds[min_diff_index]

# Plot FMR and FNMR against thresholds
plt.figure(figsize=(8, 6))
plt.plot(thresholds, FMRs, label='FMR (False Match Rate)', color='red')
plt.plot(thresholds, FNMRs, label='FNMR (False Non-Match Rate)', color='blue')
plt.axvline(optimal_threshold, color='green', linestyle='--', 
            label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.title('FMR and FNMR vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.legend()
plt.grid(True)
plt.show()

# Print the optimal threshold
print(f'Optimal Threshold: {optimal_threshold:.5f}')
