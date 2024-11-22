import numpy as np
import matplotlib.pyplot as plt

# Load genuine and impostor scores from text files
genuine_scores = np.loadtxt('genuine_scores.txt')  # Corrected file name
impostor_scores = np.loadtxt('impostor_scores.txt')  # Corrected file name

# Calculate the length of the scores
L1 = len(genuine_scores)
L0 = len(impostor_scores)

# Calculate AUC
count = 0
for s_g in genuine_scores:
    count += np.sum(s_g > impostor_scores)
AUC = count / (L0 * L1)
print(f'Calculated AUC: {AUC:.5f}')

# Define thresholds for FAR and FRR calculations
thresholds = np.linspace(
    min(min(genuine_scores), min(impostor_scores)),
    max(max(genuine_scores), max(impostor_scores)),
    10
)

# Initialize lists to store FAR and FRR values
FARs = []
FRRs = []

# Calculate FAR and FRR for each threshold
for eta in thresholds:
    FAR = np.sum(impostor_scores >= eta) / L0  # False Acceptance Rate
    FARs.append(FAR)
    FRR = np.sum(genuine_scores < eta) / L1  # False Rejection Rate
    FRRs.append(FRR)

# Print thresholds, FAR, and FRR values
print("\nThresholds, FAR, and FRR values:")
for i, eta in enumerate(thresholds):
    print(f'Threshold: {eta:.3f}, FAR: {FARs[i]:.3f}, FRR: {FRRs[i]:.3f}')

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(FARs, 1 - np.array(FRRs), label=f'ROC curve (AUC = {AUC:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate (FAR)')
plt.ylabel('True Positive Rate (1 - FRR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Plot FAR and FRR vs. Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, FARs, label="FAR", color="red")
plt.plot(thresholds, FRRs, label="FRR", color="blue")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("FAR and FRR vs. Threshold")
plt.legend()
plt.grid(True)
plt.show()
