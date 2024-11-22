import numpy as np
genuine_scores = np.loadtxt('genuine_scores.txt')
impostor_scores = np.loadtxt('impostor_scores.txt')
mean_genuine = np.mean(genuine_scores)
mean_impostor = np.mean(impostor_scores)
std_genuine = np.std(genuine_scores)
std_impostor = np.std(impostor_scores)
d_prime = (np.sqrt(2) * abs(mean_genuine - mean_impostor)) 
np.sqrt(std_genuine**2 + std_impostor**2)
print(f'D-prime value: {d_prime:.5f}')