Face Recognition with CFPW Dataset: Preprocessing, Feature Extraction, and Performance Analysis
Project Description
This project focuses on applying face detection, preprocessing, feature extraction, and performance analysis techniques using the Celebrities in Frontal-Profile Wild (CFPW) dataset(You can download the data-set here http://cfpw.io/). Specifically, it utilizes only the frontal images from the dataset to explore face recognition methodologies and evaluate their effectiveness. The primary objective is to detect faces, preprocess images, extract features using a pre-trained CNN model, compute similarity scores, and analyze performance through various metrics and visualizations.

Achievements
Dataset Preparation:

Downloaded and utilized the CFPW dataset with frontal images.
Renamed the images in a structured format for easier identification.
Applied helper scripts and custom code to facilitate the preprocessing pipeline.
Face Detection and Preprocessing:

Implemented face detection and alignment to standardize images.
Cropped and resized all detected faces to 112x112 pixels for consistency.
Explored different face detection methods, including RetinaFace and MediaPipe.
Feature Extraction and Matching:

Leveraged the DeepFace library for feature extraction using pre-trained CNN models such as ArcFace and Facenet512.
Calculated cosine similarity between feature vectors to measure match scores.
Generated and saved a similarity matrix in CSV format for further analysis.
Performance Analysis:

Analyzed genuine and impostor score distributions through visualizations.
Calculated and interpreted the d-prime value to measure system discriminability.
Plotted the ROC curve to evaluate the trade-off between true positive and false positive rates.
Visualized the relationship between False Match Rate (FMR) and False Non-Match Rate (FNMR).
Error Analysis:

Identified images where face detection failed and reported the failure rate.
Showcased examples of the top 3 impostor pairs with the highest scores and the bottom 3 genuine pairs with the lowest scores.
Deliverables:

Submitted all relevant code and plots for face detection, preprocessing, feature extraction, and performance analysis.
Highlighted key insights and challenges in a detailed report.
Key Tools and Libraries
DeepFace: Used for feature extraction with pre-trained CNN models.
Matplotlib and Seaborn: For generating score distributions, ROC curves, and FMR vs. FNMR plots.
OpenCV: For image preprocessing tasks, including face detection and alignment.
Pandas: For handling similarity matrices and data analysis.
This repository demonstrates a comprehensive workflow for facial recognition research, emphasizing preprocessing, feature extraction, and performance evaluation. The project also provides insights into system accuracy and areas of improvement through error analysis.
