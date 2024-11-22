import os
import cv2
import dlib
import shutil
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# Load the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("G:/Master Material/Biometrics/assignment5/biometrics_assignment-main/shape_predictor_68_face_landmarks.dat")

def align_face(image, landmarks):
    """Align a face based on eye coordinates."""
    left_eye_pts = [36, 37, 38, 39, 40, 41]
    right_eye_pts = [42, 43, 44, 45, 46, 47]

    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_pts], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_pts], axis=0)

    # Calculate rotation angle
    eye_delta_x = right_eye[0] - left_eye[0]
    eye_delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(eye_delta_y, eye_delta_x))

    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_image

def preprocess_and_save(image_path, output_path):
    """Preprocess an image: detect, align, crop, and resize."""
    image = cv2.imread(image_path)
    if image is None:
        return False  # Image load failed

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        return False  # No faces detected

    face = faces[0]
    landmarks = predictor(gray, face)
    aligned_image = align_face(image, landmarks)

    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cropped_face = aligned_image[y:y+h, x:x+w]

    if cropped_face.size == 0:
        return False  # Invalid crop

    resized_face = cv2.resize(cropped_face, (112, 112))
    cv2.imwrite(output_path, resized_face)
    return True

def process_images(src_folder, dest_folder, fail_folder):
    """Process images from the 'frontal' subfolders and save failures."""
    failed_count = 0
    image_files = []

    # Look for images only in 'frontal' subfolders
    for indv_id in os.listdir(src_folder):
        frontal_folder = os.path.join(src_folder, indv_id, 'frontal')
        if os.path.isdir(frontal_folder):
            for file in os.listdir(frontal_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append((indv_id, os.path.join(frontal_folder, file)))

    if not image_files:
        print("No images found in the frontal subfolders.")
        return failed_count

    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for indv_id, img_path in image_files:
            img_file = os.path.basename(img_path)
            unique_filename = f"{indv_id}_{img_file}"  # Include individual ID in filename
            output_path = os.path.join(dest_folder, unique_filename)

            if not preprocess_and_save(img_path, output_path):
                failed_count += 1
                fail_output_path = os.path.join(fail_folder, unique_filename)
                shutil.copy(img_path, fail_output_path)

            pbar.update(1)
    return failed_count

def main():
    """Main function to process images."""
    base_path = os.getcwd()  # Current working directory
    src_folder = "G:/Master Material/Biometrics/assignment5/cfp-dataset/Data/Images"  # Update source folder
    dest_folder = os.path.join(base_path, 'Pre_processed_dataset')
    fail_folder = os.path.join(base_path, 'Failed_Images')

    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(fail_folder, exist_ok=True)

    failed_images = process_images(src_folder, dest_folder, fail_folder)
    print(f"Failed to process {failed_images} images.")

# Run the script
if __name__ == "__main__":
    main()
