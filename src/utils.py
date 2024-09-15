import os
import re
import time
import cv2
import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import xgboost as xgb
import pytesseract
import multiprocessing
from io import BytesIO

# ---------------------- Image Preprocessing Functions ----------------------

def rescale_image(image, scale_percent=150):
    """Rescale input image by the specified percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

def enhance_contrast(image):
    """Enhance image contrast using histogram equalization."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def binarize_image(image):
    """Convert image to binary format using adaptive thresholding."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def remove_noise(image):
    """Remove noise from the image using median filtering."""
    return cv2.medianBlur(image, 3)

# ----------------------- OCR Functions -----------------------

def extract_text(image_path):
    """Perform OCR to extract text from the image."""
    if not image_path or not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        return ""

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image file: {image_path}")
            return ""

        # Preprocess the image for OCR
        image = rescale_image(image)
        image = enhance_contrast(image)
        image = binarize_image(image)
        image = remove_noise(image)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def batch_ocr(image_paths, num_threads=4):
    """Perform OCR on a batch of images using multiple threads."""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        ocr_results = list(tqdm(executor.map(extract_text, image_paths), total=len(image_paths)))
    return ocr_results

# ----------------------- Utility Functions -----------------------

def download_image(image_link, save_folder, retries=3, delay=3):
    """Download image from link and handle failures gracefully."""
    if not isinstance(image_link, str) or not image_link.startswith("http"):
        print(f"Invalid URL: {image_link}")
        return None

    # Use the filename from the URL to save the image
    image_save_path = os.path.join(save_folder, Path(image_link).name)

    # If the image already exists locally, skip downloading
    if os.path.exists(image_save_path):
        return image_save_path  # Image already exists

    for attempt in range(retries):
        try:
            # Fetch the image content from the URL
            response = requests.get(image_link, stream=True, timeout=5)
            if response.status_code == 200:
                # Open the image from the response content and save it locally
                image = Image.open(BytesIO(response.content))
                image.save(image_save_path)
                return image_save_path
            else:
                print(f"Failed to download {image_link}, status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            time.sleep(delay)

    # If the download fails after retries, return None and skip
    print(f"Failed to download image after {retries} retries: {image_link}")
    return None

def download_images(image_links, download_folder, use_multiprocessing=True, num_processes=4):
    """Download images in parallel or sequentially, with improved error handling."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_partial = partial(download_image, save_folder=download_folder)
    
    if use_multiprocessing:
        # Use a smaller number of processes to avoid overwhelming the system
        with multiprocessing.Pool(num_processes) as pool:
            image_paths = list(tqdm(pool.imap(download_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        # Download images sequentially
        image_paths = []
        for image_link in tqdm(image_links, total=len(image_links)):
            image_path = download_image(image_link, download_folder)
            image_paths.append(image_path)

    return image_paths
            

# ----------------------- Data Preprocessing Functions -----------------------

def preprocess_train_data(train_csv, download_folder, sample_size=30000, test_size=0.2, num_threads=4):
    """Preprocess and sample training data."""
    train_df = pd.read_csv(train_csv)

    if sample_size < len(train_df):
        train_sample_df = train_df.sample(n=sample_size, random_state=42)
    else:
        train_sample_df = train_df

    # Download images from URLs and perform OCR
    image_paths = download_images(train_sample_df['image_link'].tolist(), download_folder, num_processes=num_threads)
    ocr_results = batch_ocr(image_paths, num_threads=num_threads)

    # Process OCR results and store parsed number/unit
    train_sample_df['parsed_number'], train_sample_df['parsed_unit'] = zip(*[parse_string(text.strip()) for text in ocr_results])
    train_sample_df.fillna(0, inplace=True)
    train_sample_df, val_df = train_test_split(train_sample_df, test_size=test_size, random_state=42)

    return train_sample_df, val_df

def preprocess_test_data(test_csv, download_folder, sample_size=500, num_threads=4):
    """Preprocess test data without augmentation."""
    test_df = pd.read_csv(test_csv)

    if sample_size < len(test_df):
        test_df = test_df.sample(n=sample_size, random_state=42)

    # Download images from URLs and perform OCR
    image_paths = download_images(test_df['image_link'].tolist(), download_folder, num_processes=num_threads)
    ocr_results = batch_ocr(image_paths, num_threads=num_threads)

    test_df['parsed_number'], test_df['parsed_unit'] = zip(*[parse_string(text.strip()) for text in ocr_results])
    test_df.fillna(0, inplace=True)
    return test_df


# ----------------------- XGBoost Model Training -----------------------

def train_xgboost_model(train_df, val_df):
    """Train an XGBoost model using the parsed data."""
    X_train = train_df[['parsed_number']]
    y_train = train_df['entity_value']
    X_val = val_df[['parsed_number']]
    y_val = val_df['entity_value']

    model = xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
    print(f"Validation RMSE: {val_rmse:.4f}")
    
    return model

# ----------------------- Main Execution Pipeline -----------------------


def main(test_csv, train_csv, save_dir, sample_size=30000, num_threads=4):
    """Main pipeline to execute the full process."""
    print("Preprocessing training data...")
    train_df, val_df = preprocess_train_data(train_csv, download_folder=save_dir, sample_size=sample_size, num_threads=num_threads)

    print("Preprocessing test data...")
    test_df = preprocess_test_data(test_csv, download_folder=save_dir, num_threads=num_threads)

    print("Training XGBoost model...")
    model = train_xgboost_model(train_df, val_df)

    print("Making predictions on test data...")
    X_test = test_df[['parsed_number']]
    test_df['predictions'] = model.predict(X_test)

    print("Pipeline completed.")
    return train_df, val_df, test_df

# Run if executed directly
if __name__ == "__main__":
    TEST_CSV_PATH = "D:/Desktop_Ddrive/aws_ml/dataset/test.csv"
    TRAIN_CSV_PATH = "D:/Desktop_Ddrive/aws_ml/dataset/train.csv"
    SAVE_DIR = "./images"
    SAMPLE_SIZE = 30000  # Define how many rows to sample from the training dataset
    
    # Execute the pipeline
    train_df, val_df, test_df = main(TEST_CSV_PATH, TRAIN_CSV_PATH, SAVE_DIR, SAMPLE_SIZE)
