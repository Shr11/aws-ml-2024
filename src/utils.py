import re
import constants
import os
import requests
import pandas as pd
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import easyocr
import cv2

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

# initialize the ocr reader
def ocr_init(gpu=True):
    print("Initializing OCR")
    reader = easyocr.Reader(['en'], gpu=gpu)
    return reader
 
# clean text
def clean_text(text):
    """improving formatting of the extracted text"""
    # Removing special characters and standardizing common patterns
    text = re.sum(r'[^0-9a-zA-Z\s.,]+', '', text)  # Remove special characters
    text = text.replace("lbs", " lb").replace("in", " in").replace("cm"," cm") # standardize units
    return text
 
 # extract text from image   
def ocr_extract_and_parse_text(reader,image_folder, df , limit_rows=140000):
    text_data = {}
    
    for idx, img_path in enumerate(Path(image_folder).glob("*.jpg")):
        if idx >= limit_rows:
            break
        
        try:
            # extract text from image
            result = reader.readtext(str(img_path), detail=0, paragraph=True)
            text = ' '.join(result).strip() if result else ""
            
            # clean the text to try to fix invalid errors
            cleaned_text = clean_text(text)
            
            # parse the extracted text
            parsed_value, parsed_unit = parse_string(cleaned_text)
            
            # store the result by image_id
            text_data[img_path.stem] = (parsed_value, parsed_unit)  
            
        except ValueError as e:
            print(f"Error processing {img_path}: {e}")
            text_data[img_path.stem] = (None, None)
            
    ocr_df = pd.DataFrame.from_dict(text_data , orient='index', columns=['value', 'unit'])
        
    return ocr_df
    
def common_mistake(unit):
    # lower case and no trailing spaces
    unit = unit.lower().strip()
    
    corrections = {
        'grams': 'gram',
        'kilograms': 'kilogram',
        'ounces': 'ounce',
        'centimetres': 'centimetre',
        'inches': 'inch',
        # Add more corrections here
    }
    
    if unit in corrections:
        return corrections.get(unit,unit)
    
    if unit in constants.allowed_units:
        return unit
    
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    
    return unit

def parse_string(s):
    s_stripped = "" if s is None or str(s).lower() =='nan' else s.strip()
   
    if s_stripped == "":
        return None, None
    
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit


def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except:
            time.sleep(delay)
    
    create_placeholder_image(image_save_path) #Create a black placeholder image for invalid links/images

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(50) as pool:
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)
        
        
def main():
    # creating array
    df = pd.read_csv('D:/Desktop_Ddrive/aws_ml/dataset/train.csv')
    
    # array = df.iloc[:140000]['image_link'].to_numpy()
    
    # save_folder = "./images"
    
    # # downloading images
    # download_images(array,save_folder)
    
    reader  =  ocr_init(gpu=True)
    
    # # extract and parse text from image
    img_folder = "D:/Desktop_Ddrive/aws_ml/resized_images"
    ocr_df = ocr_extract_and_parse_text(reader,img_folder, df , limit_rows=140000)
    
    # Merge the OCR data with the input DataFrame, using the image_id to match
    df['parsed_value'], df['parsed_unit'] = zip(*df['image_id'].map(ocr_df.to_dict('index')).apply(lambda x: (x['parsed_value'], x['parsed_unit']) if x else (None, None)))

    print(df.head())
    
if __name__ == '__main__':
    main()