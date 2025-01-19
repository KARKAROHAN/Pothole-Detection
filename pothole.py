import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
import random
import os
from torch import nn
import torchvision
import timm

from PIL import Image
from xml.dom import minidom
import csv
import torch
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder

import re
pattern = r"(\d+)\.jpg$"
img_numbers=[]

for dirname, _, filenames in os.walk('/kaggle/input/annotated-potholes-dataset/annotated-images'):
    for filename in filenames:
        match = re.search(pattern,filename,re.IGNORECASE)
        if match:
            number = match.group(1)
            img_numbers.append(number)

img_numbers.sort()

def extract_xml_contents(annot_directory, image_dir, file_dir):
        
        file = minidom.parse(annot_directory)

        # Get the height and width for our image
        height, width = cv2.imread(image_dir).shape[:2]

        # Get the bounding box co-ordinates 
        xmin = file.getElementsByTagName('xmin')
        x1 = int(xmin[0].firstChild.data)

        ymin = file.getElementsByTagName('ymin')
        y1 = int(ymin[0].firstChild.data)

        xmax = file.getElementsByTagName('xmax')
        x2 = int(xmax[0].firstChild.data)

        ymax = file.getElementsByTagName('ymax')
        y2 = int(ymax[0].firstChild.data)

        files = file.getElementsByTagName('filename')
        filename = files[0].firstChild.data
        filename = os.path.join(file_dir,filename)

        return filename, width, height, x1,y1,x2,y2

import pandas as pd

# Function to convert XML files to CSV
def xml_to_csv(data_dir):

  # List containing all our attributes regarding each image
    xml_list = []

  # Loop over each of the image and its label
    for i in img_numbers:
        
        mat = f"img-{i}.xml"
        image_file = f"img-{i}.jpg"
      
      # Full mat path
        mat_path = os.path.join(data_dir, mat)

      # Full path Image
        img_path = os.path.join(data_dir, image_file)

      # Get Attributes for each image 
        value = extract_xml_contents(mat_path, img_path,data_dir)

      # Append the attributes to the mat_list
        xml_list.append(value)

  # Columns for Pandas DataFrame
    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 
                 'xmax', 'ymax']

  # Create the DataFrame from mat_list
    xml_df = pd.DataFrame(xml_list, columns=column_name)

  # Return the dataframe
    return xml_df

# Run the function to convert all the xml files to a Pandas DataFrame
labels_df1 = xml_to_csv(data_dir="/kaggle/input/annotated-potholes-dataset/annotated-images")

# Saving the Pandas DataFrame as CSV File
labels_df1.to_csv(('dataset.csv'), index=None)

labels_df1

# Read the CSV file and rename the columns
labels_df2 = pd.read_csv("train/labels.csv")
labels_df2.columns = ['filename', 'LabelName', 'xmin', 'xmax', 'ymin', 'ymax']
labels_df2.drop("LabelName", axis=1, inplace=True)

# Add a new column with the image filename and path
labels_df2['filename'] = labels_df2['filename'].apply(lambda x: "train/images/" + x)

# Add new columns with the image height and width
heights, widths = [], []
for _, row in labels_df2.iterrows():
    height, width = cv2.imread(row['filename']).shape[:2]
    heights.append(height)
    widths.append(width)

labels_df2 = labels_df2.assign(height=heights, width=widths)

"""Print the resulting DataFrame
labels_df2"""

labels_df = pd.concat([labels_df1,labels_df2],axis=0)
#labels_df



class PotholeDataset(torch.utils.data.Dataset):
    def __init__(self,df,augs=None):
        self.df = df
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        xmin = row.xmin 
        ymin = row.ymin 
        xmax = row.xmax 
        ymax = row.ymax  
        bbox = [[xmin,ymin,xmax,ymax]]

        img_path = row.filename
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.augs:
            data = self.augs(image = img, bboxes=bbox, class_labels = [None])
            img = data["image"]
           
            bbox = data["bboxes"][0]

        img = torch.from_numpy(img).permute(2,0,1) / 255.0
        bbox = torch.Tensor(bbox)

        return img, bbox

MODEL_NAME = "res2net50d.in1k"

class PotholeModel(nn.Module):
    def __init__(self) -> None:
        super(PotholeModel,self).__init__()

        self.backbone = timm.create_model(MODEL_NAME,pretrained=True,num_classes=4)


    def forward(self,images,gt_bboxes=None):
        predBboxes = self.backbone(images)

        if gt_bboxes != None:
            loss1 = torchvision.ops.complete_box_iou_loss(predBboxes,gt_bboxes,reduction="sum")
            loss2 = nn.functional.smooth_l1_loss(predBboxes,gt_bboxes)
            return predBboxes,loss2 + loss1

        return predBboxes