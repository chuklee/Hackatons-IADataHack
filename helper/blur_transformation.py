import os
import random
import shutil
import time
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import PIL.Image as Image
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from captum.attr import Occlusion
from captum.attr import visualization as viz
from model import MyModel
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision import datasets, transforms
from torchvision.io.image import read_image
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import normalize, resize, to_pil_image
from tqdm import tqdm


def generate_blur_image_based_on_heatmap(
    model_generic: MyModel,
    path: str,
    kernel_size: Tuple[int, int] = (21, 21),
    sigma: int = 10,
    threshold: int = 10,
):

    loader = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    image = Image.open(path)
    img = loader(image).float()
    img = torch.autograd.Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    img = img.cuda()

    with GradCAMpp(model_generic.model) as cam_extractor:
        out = model_generic.model(img)
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    result = overlay_mask(
        image, to_pil_image(activation_map[0].squeeze(0), mode="F"), alpha=0.0001
    )

    img_np = np.array(result)
    red_mask = (img_np[:, :, 0] > img_np[:, :, 1] + threshold) & (
        img_np[:, :, 0] > img_np[:, :, 2] + threshold
    )
    red_mask = red_mask.astype(int)
    mask_img = Image.fromarray(np.uint8(red_mask * 255), "L")

    # Convert the original PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply a Gaussian blur to the entire image
    blurred_img_cv = cv2.GaussianBlur(img_cv, kernel_size, sigma)

    blurred_img_pil = Image.fromarray(cv2.cvtColor(blurred_img_cv, cv2.COLOR_BGR2RGB))

    # Use the mask to blend the blurred and original images
    result_img = Image.composite(blurred_img_pil, image, mask_img)

    # enregistre les images dans un path specifique

    path = path.split(".")

    if not path[0].endswith("_blur"):
        path[0] = path[0] + "_blur"
        path = ".".join(path)

    result_img.save(path)


#     Display the result
#     plt.imshow(result_img)
#     plt.title('Image with Blurred Red Areas')
#     plt.axis('off')
#     plt.show()


def process_images_in_folder(
    model: MyModel, folder_path: str, processing_percentage: int = 20
):
    # Get a list of all files in the folder
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            image_path = os.path.join(root, filename)
            all_files.append(image_path)

    # Shuffle the list of files
    random.shuffle(all_files)

    # Determine the number of files to process based on the percentage
    num_files_to_process = int(len(all_files) * (processing_percentage / 100.0))

    # Iterate through a subset of files based on the percentage
    for image_path in tqdm(all_files[:num_files_to_process]):
        if not image_path.endswith("_blur.png"):
            try:
                generate_blur_image_based_on_heatmap(model, image_path)
            except Exception as e:
                print(e)
                pass


if __name__ == "__main__":
    classes = pd.read_csv("../dataset/names.csv", names=["name"])
    model = MyModel(list(classes.name))
    path_dataset = os.path.abspath("../dataset/car_data/car_data/train")
    process_images_in_folder(model, path_dataset, processing_percentage=20)
