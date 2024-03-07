import os
import json
import torch
from PIL import Image
from torchvision.transforms import transforms as T
from pycocotools.coco import COCO

class CustomCOCODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        """
        Custom dataset for loading COCO-format annotated images for instance segmentation.
        
        Args:
        - image_dir (str): Directory where images are stored.
        - annotation_file (str): Path to the COCO format annotation JSON file.
        - transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Retrieve an image and its annotations (bounding box, masks) by the index.
        
        Args:
        - index (int): Index of the data to retrieve.
        
        Returns:
        - sample (dict): Contains the image, bounding boxes, labels, and masks.
        """
        coco = self.coco
        img_id = [self.ids[index]]
        ann_ids = coco.getAnnIds(img_id)
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Load the image
        # self.coco.loadImgs(id)[0]["file_name"]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")
        
        # Number of objects in the image
        num_objs = len(annotations)
        
        # Bounding boxes for objects
        # In COCO format, bbox = [xmin, ymin, width, height]
        boxes = []
        labels = []
        masks = []
        for i in range(num_objs):
            xmin = annotations[i]['bbox'][0]
            ymin = annotations[i]['bbox'][1]
            xmax = xmin + annotations[i]['bbox'][2]
            ymax = ymin + annotations[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotations[i]['category_id'])
            masks.append(coco.annToMask(annotations[i]))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks])
        
        image_id = torch.tensor(int(img_id[0]))
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.ids)

# Example usage
# Define the dataset
# dataset = CustomCOCODataset(image_dir="val", annotation_file="instances_val.json")



# To add transformations
# transformed_dataset = CustomCOCODataset(image_dir="path/to/images",
#                                         annotation_file="path/to/annotations.json",
#                                         transforms=T.Compose([
#                                             T.ToTensor(), # Convert the image to PyTorch tensor
#                                             # Add any other transformations here
#                                         ]))

from torch.utils.data import DataLoader

# Assuming your CustomCOCODataset class is already defined as shown previously
# Initialize your dataset
dataset = CustomCOCODataset(image_dir="val/",
                            annotation_file="instances_val.json",
                            transforms=T.Compose([
                                T.ToTensor(),  # Convert the image to PyTorch tensor
                                # Add any other transformations here
                            ]))

# Create the DataLoader with batch size of 1
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


import matplotlib.pyplot as plt
import numpy as np

def plot_image_with_boxes_masks(image, target):
    """
    Function to plot an image along with its bounding boxes and masks.
    
    Args:
    - image (torch.Tensor): The input image.
    - target (dict): Contains bounding boxes, labels, and masks.
    """
    image = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()
    boxes = target['boxes'][0]
    labels = target['labels'][0]
    masks = target['masks'][0]
    for i in range(len(boxes)):
        print(boxes[i])
        xmin, ymin, xmax, ymax = boxes[i]
        label = labels[i]
        mask = masks[i].numpy()
        # category = coco.loadCats(label)[0]['name']
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2))
        # ax.text(xmin, ymin, category, fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        plt.imshow(mask, alpha=0.4, cmap='jet', interpolation='none')
    plt.axis('off')
    plt.savefig("check.png")

    # Assuming your CustomCOCODataset class is already defined and instantiated as 'dataset'

# Example of iterating over the DataLoader
for images, targets in data_loader:
    for i in range(len(images)):
        plot_image_with_boxes_masks(images[i], targets)
