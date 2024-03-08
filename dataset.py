import os
import json
import torch
from PIL import Image
from torchvision.transforms import transforms as T
from pycocotools.coco import COCO
from generalized_dataset import GeneralizedDataset

import pycocotools.mask as mask_util
from pycocotools.cocoeval import COCOeval
import copy
import numpy as np

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

class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, json_file="", train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.train = train
        
        ann_file = os.path.join(data_dir, json_file)
        self.coco = COCO(ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}

        
    def get_image(self, img_id):
        # img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "val", img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        # img_id = int(img_id)
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([int(img_id)]), boxes=boxes, labels=labels, masks=masks)
        return target

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types="bbox"):
        if isinstance(iou_types, str):
            iou_types = [iou_types]
            
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        #self.ann_labels = ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types}
        
        self.has_results = False
            
    def accumulate(self, coco_results): # input all predictions
        if len(coco_results) == 0:
            return
        
        image_ids = list(set([res["image_id"] for res in coco_results]))
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = self.coco_gt.loadRes(coco_results) # use the method loadRes
            coco_eval.params.imgIds = image_ids # ids of images to be evaluated
            coco_eval.evaluate() # 15.4s
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            coco_eval.accumulate() # 3s
            
        self.has_results = True
    
    def summarize(self):
        if self.has_results:
            for iou_type in self.iou_types:
                print("IoU metric: {}".format(iou_type))
                self.coco_eval[iou_type].summarize()
        else:
            print("evaluation has no results")
            
            
def prepare_for_coco(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        masks = prediction["masks"]

        x1, y1, x2, y2 = boxes.unbind(1)
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
        boxes = boxes.tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        masks = masks > 0.5
        rles = [
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[i],
                    "bbox": boxes[i],
                    "segmentation": rle,
                    "score": scores[i],
                }
                for i, rle in enumerate(rles)
            ]
        )
    return coco_results   


def collate_fn(batch):
    """
    Custom collate function for handling batches with varying number of objects.
    
    Args:
    - batch: List of tuples (image, target) from the dataset.
    
    Returns:
    - batched_images: Tensor of images stacked along the first dimension.
    - batched_targets: List of dictionaries, one for each image.
    """
    batched_images = torch.stack([item[0] for item in batch])
    batched_targets = [item[1] for item in batch]
    
    return batched_images, batched_targets
if __name__=="__main__":
    # dataset = COCODataset("./", json_file="instances_val.json", train=True)
    # print(len(dataset))
    # indices = torch.randperm(len(dataset)).tolist()
    # d_train = torch.utils.data.Subset(dataset, indices)

    # num_classes = max(d_train.dataset.classes) + 1 # including background class
    # print(num_classes)
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
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Example of iterating over the DataLoader
    for images, targets in data_loader:
        # Your training or validation code here
        print(images.shape)
        for target in targets:
            print(target["boxes"])