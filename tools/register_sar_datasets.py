# register a new dataset
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

CLASS_NAMES = ("ship", )

def load_sarship_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "AIR-SARShip-1.0-xml/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        tiff_file = os.path.join(dirname, "AIR-SARShip-1.0-data", fileid + ".tiff")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": tiff_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts
    
def register_sar_ship(name, dirname, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_sarship_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split, evaluator_type="coco"
    )

def show_images():
    random.seed(3)
    dataset_dicts = load_sarship_instances("datasets/sar/", "train", "ship")
    register_sar_ship("sarship", "datasets/sar/", "train")
    sar_metadata = MetadataCatalog.get("sarship")
    for d in random.sample(dataset_dicts, 5):
        # img = cv2.imread(d["file_name"], -1)
        # tif = TIFF.open(d["file_name"], mode='r')
        # img = tif.read_image()
        img = imageio.imread(d["file_name"])

        pixel_max = img.max()
        pixel_min = img.min()
        
        # img = img * 1.0 / (pixel_max - pixel_min) * 255

        k = pixel_max ** (1 / 255)
        img = np.clip(img, 1, None)
        img = np.log(img) / np.log(k)

        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)

        visualizer = Visualizer(img, metadata=sar_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        # cv2.imshow('imshow', out.get_image()[:, :, ::-1])
        plt.imshow(out.get_image())
        plt.show()
    return

class Batch_loader:
    def __init__(self, datalist):
        self.datalist = datalist

    def __getitem__(self, index):
        d = self.datalist[index]
        image = imageio.imread(d["file_name"])
        pixel_max = image.max() 
        pixel_min = image.min()
 
        # [1.45497722, 1.45497722, 1.45497722] [3.7081214, 3.7081214, 3.7081214] [0.06462957 0.06491809 0.06350067]
        # image = image * 1.0 / (pixel_max - pixel_min) * 255  

        # [78.11523, 78.11523, 78.11523] [57.375, 57.120, 58.395] [0.53109455 0.53346551 0.52181779]
        k = pixel_max ** (1 / 255)
        image = np.clip(image, 1, None)
        image = np.log(image) / np.log(k)
        
        image = image[np.newaxis, :, :]
        # image = np.concatenate((image, image, image), axis=0)
        return torch.from_numpy(image).float()
    
    def __len__(self):
        return len(self.datalist)

def get_mean_std():
    """Get mean and std by sample ratio
    """
    ratio=1
    dataset_dicts = load_sarship_instances("datasets/sar/", "train_test", "ship")
    dataset = Batch_loader(dataset_dicts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=False, num_workers=0)
    train = iter(dataloader).next()   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    print(mean, std)
    imagenet_std = np.array([57.375, 57.120, 58.395])
    cofficient = std / imagenet_std 
    print(cofficient)
    return 


if __name__ == "__main__":
    import cv2, random
    from detectron2.utils.visualizer import Visualizer
    import matplotlib.pyplot as plt
    import imageio
    import torch 
    from libtiff import TIFF

    # show_images()
    get_mean_std()

    # BGR order
    # mean = [78.11523, 78.11523, 78.11523]
    # std = np.array([57.375, 57.120, 58.395])
    # sar_std = np.array([3.7081214, 3.7081214, 3.7081214])
    # cofficient = sar_std / std # [0.53109455 0.53346551 0.52181779] [0.06462957 0.06491809 0.06350067]
    # print(cofficient)

    