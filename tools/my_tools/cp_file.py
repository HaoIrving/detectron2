import os
import shutil
from tqdm import tqdm

_base_dir = os.path.join('VOCdevkit_sar', "log_normal_new_noise_c1")
_voc_root = os.path.join("/home/sun/contest/data_preprocessed/", _base_dir)
_mask_dir = os.path.join(_voc_root, 'SegmentationClass')
_image_dir = os.path.join(_voc_root, 'JPEGImages')
_splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
_split_f = os.path.join(_splits_dir, 'trainval.txt')

# source = '/home/sun/contest/data_preprocessed/VOCdevkit_sar/log_normal_new_noise_c1/ImageSets/Segmentation/val.txt'
target = '/home/sun/contest/data_preprocessed/postprocessing'
# assert not os.path.isabs(_split_f)
# target = os.path.join(target, os.path.dirname(source))
# create the folders if not already exists
os.makedirs(target, exist_ok=True)

with open(os.path.join(_split_f), "r") as lines:
	for line in tqdm(lines):
		basename = line.rstrip('\n')
		basename = str(int(basename) + 1)
		_mask = os.path.join("/home/sun/contest/data_preprocessed/groundtruth", basename + "_gt.png")
		_pred = os.path.join("/home/sun/contest/PyTorch-Encoding/outdir", basename + "_visualize.png")
		assert os.path.isfile(_mask)
		# adding exception handling
		try:
			shutil.copy(_mask, target)
			shutil.copy(_pred, target)
		except IOError as e:
			print("Unable to copy file. %s" % e)
		except:
			print("Unexpected error:", sys.exc_info())


