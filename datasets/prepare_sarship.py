import os
import glob

def generate_dir():
    splits_dir = "/home/sun/projects/detectron2/datasets/sar"
    train_split_f = os.path.join(splits_dir, "train" + '.txt')
    test_split_f = os.path.join(splits_dir, "test" + '.txt')
    traintest_split_f = os.path.join(splits_dir, "train_test" + '.txt')

    train_split = [str(i) for i in range(1, 201)]
    test_split = [str(i) for i in range(201, 301)]
    traintest_split = [str(i) for i in range(1, 301)]
    with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
        fp.write('\n'.join(train_split) + '\n')
    with open(os.path.join(splits_dir, test_split_f), 'w') as fp:
        fp.write('\n'.join(test_split) + '\n')
    with open(os.path.join(splits_dir, traintest_split_f), 'w') as fp:
        fp.write('\n'.join(traintest_split) + '\n')

def new_dataset_crop_500():
    dirsrc = "/home/sun/projects/sar"
    imglist = glob.glob(f'{dirsrc}/AIR-SARShip-1.0-data/*.tiff')
    imgnameList = [os.path.split(imgpath)[-1][: -5] for imgpath in imglist]
    imgnameList.sort()

    splits_dir = "/home/sun/projects/detectron2/datasets/sar"
    train_split_f = os.path.join(splits_dir, "train" + '.txt')
    test_split_f = os.path.join(splits_dir, "test" + '.txt')
    traintest_split_f = os.path.join(splits_dir, "train_test" + '.txt')

    train_split = [x for x in imgnameList if int(x.split('-')[-1].split('_')[0]) < 22]
    test_split = [x for x in imgnameList if int(x.split('-')[-1].split('_')[0]) >= 22]
    traintest_split = imgnameList
    with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
        fp.write('\n'.join(train_split) + '\n')
    with open(os.path.join(splits_dir, test_split_f), 'w') as fp:
        fp.write('\n'.join(test_split) + '\n')
    with open(os.path.join(splits_dir, traintest_split_f), 'w') as fp:
        fp.write('\n'.join(traintest_split) + '\n')
    return 

if __name__ == "__main__":
    # generate_dir()
    new_dataset_crop_500()