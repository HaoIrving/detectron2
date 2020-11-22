import os


if __name__ == "__main__":
    splits_dir = "/home/sun/projects/detectron2/datasets/sar"
    train_split_f = os.path.join(splits_dir, "train" + '.txt')
    val_split_f = os.path.join(splits_dir, "val" + '.txt')
    
    train_split = [str(i) for i in range(1, 201)]
    val_split = [str(i) for i in range(201, 301)]
    with open(os.path.join(splits_dir, train_split_f), 'w') as fp:
        fp.write('\n'.join(train_split) + '\n')
    with open(os.path.join(splits_dir, val_split_f), 'w') as fp:
        fp.write('\n'.join(val_split) + '\n')