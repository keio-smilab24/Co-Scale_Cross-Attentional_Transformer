import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset

EXTENSIONS = ['jpg','.png']


def load_txt_file_and_convert_to_list(dataset_txt_file_path: str):
    with open(dataset_txt_file_path, "r", encoding="UTF-8") as dataset_txt_file:
        dataset_txt_file_readlines = dataset_txt_file.readlines()
    dataset_list = [dataset_line.replace("\n", "") for dataset_line in dataset_txt_file_readlines]

    return dataset_list


def check_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def get_img_path(root, basename, extension):
    return os.path.join(root, basename+extension)


def get_img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class ai2thor(Dataset):

    def __init__(self, root):
        super(ai2thor, self).__init__()
        self.img_t0_root = os.path.join(root, 'goal')
        self.img_t1_root = os.path.join(root, 'current')
        self.mask_root = os.path.join(root, 'mask')

        self.filenames = [get_img_basename(f) for f in os.listdir(self.mask_root) if check_img(f)]
        self.filenames.sort()

        print('{}:{}'.format(root,len(self.filenames)))

    def __getitem__(self, index):
        filename = self.filenames[index]

        fn_img_t0 = get_img_path(self.img_t0_root, filename, '.png')
        fn_img_t1 = get_img_path(self.img_t1_root, filename, '.png')
        fn_mask = get_img_path(self.mask_root, filename, '.png')

        if os.path.isfile(fn_img_t0) == False:
            print ('Error: File Not Found: ' + fn_img_t0)
            exit(-1)
        if os.path.isfile(fn_img_t1) == False:
            print ('Error: File Not Found: ' + fn_img_t1)
            exit(-1)
        if os.path.isfile(fn_mask) == False:
            print ('Error: File Not Found: ' + fn_mask)
            exit(-1)

        img_t0 = cv2.imread(fn_img_t0, cv2.IMREAD_COLOR)
        img_t1 = cv2.imread(fn_img_t1, cv2.IMREAD_COLOR)
        mask = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
        w,h,_ = img_t0.shape
        r = 256./min(w,h)
        img_t0 = cv2.resize(img_t0, (int(r*w), int(r*h)))
        img_t1 = cv2.resize(img_t1, (int(r*w), int(r*h)))
        mask = cv2.resize(mask, (int(r*w), int(r*h)))[:,:,np.newaxis]

        img_t0_ = np.asarray(img_t0).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_ = np.asarray(img_t1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        mask_ = np.asarray(mask>128).astype("int").transpose(2, 0, 1)

        input_ = torch.from_numpy(np.concatenate((img_t0_, img_t1_), axis=0))
        mask_ = torch.from_numpy(mask_).long()

        return input_, mask_

    def __len__(self):
        return len(self.filenames)

    def get_random_index(self):
        index = np.random.randint(0, len(self.filenames))
        return index
