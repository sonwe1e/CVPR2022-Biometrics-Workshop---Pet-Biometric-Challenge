import random
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import glob
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

img1_transformer = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
])
img2_transformer = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
])


class testdatset(Dataset):
    def __init__(self, path, transformer=None):
        self.path = path
        self.transformer = transformer
        self.data = pd.read_csv(self.path + "test_data.csv", sep=',', names=["imageA", "imageB"])[1:]
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index + 1
        imageA = cv2.imread(self.path + "images/" + self.data["imageA"][index], cv2.COLOR_BGR2RGB)
        imageB = cv2.imread(self.path + "images/" + self.data["imageB"][index], cv2.COLOR_BGR2RGB)
        augementedA = self.transformer(image=imageA)
        augementedB = self.transformer(image=imageB)
        return augementedA['image'], augementedB['image'], self.data["imageA"][index], self.data["imageB"][index]

class Simdataset(Dataset):
    def __init__(self, path, mode='train', img1_transformer=img1_transformer, img2_transformer=img2_transformer):
        self.path = path # 文件的路径
        # 从 csv 读取得到的数据
        self.data = pd.read_csv(self.path + "train_data.csv", sep=',', names=["id", "image"])[1:]
        # 针对两张图片的数据增强
        self.img1_transformer = img1_transformer
        self.img2_transformer = img2_transformer
        self.image_path = glob.glob(self.path + "images/*.jpg")
        # image_name 和 id 分别为图片名字和序号
        self.image_name = self.data["image"]
        self.image_id = self.data["id"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = index + 1
        # 随机生成图片的索引
        get_same_class = random.randint(0, 8)
        # 取出图片对应的 RGB 和编号
        img_1 = self.image_name[index]
        img_1_id = self.image_id[index]
        # 读取图片
        image1 = cv2.imread(self.path + 'images/' + img_1, cv2.COLOR_BGR2RGB)
        # 图片增强
        augmented1 = self.img1_transformer(image=image1)
        if get_same_class == 0: # 如果是同类
            # 定义标签
            label = 1
            label = torch.tensor(label).unsqueeze(0)
            # 得到当前图片的索引
            img_index = self.data[self.data["image"] == img_1].index.to_list()
            # 得到当前图片所有类别的索引
            total_index_list = self.data[self.data["id"] == img_1_id].index.to_list()
            # 将当前图片的索引从所有索引中去除
            total_index_list.remove(img_index[0])
            # 从备选索引中随机选取一个
            img_2 = self.data["image"][random.choice(total_index_list)]
            # 读取图片并做数据增强
            image2 = cv2.imread(self.path + 'images/' + img_2, cv2.COLOR_BGR2RGB)
            augmented2 = self.img2_transformer(image=image2)
            return augmented1["image"].float(), augmented2["image"].float(), label.float()
        if get_same_class > 0:
            # 得到与当前图片不同类别的索引
            # 定义标签
            label = 0
            label = torch.tensor(label).unsqueeze(0)
            total_index_list = self.data[self.data["id"] != img_1_id].index.to_list()
            # 从备选索引中随机选取一个
            img_2 = self.data["image"][random.choice(total_index_list)]
            # 读取图片并做数据增强
            image2 = cv2.imread(self.path + 'images/' + img_2, cv2.COLOR_BGR2RGB)
            augmented2 = self.img1_transformer(image=image2)
            return augmented1["image"].float(), augmented2["image"].float(), label.float()


if __name__ == '__main__':
    path = "/home/gdut403/sonwe1e/dogpet-master/pet_biometric_challenge_2022/train/"
    dataset = Simdataset(path, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    count = 0
    for i, (img1, img2, label) in enumerate(dataloader):
        # print(img1.shape,img2.shape)
        print(label)
        break
    # imag_list = glob.glob(path+"images/*.jpg")
    # print(len(imag_list))
