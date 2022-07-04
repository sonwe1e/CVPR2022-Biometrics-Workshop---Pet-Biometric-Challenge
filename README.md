# CVPR2022-Biometrics-Workshop---Pet-Biometric-Challenge
CVPR2022 Biometrics Workshop - Pet Biometric Challenge  
该比赛提供多只狗不同条件下拍摄的鼻子的照片，希望能够如人的指纹一样识别狗，但这是第一次识别类比赛，所以使用的方法很中规中矩 

# 比赛成绩
初赛 49/646  
复赛 31/646

# 比赛细节
## 网络架构
利用预训练的swin_base_patch4_window7_224_in22k对两张狗鼻子提取特征，在原来只有L1距离的情况下增加了多层的全连接层进行输出  
1. 修改原本特征的L1距离为特征之间差的平方
2. 多层全连接分别为相同全连接-不同全连接-相同全连接-相同全连接，最后将五个特征的差值concatenate通过全连接层输出概率，具体可以从下代码理解
‘’‘
        x11 = self.model.forward_features(x1)
        x21 = self.model.forward_features(x2)

        x_diff1 = torch.square(x11 - x21)

        x12 = self.fc1(x11)
        x22 = self.fc1(x21)
        x_diff2 = torch.square(x12 - x22)

        x13 = self.fc2(x12)
        x23 = self.fc3(x22)
        x_diff3 = torch.square(x13 - x23)

        x14 = self.fc4(x13)
        x24 = self.fc4(x23)
        x_diff4 = torch.square(x14 - x24)

        x15 = self.fc5(x14)
        x25 = self.fc5(x24)
        x_diff5 = torch.square(x15 - x25)
’‘’
## 数据集
数据输入到网络有两种方式，一种是利用
数据输入到网络有两种方式，一种是Tr
