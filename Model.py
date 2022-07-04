from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch
import timm

class Siamese(nn.Module):
    def __init__(self, pretrained=1):
        super(Siamese, self).__init__()
        self.model = timm.create_model(model_name='swin_base_patch4_window7_224_in22k', pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.input_dim = 1024
        self.feature_dim = 2048
        self.fc1 = nn.Linear(self.input_dim, self.feature_dim)
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc3 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc4 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc5 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc6 = nn.Linear(self.feature_dim, self.feature_dim)
        self.classifier1 = nn.Linear(self.input_dim, 1)
        self.classifier2 = nn.Linear(self.feature_dim, 1)
        self.classifier3 = nn.Linear(self.feature_dim, 1)
        self.classifier4 = nn.Linear(self.feature_dim, 1)
        self.classifier5 = nn.Linear(self.feature_dim, 1)
        self.output = nn.Linear(5, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
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

        out1 = self.classifier1(x_diff1)
        out2 = self.classifier2(x_diff2)
        out3 = self.classifier3(x_diff3)
        out4 = self.classifier4(x_diff4)
        out5 = self.classifier5(x_diff5)

        out = torch.cat((out1, out2, out3, out4, out5), 1)
        out = self.sig(self.output(out))

        return out

if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    y = torch.randn(4, 3, 224, 224)

    model = Siamese()
    print(model(x, y).shape)
    # summary(model, (3, 224, 224), device='cpu')