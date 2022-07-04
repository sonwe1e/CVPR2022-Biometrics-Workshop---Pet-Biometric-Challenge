import csv
import torch
import torch.nn as nn
import argparse
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Model import Siamese
import pytorch_lightning as pl
from pytorch_lightning import loggers
from simdataset import Simdataset, testdatset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class pointclassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.classifier = Siamese()
        self.lr = args.lr
        self.cretirion = nn.BCELoss()

    def forward(self, x1, x2):
        out = self.classifier(x1, x2)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2500, T_mult=2)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x1, x2, y = train_batch
        out = self.classifier(x1, x2)
        loss4 = self.cretirion(out, y)
        self.log('train_sigsim_loss/bce_loss', loss4)
        return loss4

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y = val_batch
        out = self.classifier(x1, x2)
        loss4 = self.cretirion(out, y)
        self.log('valid_sigsim_loss/bce_loss', loss4)

    def predict_step(self, val_batch, batch_idx):
        file = open(args.results_dir + 'test.csv', 'a')
        writer = csv.writer(file)
        if batch_idx == 0:
            file = open(args.results_dir + 'test.csv', 'w')
            writer = csv.writer(file)
            writer.writerow(['imageA', 'imageB', 'prediction'])
        x1, x2, x1_name, x2_name = val_batch
        pred = self.classifier(x1, x2)
        writer.writerow([x1_name[0], x2_name[0], pred.item()])
        file.close()
        return pred



parser = argparse.ArgumentParser(description='Point Cloud Classification')
parser.add_argument('--exp_name', type=str, default='xcit_tiny+2sl+1dl', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=413, metavar='S',
                    help='random seed (default: 413)')
parser.add_argument('--device_ids', type=str, default='[1]',
                    help='induct fix id to train')
parser.add_argument('--precision', type=int, default=32,
                    help='precision of gpu')
parser.add_argument('--test', type=bool, default=0,
                    help='decide whether to test')
parser.add_argument('--lr_find', type=bool, default=1,
                    help='decide whether to find lr')
parser.add_argument('--train_path', type=str,
                    default='/home/gdut403/sonwe1e/dogpet-master/pet_biometric_challenge_2022/train/', )
parser.add_argument('--test_path', type=str,
                    default='/home/gdut403/sonwe1e/dogpet-master/pet_biometric_challenge_2022/test/', )
parser.add_argument('--results_dir', type=str,
                    default='/home/gdut403/sonwe1e/dogpet-master/result/', )
parser.add_argument('--img_size', type=int, default=224,
                    help='size of image')
args = parser.parse_args()

img1_transformer = A.Compose([
    A.Resize(height=args.img_size, width=args.img_size),
    A.Normalize(max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
])
train_transformer = A.Compose([
    A.Resize(height=args.img_size, width=args.img_size),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),

    ], p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(rotate_limit=1, p=0.5),
    A.Normalize(max_pixel_value=255.0,p=1.0),
    ToTensorV2(p=1.0),
])

test_transformer = A.Compose([
    A.Resize(height=args.img_size, width=args.img_size),
    A.Normalize(max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
])

def main():
    pl.seed_everything(args.seed)
    tb_logger = loggers.TensorBoardLogger(save_dir='./lightning_logs', name=args.exp_name)
    train_set = Simdataset(args.train_path, mode='train', img1_transformer=img1_transformer, img2_transformer=train_transformer)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)

    model = pointclassifier(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_sigsim_loss/bce_loss', mode='min', save_top_k=5)
    stochastic_stop_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    if args.lr_find:
        trainer = pl.Trainer(gpus=[1])
        lr_finder = trainer.tuner.lr_find(model, train_loader)
        print(lr_finder.suggestion())
        model.hparams.lr = lr_finder.suggestion()
    trainer = pl.Trainer(devices=eval(args.device_ids), accelerator='auto', max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, stochastic_stop_callback], fast_dev_run=args.test,
                         logger=tb_logger, precision=args.precision)
    trainer.fit(model, train_loader,
                ckpt_path='/home/gdut403/sonwe1e/dogpet-master/lightning_logs/xcit_tiny+2sl+1dl/version_4/checkpoints/epoch=246-step=154375.ckpt')
    dataset = testdatset(args.test_path, transformer=test_transformer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    trainer.predict(model, dataloaders=dataloader)

if __name__ == '__main__':
    main()
