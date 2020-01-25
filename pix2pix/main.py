import argparse
import os
import time
import numpy as np
import datetime
import sys
import pandas as pd

from models import *
from datasets import *
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="sample", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "--val_rate", type=float, default=0.1, help="validation rate"
)
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--in_channels", type=int, default=3, help="number of input image channels")
parser.add_argument("--out_channels", type=int, default=3, help="number of output image channels")
parser.add_argument("--dropout_ratio_UNet", type=float, default=0.5, help="dropout ratio in Generator (UNet)")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=30, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# 'images', 'images/opt.dataset_name', 'saved_models', 'saved_models/opt.dataset_name'が存在しなければフォルダを作成する
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# cudaが使えるならTrue,使えないならFalse
cuda = True if torch.cuda.is_available() else False

# Loss functions === 要確認 & 検討 ===
criterion_GAN = nn.MSELoss() # Mean Squared Loss Error
criterion_pixelwise = nn.L1Loss() # ピクセル単位のL1ロス?

# L1ロスに対する重み (100が論文で実験された値; 大きいほど元画像に近くなる)
lambda_pixel = 100

# Discriminatorが出力する特徴マップのサイズ計算
# やりたいことは、画像の部分ごとの真贋判定、つまりPatchGAN
# 入力画像の一部分ずつ入力して判定するよりも、Discriminatorが特徴マップを出力する方が効率的（出力値が各patchに対する真贋判定結果）
# https://nodaki.hatenablog.com/entry/2018/07/27/235914
# https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
# 縦横それぞれ16分割（全体で256分割）で真贋判定
patch = (1, opt.img_height // (2 ** 4), opt.img_width // (2 ** 4))

# 単一の物理量分布同士の変換の場合はin_channels = out_channels = 3
# 速度場分布などを追加で変換に用いる場合は、in_channelsを増やす
# 白黒の分布図を1つ追加する場合は＋1, カラーの分布図を1つ追加する場合は＋3
generator = GeneratorUNet(in_channels=opt.in_channels, out_channels=opt.out_channels, dropout_ratio = opt.dropout_ratio_UNet)

# Discriminatorには変換前後の単一の物理量分布図を入力する
# Discriminator内でch*2をしているので、変換前のチャンネル数を入力
discriminator = Discriminator(in_channels=3)

#cudaが使用可能なら変換する
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

#開始epoch数を指定すると前回保存した重みファイルをロードして訓練を再開する
if opt.epoch != 0:
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    #初めての場合、重みを初期化する
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# 最適化の重みを指定
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#画像の前処理方法の設定
transforms_ = [
# (PIL.)Image.BICUBIC: cubic spline interpolationで各ピクセルの値を計算
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
# Tensor型（channel x Height x Width）で輝度を0～1に変換
    transforms.ToTensor(),
# 3つのチャンネルをそれぞれ-1～1になるように規格化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms1D_ = [
# (PIL.)Image.BICUBIC: cubic spline interpolationで各ピクセルの値を計算
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
# Tensor型（channel x Height x Width）で輝度を0～1に変換
    transforms.ToTensor(),
# 3つのチャンネルをそれぞれ-1～1になるように規格化
    transforms.Normalize((0.5, ), (0.5, )),
]

#trainに使う画像と、valに使う画像を分ける。
train_A, train_B, train_C, val_A, val_B, val_C = make_datapath_list(opt)

train_dataloader = DataLoader(
    #MyDatasetはDatasetクラスをオリジナルに改変したもの->datasets.py
    MyDataset(train_A, train_B, train_C, opt, transforms_=transforms_, transforms1D_=transforms1D_),
    batch_size=opt.batch_size,
    shuffle=True,
#    num_workers=opt.n_cpu,
    num_workers=0,
)

val_dataloader = DataLoader(
    MyDataset(val_A, val_B, val_C, opt, transforms_=transforms_, transforms1D_=transforms1D_),
    batch_size=10,
    shuffle=True,
#    num_workers=1,
    num_workers=0,
)

# Tensorのタイプの指定
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done, out_channels=3):
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    fake_B = generator(real_A)
#    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)

    real_D = real_A[:, :out_channels, :, :]
    img_sample = torch.cat((real_D.data, fake_B.data, real_B.data), -2)

    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------
prev_time = time.time()

list_loss_D = []
list_loss_G = []
list_loss_pixel = []
list_loss_GAN = []
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dataloader):


        # 本物の画像の行列データ
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))


        # 正解ラベル
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        # 偽物ラベル
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        #optimizerが持つvariable型のパラメーターを初期化する
        optimizer_G.zero_grad()

        fake_B = generator(real_A)

        pred_fake = discriminator(real_A, fake_B, out_channels=opt.out_channels)

        #正解ラベルを与え、それに近づけるように学習
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        #勾配の計算
        loss_G.backward()
        #パラメーターの更新
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_A, real_B, out_channels=opt.out_channels)
        loss_real = criterion_GAN(pred_real, valid)

        # detach()とすることでrequires_grad=Falseとなるのでそれ以降の微分はされない。
        # detach()なしだと、fake_Bを通じて勾配がGに伝わってしまう。
        pred_fake = discriminator(real_A, fake_B.detach(), out_channels=opt.out_channels)

        #偽物と判断するようにする。
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch+1,
                opt.n_epochs,#総エポック数
                i,#iter数
                len(train_dataloader),#バッチ数
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, out_channels=opt.out_channels)

    # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:

    list_loss_D.append(loss_D.item())
    list_loss_G.append(loss_G.item())
    list_loss_pixel.append(loss_pixel.item())
    list_loss_GAN.append(loss_GAN.item())

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

torch.save(generator.state_dict(), "saved_models/%s/generator_last.pth" % (opt.dataset_name))
torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_last.pth" % (opt.dataset_name))

df = pd.DataFrame(data=None, columns=['loss_D', 'loss_G', 'loss_pixel', 'loss_GAN'])
df['loss_D'] = list_loss_D
df['loss_G'] = list_loss_G
df['loss_pixel'] = list_loss_pixel
df['loss_GAN'] = list_loss_GAN
df.to_csv('loss_history.csv')
