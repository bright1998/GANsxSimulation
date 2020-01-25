import os
import glob
import random
import math

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def make_datapath_list(opt):
    #data/'opt.dataset_name'以下の2つのフォルダ名を取得する(.で始まるフォルダは除く)
    dirs = [dir_name for dir_name in os.listdir('data/{}'.format(opt.dataset_name)) if not dir_name.startswith('.')]
    #フォルダをA,B順に並べ替える
    dirs = sorted(dirs)
    #カレントディレクトリから見た時の画像ファイル名のうち、
    #相対パス部分（ファイル名以外）の文字数をカウント
    str_len = len('data/{}/{}/'.format(opt.dataset_name, dirs[0]))
    #訓練用の画像リスト(***.jpg形式)
    filenames =  glob.glob('data/{}/{}/*.jpg'.format(opt.dataset_name, dirs[0]))
    img_list = [filename[str_len:] for filename in filenames]
    #imgのリストをシャフルする
    img_list = random.sample(img_list, len(img_list))
    #バリデーションの割合val_rateに従い(deafult=0.1)、学習用の画像リストとバリテーション用の画像リストに分割
    val_num = math.floor(len(img_list)*(1 - opt.val_rate))
    train_list = img_list[:val_num]
    val_list = img_list[val_num:]

    # 画像A,画像Bのpathのテンプレートを作成
    img_path_A = os.path.join("data/{}".format(opt.dataset_name), dirs[0], '%s')
    img_path_B = os.path.join("data/{}".format(opt.dataset_name), dirs[1], '%s')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_A = list()
    train_B = list()
    val_A = list()
    val_B = list()

    for train in train_list:
        path_A = (img_path_A % train)  # 画像のパス
        path_B = (img_path_B % train)  # アノテーションのパス
        train_A.append(path_A)
        train_B.append(path_B)

    for val in val_list:
        path_A = (img_path_A % val)  # 画像のパス
        path_B = (img_path_B % val)  # アノテーションのパス
        val_A.append(path_A)
        val_B.append(path_B)

    train_C = list()
    val_C = list()

    if opt.in_channels != opt.out_channels:
# auxiliary_data/'opt.dataset_name'/には'opt.dataset_name'_Cしかない前提
        dir_aux = [dir_name for dir_name in os.listdir('auxiliary_data/{}'.format(opt.dataset_name)) if not dir_name.startswith('.')]
        img_path_C = os.path.join("auxiliary_data/{}".format(opt.dataset_name), dir_aux[0], '%s')

        for train in train_list:
            path_C = (img_path_C % train)
            train_C.append(path_C)
        for val in val_list:
            path_C = (img_path_C % val)
            val_C.append(path_C)

    return train_A, train_B, train_C, val_A, val_B, val_C

class MyDataset(Dataset):
    def __init__(self, train_A, train_B, train_C, opt, transforms_ = None, transforms1D_ = None):
        self.transform = transforms.Compose(transforms_)
        self.transform1D = transforms.Compose(transforms1D_)
        self.data_num = len(train_A)
        self.train_A = train_A
        self.train_B = train_B
        self.train_C = train_C
        self.opt = opt

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        train_A = self.train_A[idx]
        img_A = Image.open(train_A)

        train_B = self.train_B[idx]
        img_B = Image.open(train_B)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        if self.opt.in_channels != self.opt.out_channels:
            train_C = self.train_C[idx]
            img_C = Image.open(train_C)
            # カラー画像⇒白黒画像
            if img_C.mode == 'RGB':
                img_C = img_C.convert('L')
            img_C = self.transform1D(img_C)
            img_A = torch.cat([img_A, img_C], dim=0)

        return {'A' : img_A, 'B' : img_B}