# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- インライン補足 ---
# nn.Module: PyTorchでAIのネットワークを作るための「基本の型（クラス）」です。Chainerの chainer.Chain に対応します。
# nn.Conv2d: 2次元の畳み込み層。Chainerの L.Convolution2D に対応します。
# nn.BatchNorm2d: バッチ正規化層。データの偏りを整えて学習を速くします。


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1):
        super(ConvBlock, self).__init__()
        # パディング（画像の端のパディング幅）を自動計算します（ksize=3なら1になります）
        padding = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 畳み込み -> バッチ正規化 -> 活性化関数(ReLU) の順番で計算します
        return F.relu(self.bn(self.conv(x)))


class Unet2D(nn.Module):
    def __init__(self, num_of_labels, ch=32, ksize=3):
        super(Unet2D, self).__init__()

        # 1. Encoder pass (画像を縮小しながら特徴を掴む部分)
        # ※ 後ほど、ここを丸ごと「UniConvNet」にすり替えるので楽しみにしていてください！
        self.ce0 = ConvBlock(in_channels=1, out_channels=ch, ksize=ksize)
        self.ce1 = ConvBlock(in_channels=ch, out_channels=ch, ksize=ksize)
        self.ce2 = ConvBlock(in_channels=ch, out_channels=ch * 2, ksize=ksize)
        self.ce3 = ConvBlock(in_channels=ch * 2, out_channels=ch * 2, ksize=ksize)
        self.ce4 = ConvBlock(in_channels=ch * 2, out_channels=ch * 4, ksize=ksize)
        self.ce5 = ConvBlock(in_channels=ch * 4, out_channels=ch * 4, ksize=ksize)
        self.ce6 = ConvBlock(in_channels=ch * 4, out_channels=ch * 8, ksize=ksize)
        self.ce7 = ConvBlock(in_channels=ch * 8, out_channels=ch * 8, ksize=ksize)
        self.ce8 = ConvBlock(in_channels=ch * 8, out_channels=ch * 16, ksize=ksize)

        # 2. Decoder pass (特徴を元の画像サイズに拡大しながら綺麗に塗り分ける部分)
        # nn.ConvTranspose2d: 画像の縦横サイズを2倍に拡大する「逆畳み込み層」です。Chainerの L.Deconvolution2D に対応します。
        self.cd8 = ConvBlock(in_channels=ch * 16, out_channels=ch * 8, ksize=ksize)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=ch * 8, out_channels=ch * 8, kernel_size=2, stride=2
        )

        self.cd7 = ConvBlock(
            in_channels=ch * 8 + ch * 8, out_channels=ch * 8, ksize=ksize
        )
        self.cd6 = ConvBlock(in_channels=ch * 8, out_channels=ch * 4, ksize=ksize)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=ch * 4, out_channels=ch * 4, kernel_size=2, stride=2
        )

        self.cd5 = ConvBlock(
            in_channels=ch * 4 + ch * 4, out_channels=ch * 4, ksize=ksize
        )
        self.cd4 = ConvBlock(in_channels=ch * 4, out_channels=ch * 2, ksize=ksize)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=ch * 2, out_channels=ch * 2, kernel_size=2, stride=2
        )

        self.cd3 = ConvBlock(
            in_channels=ch * 2 + ch * 2, out_channels=ch * 2, ksize=ksize
        )
        self.cd2 = ConvBlock(in_channels=ch * 2, out_channels=ch, ksize=ksize)
        self.deconv0 = nn.ConvTranspose2d(
            in_channels=ch, out_channels=ch, kernel_size=2, stride=2
        )

        self.cd1 = ConvBlock(in_channels=ch + ch, out_channels=ch, ksize=ksize)
        self.cd0 = ConvBlock(in_channels=ch, out_channels=ch, ksize=ksize)

        # 最終出力層 (1x1の畳み込みで、目標の13クラス(num_of_labels)のチャンネル数に変換します)
        self.lcl = nn.Conv2d(
            in_channels=ch, out_channels=num_of_labels, kernel_size=1, padding=0
        )

        # nn.MaxPool2d: 画像の縦横を半分にするプーリング層です。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # --- Encoder pass ---
        e0 = self.ce0(x)
        e1 = self.ce1(e0)
        e2 = self.ce2(self.pool(e1))
        e3 = self.ce3(e2)
        e4 = self.ce4(self.pool(e3))
        e5 = self.ce5(e4)
        e6 = self.ce6(self.pool(e5))
        e7 = self.ce7(e6)
        e8 = self.ce8(self.pool(e7))

        # --- Decoder pass ---
        d8 = self.cd8(e8)

        # torch.cat(..., dim=1): 前半でとっておいた特徴(e7など)と、拡大した特徴をチャンネル方向(横方向)に合体させます。
        # これがU-Netの命である「スキップ接続」です！Chainerの F.concat に対応します。
        d7 = self.cd7(torch.cat([self.deconv3(d8), e7], dim=1))
        d6 = self.cd6(d7)

        d5 = self.cd5(torch.cat([self.deconv2(d6), e5], dim=1))
        d4 = self.cd4(d5)

        d3 = self.cd3(torch.cat([self.deconv1(d4), e3], dim=1))
        d2 = self.cd2(d3)

        d1 = self.cd1(torch.cat([self.deconv0(d2), e1], dim=1))
        d0 = self.cd0(d1)

        # ロジット（Softmaxをかける前の生のスコア）を出力します。
        # ※ PyTorchでは、計算の安定性と正確性を高めるため、モデルの最後でSoftmaxをかけず、
        #   損失関数（CrossEntropyLoss）側でまとめてSoftmaxを計算するのが標準的です！
        out = self.lcl(d0)
        return out
