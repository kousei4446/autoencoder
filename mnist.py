import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optimizers
import matplotlib.pyplot as plt


#データ前処理
# トランスフォームオブジェクトを生成
transform = transforms.Compose(
    [transforms.ToTensor(),     #PyTorch Tensor型に変換＋ピクセル値を0〜1に正規化
     nn.Flatten(),              # 28行 × 28列　→　1行 × 784列に変換
     ])
# データセットとデータローダの生成
mnist_train = torchvision.datasets.MNIST(
    root='MNIST',
    download = True,
    train = True,
    transform=transform
)

mnist_test = torchvision.datasets.MNIST(
    root='MNIST',
    download=True,
    train=False,
    transform=transform
)
print(mnist_train)
# MNISTデータの「1件分」取得,出力(tensor画像データ,正解ラベルの形)
print(mnist_train[0])






# データローダ
train_dataloader = DataLoader(mnist_train,
                              batch_size=124,
                              shuffle=True)
test_dataloader = DataLoader(mnist_test,
                             batch_size=1,
                             shuffle=False)
_ = mnist_train[0]
print(f'type:{type(_)}, data:{_[0].shape}, label:{_[1]}')
print(f'最大値:{_[0].max()}, 最小値:{_[0].min()}')






# Autoencoderクラス
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 200)   # エンコーダー
        self.l2 = nn.Linear(200, 784)   # デコーダー

    def forward(self, x):
        h = self.l1(x)       # エンコーダーに入力
        h = torch.relu(h)    # ReLU関数を適用

        h = self.l2(h)       # デコーダーに入力
        y = torch.sigmoid(h) # シグモイド関数を適用
        return y






# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)






# 学習の実行
model = Autoencoder().to(device)                # オートエンコーダーを生成

# 損失関数はバイナリクロスエントロピー誤差
# BCE = -(正解ラベル * log(予測確率) + (1 - 正解ラベル) * log(1 - 予測確率))
criterion = nn.BCELoss()                        

optimizer = optimizers.Adam(model.parameters())     # 最適化手法はAdam
epochs = 1                                        # エポック数

for epoch in range(epochs):
    train_loss = 0.
    
    # ミニバッチのループ(ステップ)
    for (x, _) in train_dataloader:
        x = x.to(device)            # デバイスの割り当て
        model.train()               # 訓練モードにする
        preds = model(x)            # モデルの出力を取得
        loss = criterion(preds, x)  # 入力xと復元predsの誤差を取得
        optimizer.zero_grad()       # 勾配を0で初期化
        loss.backward()             # 誤差の勾配を計算
        optimizer.step()            # パラメーターの更新
        train_loss += loss.item()   # 誤差(損失)の更新
        
    # 1エポックあたりの損失を求める
    train_loss /= len(train_dataloader)
    # 1エポックごとに損失を出力
    print('Epoch({}) -- Loss: {:.3f}'.format(
        epoch+1,
        train_loss
    ))









# 未知データの予測
# テストデータを1個取り出す
_x, _ = next(iter(test_dataloader))
_x = _x.to(device)
print(_x)
print(_)

model.eval() # ネットワークを評価モードにする
x_rec = model(_x) # テストデータを入力して結果を取得

# 入力画像、復元画像を表示
titles = {0: 'Original', 1: 'Autoencoder:Epoch=1'}
for i, image in enumerate([_x, x_rec]):
    image = image.view(28, 28).detach().cpu().numpy()
    plt.subplot(1, 2, i+1)
    plt.imshow(image, cmap='binary_r')
    plt.axis('off'), plt.title(titles[i])
plt.show()
