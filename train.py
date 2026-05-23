import torch
from torch.utils.data import TensorDataset, DataLoader
from model import MDM
from scheduler import NoiseScheduler
from torch.optim import Adam
import pickle
import os
import torch.nn.functional as F

class HumanAct12Dataset(torch.utils.data.Dataset):
    def __init__(self,pkl_path,num_flames=60,num_joints=22):
        data = pickle.load(open(pkl_path,'rb'))
        joints = data["joints3D"] # [N, 24, 3]のnumpy配列
        labels = data["y"] # intのnumpy配列
        self.samples = []
        self.labels = []
        for joints,label in zip(joints,labels) :
            if joints.shape[0] >= num_flames : # 60フレーム以上あるサンプルだけ使う
                x = joints[:num_flames,:num_joints,:] # 最初の60フレームを切り取る
                x = x.reshape(num_flames, num_joints*3) # [60, 22,3]に変形
                self.samples.append(torch.tensor(x, dtype=torch.float32)) # テンソルに変換
                self.labels.append(label) # ラベルも保存
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx], self.labels[idx],torch.tensor(self.labels[idx],dtype=torch.long)
def train(num_epochs=5,batch_size=16,lr=1e-4,save_path="checkpoints"):
    #lrは学習率lerning rate、save_pathはモデルの保存先ディレクトリを指定する引数です。
    """HumanAct12Dataset を使って MDM をトレーニングする関数。"""
    print("--- トレーニング開始 ---")
    # --- 1. 準備 ---
    dataset = HumanAct12Dataset("reference/dataset/HumanAct12Poses/humanact12poses.pkl") 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MDM(num_actions=12, num_joints=22)
    scheduler = NoiseScheduler()
    optimizer = Adam(model.parameters(), lr=lr)
    # --- 2. エポックループ ---
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_0, action_class, action_class_tensor in dataloader:
            batch_size = x_0.size(0) # 現在のバッチのサイズを取得
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,)) # バッチサイズに合わせてランダムな時間ステップを生成
            noise = torch.randn_like(x_0) # x_0 と同じ形状のノイズを生成
            x_t = scheduler.add_noise(x_0, noise, t) # ノイズを加えて x_t を作成
            optimizer.zero_grad()
            predicted_x_0 = model(x_t, t, action_class)# モデルに (x_t, t, action_class) を入れて x_0 を予測
            loss = F.mse_loss(predicted_x_0, x_0)   # predicted_x_0 と x_0 の MSE Loss を計算
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size  # バッチのサイズを掛けて合計損失を更新
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # --- 3. モデル保存 ---
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_path}/mdm_final.pth")
    print(f"トレーニング完了。モデルを {save_path}/mdm_final.pth に保存しました。")


if __name__ == "__main__":
    train()