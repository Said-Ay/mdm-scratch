import torch
from torch.utils.data import DataLoader
from model import MDM
from scheduler import NoiseScheduler
from torch.optim import Adam
import pickle
import os
import torch.nn.functional as F

class HumanAct12Dataset(torch.utils.data.Dataset):
    def __init__(self,pkl_path,num_frames=60,num_joints=22):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        joints = data["joints3D"] # [N, 24, 3]のnumpy配列
        labels = data["y"] # intのnumpy配列
        self.samples = []
        self.labels = []
        for joints,label in zip(joints,labels) :
            if joints.shape[0] >= num_frames : # 60フレーム以上あるサンプルだけ使う
                x = joints[:num_frames,:num_joints,:] # 最初の60フレームを切り取る
                x = x.reshape(num_frames, num_joints*3) # [60, 22,3]に変形
                self.samples.append(torch.tensor(x, dtype=torch.float32)) # テンソルに変換
                self.labels.append(label) # ラベルも保存
        # 正規化: 全サンプル・全フレームで共通の mean/std を計算
        all_data = torch.stack(self.samples)  # [N, F, D]
        self.mean = all_data.mean()
        self.std = all_data.std().clamp(min=1e-6)
        self.samples = [(s - self.mean) / self.std for s in self.samples]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx], torch.tensor(self.labels[idx], dtype=torch.long)
def train(num_epochs=200,batch_size=16,lr=5e-4,save_path="checkpoints"):
    #lrは学習率lerning rate、save_pathはモデルの保存先ディレクトリを指定する引数です。
    """HumanAct12Dataset を使って MDM をトレーニングする関数。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU があれば使う、なければ CPU
    print(f"--- トレーニング開始 (device: {device}) ---")
    # --- 1. 準備 ---
    dataset = HumanAct12Dataset("reference/dataset/HumanAct12Poses/humanact12poses.pkl") 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MDM(num_actions=12, num_joints=22).to(device)  # モデルをデバイスへ
    scheduler = NoiseScheduler()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # --- 2. エポックループ ---
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_0, action_class in dataloader:
            x_0 = x_0.to(device)                  # テンソルをデバイスへ
            action_class = action_class.to(device)
            batch_size = x_0.size(0) # 現在のバッチのサイズを取得
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device) # バッチサイズに合わせてランダムな時間ステップを生成
            noise = torch.randn_like(x_0) # x_0 と同じ形状のノイズを生成
            x_t = scheduler.add_noise(x_0, noise, t) # ノイズを加えて x_t を作成
            optimizer.zero_grad()
            predicted_x_0 = model(x_t, t, action_class)
            mse_loss = F.mse_loss(predicted_x_0, x_0)
            vel_pred = predicted_x_0[:, 1:] - predicted_x_0[:, :-1]
            vel_gt   = x_0[:, 1:] - x_0[:, :-1]
            velocity_loss = F.mse_loss(vel_pred, vel_gt)
            loss = mse_loss + velocity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size  # バッチのサイズを掛けて合計損失を更新
        avg_loss = total_loss / len(dataset)
        scheduler_lr.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # --- 3. モデル保存 ---
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_path}/mdm_final.pth")
    torch.save({'mean': dataset.mean, 'std': dataset.std}, f"{save_path}/norm_stats.pt")
    print(f"トレーニング完了。モデルを {save_path}/mdm_final.pth に保存しました。")


if __name__ == "__main__":
    train()