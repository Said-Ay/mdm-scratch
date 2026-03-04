import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import MDM
from scheduler import NoiseScheduler

def train_step():
    print("--- トレーニング開始 ---")
    num_joints = 22
    num_actions = 10 #のちに変更
    # --- 1. 準備 ---
    # MDM、NoiseScheduler、Adamオプティマイザ（lr≈1e-4）を初期化する。
    mdm_model = MDM(num_actions, num_joints)
    scheduler = NoiseScheduler()
    optimizer = Adam(mdm_model.parameters(), lr=1e-4)
    # --- 2. ダミーデータ ---
    # きれいな動き x_0、時刻 t、アクションラベルを用意する。
    # 形状の目安: batch=4, frames=60, joints=66, action∈[0,9]。
    batch = 4
    frames = 60
 
    x_0 = torch.randn(batch, frames, num_joints*3)  
    t = torch.randint(0, scheduler.num_timesteps, (batch,))
    action_class = torch.randint(0, 10, (batch,))
    # --- 3. 1ステップ学習 ---
    # 1) ノイズ noise をサンプルする。
    noise = torch.randn_like(x_0)
    # 2) スケジューラで x_0 に noise を混ぜて時刻 t の x_t を作る。
    x_t = scheduler.add_noise(x_0, noise, t)
    # 3) オプティマイザの勾配をゼロクリア。
    optimizer.zero_grad()
    # 4) モデルに (x_t, t, action_class) を入れて x_0 を予測する → predicted_x_0。
    predicted_x_0 = mdm_model(x_t, t, action_class)
    # 5) predicted_x_0 と x_0 の MSE Loss を計算。
    loss = F.mse_loss(predicted_x_0, x_0)
    # 6) Loss を逆伝播。
    loss.backward()
    # 7) パラメータを更新。
    optimizer.step()


    # 最後に Loss を表示する。
    print(f"完了。Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_step()