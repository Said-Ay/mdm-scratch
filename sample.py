import os
import torch
from model import MDM
from scheduler import NoiseScheduler
import numpy as np
import argparse

def sample(checkpoint,action_id,num_samples,output_path):
    """トレーニング済みの MDM を使ってサンプリングを行う関数。"""
    print("--- サンプリング開始 ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    mdm_model = MDM(num_actions=12, num_joints=22).to(device)
    mdm_model.load_state_dict(torch.load(checkpoint, map_location=device))
    mdm_model.eval()  # 推論モードに切り替え
    scheduler = NoiseScheduler()

    B, F, J = num_samples, 60, 22
    x_t = torch.randn(B, F, J*3).to(device)  # [B, F, J*3]
    action_class = torch.full((B,), action_id, dtype=torch.long).to(device)  # [B] - 指定されたアクションID

    with torch.no_grad():  # 勾配計算をオフにする
        for i in reversed(range(scheduler.num_timesteps)):
            t = torch.full((B,), i, dtype=torch.long).to(device)  # 現在のステップtを全バッチに対して同じ値で作成
            pred_x0 = mdm_model(x_t, t, action_class)  # モデルに現在のx_t、t、action_classを入力してx_0を予測
            x_t = scheduler.step(pred_x0, x_t, t)  # スケジューラーのstep関数を呼び出して次のx_tを計算
    stats = torch.load(os.path.join(os.path.dirname(checkpoint), 'norm_stats.pt'), map_location='cpu')
    x_0_generated = x_t.cpu() * stats['std'] + stats['mean']  # 逆正規化
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"generated_action{action_id}_samples{num_samples}.npy")
    np.save(output_file, x_0_generated.numpy())
    print(f"サンプリング完了。生成されたx_0を {output_file} に保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDM サンプリングスクリプト")
    parser.add_argument("--checkpoint", type=str, required=True, help="トレーニング済みモデルのチェックポイントファイルパス")
    parser.add_argument("--action_id", type=int, required=True, help="生成したいアクションのID (0-11)")
    parser.add_argument("--num_samples", type=int, default=4, help="生成するサンプルの数")
    parser.add_argument("--output_path", type=str, default="output", help="生成されたサンプルの保存先ディレクトリ")
    args = parser.parse_args()

    sample(args.checkpoint, args.action_id, args.num_samples, args.output_path)