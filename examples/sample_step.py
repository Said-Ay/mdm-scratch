"""このファイルは、デモンストレーションのための単一のサンプリングステップを動作確認として実装しています。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import MDM
from scheduler import NoiseScheduler

def sample_step():
    """サンプリングのデモンストレーションを行う関数。"""
    print("--- サンプリング開始 ---")
    mdm_model = MDM(num_actions=10, num_joints=22)
    mdm_model.eval()  # 推論モードに切り替え
    scheduler = NoiseScheduler()

    B, F, J = 4, 60, 22
    #x_T~N(0,I) なので、x_t をランダムノイズで初期化する。
    x_t = torch.randn(B, F, J*3)  # [B, F, J*3]
    action_class = torch.randint(0, 10, (B,))  # [B] - ランダムなアクションID、サイズBの1次元テンソルを３項めで指定



    with torch.no_grad():  # 勾配計算をオフにする
        for i in reversed(range(scheduler.num_timesteps)):# 999→0へループ
            t = torch.full((B,), i, dtype=torch.long)  # 現在のステップtを全バッチに対して同じ値で作成
            pred_x0 = mdm_model(x_t, t,action_class)  # モデルに現在のx_t、t、action_classを入力してx_0を予測
            x_t = scheduler.step(pred_x0, x_t, t)  # スケジューラーのstep関数を呼び出して次のx_tを計算
    x_0_generated = x_t  # 最終的にx_0が生成される
    print("サンプリング完了。生成されたx_0の形状:", x_0_generated.shape) #[4, 60, 66] であることを期待

if __name__ == "__main__":
    sample_step()