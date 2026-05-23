import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scheduler import NoiseScheduler




def test_add_noise_shape():
    """NoiseScheduler の add_noise 関数が、入力と同じ形状の出力を返すことを確認するテスト関数。"""
    #arrange(準備) - NoiseScheduler とダミーデータを作成
    scheduler = NoiseScheduler()
    batch_size = 4
    frames = 60
    joints = 22
    x_0 = torch.randn(batch_size, frames, joints*3)  # [B, F, J*3]
    noise = torch.randn_like(x_0)  # x_0 と同じ形状のノイズ
    t = torch.randint(0, scheduler.num_timesteps, (batch_size,))  # ランダムな時間ステップ
    #act(実行) - add_noise 関数を呼び出す
    x_t = scheduler.add_noise(x_0, noise, t)  # ノイズを加える
    #assert(検証) - 出力の形状が入力と同じであることを確認
    assert x_t.shape == x_0.shape, f"Expected shape {x_0.shape}, but got {x_t.shape}"
    print("test_add_noise_shape passed.")

def test_add_noise_statistics():
    """NoiseScheduler の add_noise 関数が、ノイズを加えた後の統計量が期待通りであることを確認するテスト関数。"""
    #arrange(準備) - NoiseScheduler とダミーデータを作成
    scheduler = NoiseScheduler()
    batch_size = 1000
    frames = 60
    joints = 22
    x_0 = torch.zeros(batch_size, frames, joints*3)  # ゼロの入力
    noise = torch.randn_like(x_0)  # 標準正規分布のノイズ
    # t=T-1（最終ステップ）なら ā_t≈0 なので x_t≈ε となり std≈1 になるはず
    t = torch.full((batch_size,), scheduler.num_timesteps - 1, dtype=torch.long)
    #act(実行) - add_noise 関数を呼び出す
    x_t = scheduler.add_noise(x_0, noise, t)  # ノイズを加える
    mean = x_t.mean().item()
    std = x_t.std().item()
    #assert(検証) - 平均が0に近く、標準偏差が1に近いことを確認
    assert abs(mean) < 0.1, f"Expected mean close to 0, but got {mean}"
    assert abs(std - 1.0) < 0.1, f"Expected std close to 1, but got {std}"
    print("test_add_noise_statistics passed.")

def test_step_no_noise_at_t0():
    #arrange(準備) - NoiseScheduler とダミーデータを作成
    scheduler = NoiseScheduler()
    B, F, J = 4, 60, 22
    pred_x0 = torch.randn(B, F, J*3)
    x_t = torch.randn(B, F, J*3)
    t = torch.zeros(B, dtype=torch.long)  # t=0
    #act(実行) - step 関数を呼び出す
    # 2回呼ぶ → ノイズなしなら毎回同じ結果になる
    out1 = scheduler.step(pred_x0, x_t, t)
    out2 = scheduler.step(pred_x0, x_t, t)
    #assert(検証) - t=0 のとき step() は決定論的であることを確認
    assert torch.allclose(out1, out2), "t=0 のとき step() は決定論的であるべき"
    print("test_step_no_noise_at_t0 passed.")