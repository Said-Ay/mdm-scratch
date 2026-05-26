#1. argparse をインポートして、コマンドライン引数を処理できるようにする。
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # GUI 不要 (CI / リモート対応)。pyplot より先に呼ぶ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 用 import (副作用で有効化)
from matplotlib.animation import FuncAnimation, PillowWriter

#2. コマンドライン引数を定義する。必要な引数は以下の通り。
#   --input: 可視化したい NumPy ファイルのパス（必須）
#   --output: 出力するGIFファイルのパス（デフォルトは   output/motion.gif）
#   --sample_idx: 入力ファイル内のどのサンプルを表示するかのインデックス（デフォルトは0）
#   --fps: GIFのフレームレート（デフォルトは20）
#  --title: GIFのタイトル（デフォルトは空文字）
parser = argparse.ArgumentParser(description="MDM 可視化スクリプト")
parser.add_argument("--input", type=str, required=True, help="入力ファイルのパス")
parser.add_argument("--output", type=str, default="output/motion.gif", help="出力GIFのパス")
parser.add_argument("--sample_idx", type=int, default=0, help="表示するサンプルのインデックス")
parser.add_argument("--fps", type=int, default=20, help="GIFのフレームレート")
parser.add_argument("--title", type=str, default="", help="GIFのタイトル")


# 3. 引数を解析して、変数に格納する。    
args = parser.parse_args()
data = np.load(args.input)  # shape: [B, F, 66]
motion = data[args.sample_idx]  # [F, 66]
motion = motion.reshape(motion.shape[0], 22, 3)  # [F, 22, 3]

#4. 前処理 - 可視化のために、以下の前処理を行う。
# HumanAct12 用スケール: 軸反転 + 拡大
motion = motion * -1.5

# Pelvis (joint 0) の x, z だけを毎フレーム引く → y は保つので床に立つ
motion[:, :, [0, 2]] -= motion[:, 0:1, [0, 2]]

# 床面の高さ調整: 全フレーム通して最低 y を 0 に
motion[..., 1] -= motion[..., 1].min()
motion[..., 1] *= 2.0  # 縦方向を引き延ばす

#5. 軸範囲とカラー定数を定義する。軸範囲は全フレームの最大値と最小値で固定し、カラーは関節ごとに異なる色を指定する。
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],          # 右脚
    [0, 1, 4, 7, 10],          # 左脚
    [0, 3, 6, 9, 12, 15],      # 背骨 + 頭
    [9, 14, 17, 19, 21],       # 右腕
    [9, 13, 16, 18, 20],       # 左腕
]
COLORS = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]

# 軸範囲は全フレーム max/min で固定（カメラぶれ防止）
mins = motion.min(axis=(0, 1))  # [3]
maxs = motion.max(axis=(0, 1))

#6. figureとAxesを作成する。Matplotlib を使って、3Dプロットの figure と axes を作成する。
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="3d")

#7. アニメーション関数を定義する。FuncAnimation を使って、各フレームで関節を線でつなげて描画するアニメーション関数を定義する。
def update(i):
    ax.clear()  # 前フレームの線を消す（重要）
    
    # 軸範囲を毎回固定
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    
    # 視点固定
    ax.view_init(elev=120, azim=-90)
    
    # 軸ラベル/目盛 off で見た目クリーン
    ax.set_axis_off()
    
    # 各 chain を描画
    for chain_idx, (chain, color) in enumerate(zip(KINEMATIC_CHAIN, COLORS)):
        linewidth = 4.0 if chain_idx < 2 else 2.0  # 脚は太く
        xs = motion[i, chain, 0]
        ys = motion[i, chain, 1]
        zs = motion[i, chain, 2]
        ax.plot3D(xs, ys, zs, linewidth=linewidth, color=color)
    
    # タイトル
    if args.title:
        ax.set_title(f"{args.title} [frame {i}]")

#8. アニメーションを保存する。FuncAnimation を使って、定義したアニメーション関数を呼び出し、GIFファイルとして保存する。
n_frames = motion.shape[0]
ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps)

out_dir = os.path.dirname(args.output)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
ani.save(args.output, writer=PillowWriter(fps=args.fps), dpi=80)
plt.close(fig)
print(f"保存: {args.output}")