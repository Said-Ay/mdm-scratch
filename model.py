import torch
import torch.nn as nn
import math
from typing import cast


# 位置エンコーディング（フレーム位置の情報を埋め込む）
class PositionalEncoding(nn.Module):
    """Transformer の位置エンコーディングを実装。フレーム位置の情報を埋め込むために使用。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 埋め込み次元（Transformer の特徴量次元）
        dropout: ドロップアウト率(過学習を防ぐためランダムにニューロンを無効化する割合)
        max_len: 最大フレーム数（位置エンコーディングのテーブルサイズ）
        """
        super().__init__() # nn.Module の初期化
        self.dropout = nn.Dropout(p=dropout) 
    # 位置エンコーディングを事前計算してバッファに登録
        position = torch.arange(max_len).unsqueeze(1)
    # 位置エンコーディングの周波数を計算
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    # 位置エンコーディングを計算    
        pe = torch.zeros(max_len, 1, d_model)
    # 偶数次元に sin、奇数次元に cos を割り当てる
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
    # 位置エンコーディングをバッファに登録（モデルのパラメータではないが、GPUに移動させるため）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x の形状は [B, F, D] で、位置エンコーディング pe は [max_len, 1, D]。
        # フレーム数 F に合わせて pe を切り取って加算する   
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class MDM(nn.Module):
    """
    MDM（Motion Diffusion Model）を実装。Transformer をベースに、アクション条件と時間条件を組み込んだ構造。 
    """
    def __init__(self , num_actions, num_joints , latent_dim = 512 , num_layers = 8):
        super().__init__()
        # 1. アクションIDをベクトル化する層
        # action_embedding は、アクションIDを埋め込みベクトルに変換するための層で、num_actions はアクションの種類数、latent_dim は埋め込み次元を表す。これにより、モデルは異なるアクションに対して異なる特徴量を学習できるようになる。
        self.action_embedding = nn.Embedding(num_actions , latent_dim)
        # 2. 時間 t（ノイズステップ）をベクトル化する層
        # sinusoidal embedding: t を周波数成分に分解して latent_dim 次元のベクトルに変換する。
        # raw integer (0-999) を直接 Linear に渡すと数値スケールが不安定なため、sin/cos で正規化してから MLP に通す。
        self.latent_dim = latent_dim
        self.time_mlp = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        # 3. 動きデータを Transformer 次元に射影する入力層
        # pose 層は、動きデータを Transformer の特徴量次元に変換するための層で、これにより、モデルは動きデータを効果的に処理できるようになる。
        self.pose = nn.Linear(num_joints*3 , latent_dim)
        # 4. メインの Transformer エンコーダー
        # TransformerEncoderLayer は、Transformer の基本的な構成要素で、自己注意機構とフィードフォワードネットワークを含む。
        # nhead はマルチヘッド注意のヘッド数を表し、batch_first=True は入力テンソルの形状が (batch, seq, feature) であることを指定する。
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        #encoder_layerはTransformerEncoder に渡すための設定オブジェクト。保存不要なのでインスタンス変数にしない
        self.seq_pos_enc = PositionalEncoding(latent_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. 出力を元の動きデータ次元へ戻す射影層
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)

    def _sinusoidal_embedding(self, timesteps):
        half = self.latent_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, latent_dim]

    def forward(self, x, t, action_class):
        """
        入力:
        x: [B, F, J*3] - ノイズあり動きデータ（バッチサイズ B、フレーム数 F、関節数 Jのx,y,z座標）
        t: [B] - ノイズステップ（時間条件）
        action_class: [B] - アクションID（条件） 
        出力: 
        [B, F, J*3] - 予測された動きデータ（元の次元に戻す）
        """
        # 1. 時間とアクションをベクトル化し、フレーム軸を持たせる
        #unsqueeze(1) は、t と action_class の次元を増やして [B, 1] にするために使用される。
        t_emb = self.time_mlp(self._sinusoidal_embedding(t)).unsqueeze(1)  # [B, 1, D]
        c_emb = self.action_embedding(action_class).unsqueeze(1)
        
        # 2. 動きデータを埋め込み次元へ射影
        x_emb = self.pose(x)
        
        # 3. [条件, 時間, 動き] の順で結合
        #cat は、c_emb、t_emb、x_emb をフレーム軸（dim=1）で結合するために使用される。
        seq = torch.cat([c_emb, t_emb, x_emb], dim=1)
        
        # 4. 位置エンコーディングを加える
        seq = self.seq_pos_enc(seq)
        
        # 5. Transformer へ入力
        out = self.transformer(seq)
        
        # 6. 先頭の条件・時間トークンを除き、出力射影
        out = out[:,2:,:]
        out = self.output_proj(out)
        
        return out
    


