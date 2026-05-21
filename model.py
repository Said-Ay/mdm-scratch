import torch
import torch.nn as nn
import math
from typing import cast


# 位置エンコーディング（フレーム位置の情報を埋め込む）
class PositionalEncoding(nn.Module):
    """Transformer の位置エンコーディングを実装。フレーム位置の情報を埋め込むために使用。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 埋め込み次元数 
        dropout: ドロップアウト率
        max_len: 位置エンコーディングの最大長（フレーム数の上限）
        """
        super().__init__()
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
    入力:
        x: [B, F, J*3] - 動きデータ（フレーム数 F、関節数 J）
        t: [B] - ノイズステップ（時間条件）
        action_class: [B] - アクションID（条件） 
        出力: [B, F, J*3] - 生成された動きデータ
    """
    def __init__(self , num_actions, num_joints , latent_dim = 512 , num_layers = 8):
        super().__init__()
        # 1. アクションIDをベクトル化する層
        self.action_embedding = nn.Embedding(num_actions , latent_dim)
        # 2. 時間 t（ノイズステップ）をベクトル化する層
        self.time_embedding = nn.Sequential(nn.Linear(1 , latent_dim),nn.SiLU(),nn.Linear(latent_dim , latent_dim))
        # 3. 動きデータを Transformer 次元に射影する入力層
        self.pose = nn.Linear(num_joints*3 , latent_dim)
        # 4. メインの Transformer エンコーダー
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        self.seq_pos_enc = PositionalEncoding(latent_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. 出力を元の動きデータ次元へ戻す射影層
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)

    def forward(self, x, t, action_class):
        """
        x: [B, F, J*3] - 動きデータ
        t: [B] - ノイズステップ（時間条件）
        action_class: [B] - アクションID（条件）"""
        # 1. 時間とアクションをベクトル化し、フレーム軸を持たせる
        t_emb = self.time_embedding(t.unsqueeze(1).float()).unsqueeze(1)
        c_emb = self.action_embedding(action_class).unsqueeze(1)
        
        # 2. 動きデータを埋め込み次元へ射影
        x_emb = self.pose(x)
        
        # 3. [条件, 時間, 動き] の順で結合
        seq = torch.cat([c_emb, t_emb, x_emb], dim=1)
        
        # 4. 位置エンコーディングを付与
        seq = self.seq_pos_enc(seq)
        
        # 5. Transformer へ入力
        out = self.transformer(seq)
        
        # 6. 先頭の条件・時間トークンを除き、出力射影
        out = out[:,2:,:]
        out = self.output_proj(out)
        
        return out
    


