import torch
import torch.nn as nn
import math


# 位置エンコーディング（フレーム位置の情報を埋め込む）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
# ↑ここまで追加

class MDM(nn.Module):
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
    
