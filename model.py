import torch
import torch.nn as nn
import math


# パラパラ漫画のページ番号を教えるパーツ
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
        # 1. アクション（ただの数字）を、計算できるベクトル（Embedding）に変換する層
        self.action_embedding = nn.Embedding(num_actions , latent_dim)
        # 2. 時間 t（ノイズのステップ）をベクトルに変換する層
        self.time_embedding = nn.Sequential(nn.Linear(1 , latent_dim),nn.SiLU(),nn.Linear(latent_dim , latent_dim))
        # 3. 動きのデータ（関節の座標とか）をTransformerの次元に合わせる入り口
        self.pose = nn.Linear(num_joints*3 , latent_dim)
        # 4. メインのTransformerエンコーダー
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        self.seq_pos_enc = PositionalEncoding(latent_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. 最後にTransformerの出力を、元の動きのデータの次元に戻す出口
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)

    def forward(self, x, t, action_class):
        # 1. 時間とアクションをベクトルにして、2次元（1フレーム分の形）に引き伸ばす
        # t は [バッチサイズ] で入ってくるから、[バッチサイズ, 1] にしてから time_embedding に入れるよ
        t_emb = self.time_embedding(t.unsqueeze(1).float()).unsqueeze(1)
        
        # action_class は Embedding を通すと [バッチサイズ, 512] になるから、
        # そのあとに unsqueeze(1) をつけて [バッチサイズ, 1, 512] に引き伸ばしてね。
        c_emb = self.action_embedding(action_class).unsqueeze(1)
        
        # 2. 動きデータ x を、入り口（self.pose）に通して太さを揃える
        x_emb = self.pose(x)
        
        # 3. 3つをガッチャンコする！ [条件, 時間, 動き] の順番で列にするよ
        # ヒント: torch.cat([これ, と, これ], dim=1) って書くんだ
        seq = torch.cat([c_emb, t_emb, x_emb], dim=1)  # ← ここに書いてみて！
        
        # 4. さっき追加した「位置エンコーディング」で順番を教える
        seq = self.seq_pos_enc(seq)
        
        # 5. メインの脳みそ（Transformer）に流し込む
        out = self.transformer(seq)
        
        # 6. 先頭にくっつけた「条件(c)」と「時間(t)」の2フレーム分を切り捨てて、出口(self.output_proj)に通す
        # ヒント: Pythonのスライスを使って out[:, 2:, :] って書くと、先頭2つを無視できるよ
        out = out[:,2:,:] # ← ここに書いてみて！
        out = self.output_proj(out) # ← 最後に self.output_proj に通す！
        
        return out
    
