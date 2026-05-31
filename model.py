import torch
import torch.nn as nn
import math
from typing import cast


class PositionalEncoding(nn.Module):
    """Transformer の位置エンコーディングを実装。フレーム位置の情報を埋め込むために使用。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) 
        position = torch.arange(max_len).unsqueeze(1) # position: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  
        #周波数は次元ごとに異なり、偶数次元には正弦波、奇数次元には余弦波が使用されるため、div_termはd_modelの半分のサイズになります。
        #div_termは、位置エンコーディングの周波数を決定するための値
        pe = torch.zeros(max_len, 1, d_model) #pe: [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, F, D]  pe: [max_len, 1, D]
        pe = cast(torch.Tensor, self.pe) #peはregister_bufferで登録されているため、self.peはTensor型であることが保証されているが、型ヒントのためにcastを使用して明示的にTensor型にキャストしています。
        x = x + pe[:x.size(1)].transpose(0, 1) # pe[:x.size(1)]: [F, 1, D] → transpose(0, 1) → [1, F, D] これをxに加算して位置情報を埋め込む
        return self.dropout(x)


class MDM(nn.Module):
    """
    MDM（Motion Diffusion Model）を実装。Reference (trans_enc) に準拠した構造。
    - 時間埋め込み: PE テーブルを timestep t でインデックスして MLP に通す（Reference TimestepEmbedder と同一）
    - 条件注入: time_emb + action_emb を 1 トークンに合算してシーケンス先頭に連結（Reference と同一）
    - 出力: 先頭 1 トークンを除去して output_proj
    """
    def __init__(self, num_actions, num_joints, latent_dim=512, num_layers=8): # latent_dimはTransformerの埋め込み次元、num_layersはTransformerエンコーダーの層数を指定するための引数で、モデルの表現力や計算コストに影響します。
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        # Reference の TimestepEmbedder: PE テーブルを t でインデックス → Linear-SiLU-Linear
        self.seq_pos_enc = PositionalEncoding(latent_dim, dropout=0.1)
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        ) # 時間埋め込み用のMLP
        self.pose = nn.Linear(num_joints * 3, latent_dim) # フレームごとの関節位置を埋め込み次元に射影
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True) #selfがつかない
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)

    def forward(self, x, t, action_class):
        """
        x: [B, F, J*3]  t: [B]  action_class: [B]
        returns: [B, F, J*3]
        """
        # 1. 時間埋め込み: PE[t] → MLP  (Reference TimestepEmbedder と同一)
        pe = cast(torch.Tensor, self.seq_pos_enc.pe)  # [max_len, 1, D]
        t_pe = pe[t].squeeze(1)           # [B, D]
        t_emb = self.time_embed(t_pe)     # [B, D]

        # 2. アクション埋め込み: action_class → Embedding
        c_emb = self.action_embedding(action_class)  # [B, D]

        # 3. 条件トークン 1 個 = time_emb + action_emb  (Reference と同一)
        emb = (t_emb + c_emb).unsqueeze(1)  # [B, 1, D]

        # 4. フレームを埋め込み次元へ射影
        x_emb = self.pose(x)  # [B, F, D]

        # 5. [cond_token, frame_0, ..., frame_{F-1}] に連結して PE を加算
        seq = torch.cat([emb, x_emb], dim=1)  # [B, F+1, D]
        seq = self.seq_pos_enc(seq)

        # 6. Transformer エンコーダー
        out = self.transformer(seq)

        # 7. 先頭の条件トークンを除去して出力射影
        out = out[:, 1:, :]        # [B, F, D]
        out = self.output_proj(out)  # [B, F, J*3]
        return out
