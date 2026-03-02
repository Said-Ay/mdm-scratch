import torch

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 1. beta (ベータ) を作る
        # torch.linspace を使うと、beta_start から beta_end まで、
        # num_timesteps 個に均等に分けた数字のリストを一発で作れるよ。
        self.betas = torch.linspace(beta_start,beta_end,num_timesteps) # ← ここに書いてみて！

        # 2. alpha (アルファ) を作る
        # alpha は 1 から beta を引いたもの。
        self.alphas = 1.0 - self.betas # ← ここに書いてみて！

        # 3. alpha_bar (アルファ・バー) を作る
        # 今までの alpha を順番に掛け合わせたもの（累積積）。
        # ヒント: torch.cumprod(これ, dim=0) っていう便利な関数があるんだ。
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # ← ここに書いてみて！

    def _extract(self, a, t, x_shape):
        """
        表(a)から、時間(t)の場所の数字を取り出して、
        データ(x)の次元に合わせて後ろにダミーの次元を追加する便利ツール。
        """
        # バッチサイズ分の数字を取り出す
        out = a.to(t.device)[t]
        # x_shape が [B, F, D] なら、out を [B, 1, 1] に引き伸ばす
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out       
    def add_noise(self, x_start, noise, t):
        """
        綺麗なデータ(x_start)にノイズ(noise)を混ぜて、tステップ目の状態(x_t)にする
        """
        # 1. 魔法の表から、tステップ目の alpha_bar を【1回だけ】取り出す
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        
        # 2. 取り出した alpha_bar_t を使い回してルートを計算する
        sqrt_alphas_cumprod = torch.sqrt(alpha_bar_t)
        
        # 3. ルートの中身がマイナスにならないように clamp でガードする
        # torch.clamp(数字, min=0.0) で、もしマイナスになっても強制的に0にしてくれる
        sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))

        # 4. x_t を作る
        x_t = (sqrt_alphas_cumprod * x_start) + (sqrt_one_minus_alphas_cumprod * noise)

        return x_t
# ---------------------------------------------------------
# テスト運転用コード（動くか確認したら消してOK！）
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. スケジューラの準備
    scheduler = NoiseScheduler()
    
    # 2. ダミーデータを作る
    batch_size = 4
    frames = 60
    dims = 66
    
    # 綺麗なデータ(x0)と、足し込む用のノイズ(epsilon)
    dummy_x_start = torch.randn(batch_size, frames, dims)
    dummy_noise = torch.randn_like(dummy_x_start)
    
    # バッチごとの時間ステップ t (例: 10歩目, 50歩目, 500歩目, 999歩目)
    dummy_t = torch.tensor([10, 50, 500, 999])
    
    # 3. ノイズを混ぜてみる！
    noisy_x = scheduler.add_noise(dummy_x_start, dummy_noise, dummy_t)
    
    print("元の綺麗なデータの形:", dummy_x_start.shape)
    print("ノイズ入りデータの形:", noisy_x.shape)