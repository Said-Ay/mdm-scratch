import torch

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 1. beta を作る: beta_start から beta_end までを num_timesteps 分割。
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # 2. alpha を作る: alpha = 1 - beta。
        self.alphas = 1.0 - self.betas

        # 3. alpha_bar を作る: alpha の累積積。
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _extract(self, a, t, x_shape):
        """
        配列 a から時間 t の要素を取り出し、x の次元に合わせて末尾に次元を追加する。
        """
        # バッチサイズ分の数字を取り出す
        out = a.to(t.device)[t]
        # x_shape が [B, F, D] なら、out を [B, 1, 1] に引き伸ばす
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out       
    def add_noise(self, x_start, noise, t):
        """
        x_start に noise を加えて、t ステップ目の x_t を生成する。
        """
        # 1. t ステップ目の alpha_bar を取り出す
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        
        # 2. alpha_bar_t の平方根を計算
        sqrt_alphas_cumprod = torch.sqrt(alpha_bar_t)
        
        # 3. 1 - alpha_bar_t の平方根を計算（負値を clamp で防ぐ）
        sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))

        # 4. x_t を合成する
        x_t = (sqrt_alphas_cumprod * x_start) + (sqrt_one_minus_alphas_cumprod * noise)

        return x_t
