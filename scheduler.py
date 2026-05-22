import torch

class NoiseScheduler:
    """ノイズスケジューラーを実装。拡散過程のノイズ量を管理するクラス。"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 1. beta を作る: beta_start から beta_end までを num_timesteps 分割。
        #betaはノイズの強さを表すパラメータで、時間とともに増加する。これにより、拡散過程が進むにつれてノイズが強くなり、最終的には完全なノイズになる。
        # β_t = linspace(β_start, β_end, T)  ← 線形スケジュール
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # 2. alpha を作る: # α_t = 1 − β_t
        # alphaは、逆に元のデータをどれだけ保持するかを表すパラメータで、時間とともに減少する。これにより、拡散過程が進むにつれて元のデータが徐々に失われていく。
        self.alphas = 1.0 - self.betas

        # 3. alpha_bar を作る: alpha の累積積。
        # ā_t = ∏_{s=1}^{t} α_s  （t=0 で ā→1、t=T で ā→0）
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        #ā_{t-1} の配列(=alphas_cumprodを一つずらした配列)。t=0 のときは ā_{-1} = 1.0（ノイズゼロ）と定義するため先頭に 1.0 を追加。
        #ā_{t-1} は、サンプリング時に必要なパラメータで、１ステップ前のαの累積積を表す。サンプリング過程で前のステップの状態を導く際に使用される。
        #t=0 のときは、ā_{-1} = 1.0(ノイズゼロ) と定義されるため、先頭に 1.0 を追加して配列を作成する
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

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
        # DDPM Eq.(4): q(x_t | x_0) = N( sqrt(ā_t)・x_0,  (1−ā_t)・I )
        #ただしq(x_t | x_0) は、t ステップ目の x_t が、x_0 からどのように生成されるかを表す(既知の)確率分布

        # 再パラメータ化(確率的なサンプリングを、"決定的な式 + ランダムノイズ” に書き換える): x_t = sqrt(ā_t)・x_0 + sqrt(1−ā_t)・ε,  ε ~ N(0, I)
        #つまり、t ステップ目の x_t は、x_0 を sqrt(ā_t) でスケーリングしたものと、ノイズを sqrt(1−ā_t) でスケーリングしたものの和として生成される。
        
        # 1. t ステップ目の alpha_bar を取り出す
        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        
        # 2. alpha_bar_t の平方根を計算
        sqrt_alphas_cumprod = torch.sqrt(alpha_bar_t)
        
        # 3. 1 - alpha_bar_t の平方根を計算（負値を clamp で防ぐ）
        sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))

        # 4. x_t を合成する
        x_t = (sqrt_alphas_cumprod * x_start) + (sqrt_one_minus_alphas_cumprod * noise)

        return x_t

    def step(self, pred_x0,x_t,t):
        """
        予測された x_0（pred_x0）と現在の x_t から、t-1 ステップ目の x_{t-1} を計算する。
        """
        # DDPM Eq.(11): p(x_{t-1} | x_t) = N( μ_θ(x_t, t), σ_t^2 I )
        #ただしp(x_{t-1} | x_t) は、t ステップ目の x_t から、t-1 ステップ目の x_{t-1} がどのように生成されるかを表す(未知の)確率分布

        # 1. t ステップ目の alpha_bar と alpha_bar_prev を取り出す
        #pred_x0.shapeは [B, F, D] の形状を持つため、alpha_bar_t と alpha_bar_prev は [B, 1, 1] の形状になり、pred_x0 とブロードキャスト可能になる。
        alpha_bar_t = self._extract(self.alphas_cumprod, t, pred_x0.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, pred_x0.shape)

        # 2. t ステップ目の beta を取り出す
        beta_t = self._extract(self.betas, t, pred_x0.shape)

        # 3. μ_θ(x_t, t) を計算する
        mu_theta = (torch.sqrt(alpha_bar_prev) * beta_t * pred_x0 + torch.sqrt(alpha_bar_t) * (1 - alpha_bar_prev) * x_t) / (1 - alpha_bar_t)

        # 4. 事後分散 β̃_t を計算する（リバース1ステップ分の不確かさ）
        # β̃_t = β_t * (1 − ā_{t-1}) / (1 − ā_t)  ← β_t より常に小さい（x_0 の予測情報で不確かさが減る）
        sigma_squared = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

        # 5. x_{t-1} をサンプリングする
        # t == 0 のときはノイズなし（最後のステップは決定的に x_0 を返す）
        noise = torch.randn_like(pred_x0)
        mask =  (t > 0).float().view(-1, 1, 1)  # [B, 1, 1]  # t==0 なら 0.0、それ以外は 1.0
        x_prev = mu_theta + mask * torch.sqrt(torch.clamp(sigma_squared, min=1e-20)) * noise

        return x_prev