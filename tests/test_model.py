import torch
from model import MDM

def test_forward_output_shape():
    """MDM の forward 関数が、入力に対して正しい形状の出力を返すことを確認するテスト関数。"""
    #arrange(準備) - モデルとダミーデータを作成
    num_joints = 22
    num_actions = 10
    batch_size = 4
    frames = 60
    mdm_model = MDM(num_actions, num_joints)
    
    x_t = torch.randn(batch_size, frames, num_joints*3)  # [B, F, J*3]
    t = torch.randint(0, 1000, (batch_size,))  # ランダムな時間ステップ
    action_class = torch.randint(0, num_actions, (batch_size,))  # ランダムなアクションクラス
    #act(実行) - モデルの forward 関数を呼び出す
    predicted_x_0 = mdm_model(x_t, t, action_class)  # モデルの出力
    #assert(検証) - 出力の形状が入力と同じであることを確認
    assert predicted_x_0.shape == x_t.shape, f"Expected shape {x_t.shape}, but got {predicted_x_0.shape}"
    print("test_forward_output_shape passed.")

def test_forward_no_grad():
    """MDM の forward 関数が、勾配を計算せずに出力を返すことを確認するテスト関数。"""
    #arrange(準備) - モデルとダミーデータを作成
    num_joints = 22
    num_actions = 10
    batch_size = 4
    frames = 60
    mdm_model = MDM(num_actions, num_joints)
    
    x_t = torch.randn(batch_size, frames, num_joints*3)  # [B, F, J*3]
    t = torch.randint(0, 1000, (batch_size,))  # ランダムな時間ステップ
    action_class = torch.randint(0, num_actions, (batch_size,))  # ランダムなアクションクラス

    with torch.no_grad():  # 勾配を計算しないコンテキスト
        #act(実行) - モデルの forward 関数を呼び出す
        predicted_x_0 = mdm_model(x_t, t, action_class)  # モデルの出力
        #assert(検証) - 出力が勾配を必要としないことを確認
        assert not predicted_x_0.requires_grad, "Expected output to not require grad"
    print("test_forward_no_grad passed.")