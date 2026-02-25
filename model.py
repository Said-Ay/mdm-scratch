import torch
import torch.nn as nn
class MDM(nn.Module):
    def __init__(self , num_actions, num_joints , latent_dim = 512 , num_layers = 8):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions , latent_dim)
        self.time_embedding = nn.Sequential(nn.Linear(1 , latent_dim),nn.SiLU(),nn.Linear(latent_dim , latent_dim))
        self.pose = nn.Linear(num_joints*3 , latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, num_joints * 3)