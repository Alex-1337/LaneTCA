import torch
import torch.nn as nn

from .position import PositionEmbeddingSine
from .math import generate_permute_matrix


class LSTN(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_wise_id_bank = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 5), stride=(2, 2),padding=(1,2))
        
        self.id_dropout = nn.Dropout(0.0, True)
        self.pos_generator = PositionEmbeddingSine(64, normalize=True)


    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x) 
        return pos_emb

    def assign_identity(self, one_hot_mask, batch_size, enc_hw):      
        id_shuffle_matrix = generate_permute_matrix(2, batch_size, gpu_id=0)
        one_hot_mask = torch.einsum('bohw, bot->bthw', one_hot_mask, id_shuffle_matrix)   
        id_emb = self.get_id_emb(one_hot_mask).view(batch_size, -1, enc_hw).permute(2, 0, 1) 

        return id_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x) 
        id_emb = self.id_dropout(id_emb) 
        return id_emb

