####### PALM
import torch 
import torch.nn as nn
from torch.optim import Adam
from palm_rlhf_pytorch import PaLM
import open_clip
import re

vit_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')





def token_is_text(token):
    return isinstance(token, str)


class ViTProjector(nn.Module):
    def __init__(self, vit_model, input_dim, output_dim):
        super(ViTProjector, self).__init__()
        self.vit_model = vit_model
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        encoded_observation = self.vit_model(x)
        return self.linear(encoded_observation)
    
class PALME(nn.Module):
    def __init__(self, LLM,  ViT_model):
        super(PALME, self).__init__()
        self.LLM = LLM
        self.ViT_model = ViT_model


    def forward(self, continuous_observations, text):
        embeddings = []
        for token in text:
            if token_is_text(token):
                embeddings.append(self.LLM.embed(token))
            else:
                observation = continuous_observations[token.observation_index]
                embeddings.append(self.projector(observation))
        embeddings = torch.stack(embeddings)
        output = self.LLM(embeddings)
        return output
        
#init the pre trained lm

LLM = PaLM(
    num_tokens=50304,
    dim=2048,
    depth=16,
    dim_head=128,
    heads=8,
    flash_attn=True,
    qk_rmsnorm=False,
)

#INIT projector
projector = ViTProjector(vit_model=vit_model, input_dim=vit_model.config.hidden_size, output_dim=LLM.config.dim)


#create the palme model
palme_model = PALME(LLM, projector)
