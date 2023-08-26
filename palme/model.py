
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPProcessor

from palme.transformer import (
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)


class PalmeETokenizer:
    def __init__(self):
        try:

            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token ="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )

            self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])
        except Exception as e:
            print(f"Error init tokenizer: {e}")


    def tokenize_texts(self, texts):
        try:

            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            print(f"Error tokenizing texts: {e}")

        

    def tokenize_images(self, images):
        try:
                
            tokenized_images = self.processor(images=images, return_tensors="pt").pixel_values
            print(f"Tokenized image: {tokenized_images.shape}")
            return tokenized_images
        
        except Exception as e:
            print(f"Error tokenizing texts: {e}")

    def tokenize(self, sample):
        try:
            
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        
        except Exception as e:
            print(f"Error during tokenization {e}")
        


class PalmE(torch.nn.Module):
    def __init__(self, 
                 image_size=256, 
                 patch_size=32, 
                 encoder_dim=512, 
                 encoder_depth=6, 
                 encoder_heads=8,
                 num_tokens=20000, 
                 max_seq_len=1024, 
                 decoder_dim=512, 
                 decoder_depth=6, 
                 decoder_heads=8, 
                 alibi_num_heads=4,
                 use_abs_pos_emb=False,
                 cross_attend=True,
                 alibi_pos_bias=True,
                 rotary_xpos=True,
                 attn_flash=True,
                 qk_norm=True):
        
        super(PalmE, self).__init__()
        
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads
            )
        )

        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            )
        )

    def forward(self, img, text):
        try:    
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise


