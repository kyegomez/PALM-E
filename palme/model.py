
import bitsandbytes
import torch
import torch.nn as nn
from flamingo_pytorch import PerceiverResampler
from palme.palm import PaLM
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

import torch.nn.functional as F

class PALMETokenizer:
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
        
class PalmE(nn.Module):
    def __init__(self,
                 num_tokens: int = 50304,
                 dim: int = 2048,
                 depth: int = 16,
                 dim_head: int = 128,
                 heads: int = 8,
                 flash_attn=True,
                 qk_rmsnorm=False):
        super(PalmE, self).__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head

        self.heads = heads
        self.flash_attn = flash_attn
        self.qk_rmsnorm = qk_rmsnorm
        
        try:

            self.vit_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model

            self.embed = bitsandbytes.nn.modules.Embedding(
                self.num_tokens,
                self.dim,
                padding_idx=1
            )



            self.output_projection = torch.nn.Linear(
                self.dim, self.num_tokens, bias=False
            )
            torch.nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.dim**-0.5
            )

            self.decoder = PaLM(
                num_tokens=self.num_tokens,
                dim=self.dim,
                depth=self.depth,
                dim_head=self.dim_head,
                heads=self.heads,
                flash_attn=self.flash_attn,
                qk_rmsnorm=self.qk_rmsnorm,
            )

            self.perceive = PerceiverResampler(
                dim = 1024,
                depth = 2,
                dim_head = 8,
                num_latents = 50,
                num_media_embeds = 257
            )

            self.image_proj = torch.nn.Linear(1024, self.num_tokens, bias=False)
            torch.nn.init.normal_(
                self.image_proj.weight, mean=0, std=self.num_tokens**-0.5
            )

        except Exception as e:
            print(f"Error initlizing palme components: {e}")
    def forward(self, text_tokens, images):
        # Explicitly cast text_tokens to int64 (long)
        print(f"Original text tokens type: {text_tokens.dtype}")
        
        text_tokens = text_tokens.to(torch.long)
        print(f'text tokens shape conversion to torch long: {text_tokens.dtype}')

        # Print the initial shape of text tokens for clarity
        print("Initial text tokens shape:", text_tokens.shape)
        print(f"Initial text tokens dtype {text_tokens.dtype}")
        
        # Process images with the VIT model
        images = self.vit_model(pixel_values=images)["last_hidden_state"]
        print("Images after VIT model:", images.shape)
        print(f"Images dtype: {images.dtype}")
        
        # Reshape images with perceive and project
        images = self.perceive(images).squeeze(1)
        print("Images after PerceiverResampler:", images.shape)
        print(f"Images dtype: {images.dtype}")
        
        images = self.image_proj(images)
        print("Images after image_proj:", images.shape)
        print(f"Images dtype: {images.dtype}")

        # Process the text tokens
        model_input = self.decoder(text_tokens)
        print("Text tokens after decoding:", model_input.shape)
        print(f"Model input type: {model_input.dtype}")

        # Check dimension equality before concatenation
        if model_input.shape[:2] != images.shape[:2]:
            raise ValueError("Mismatched dimensions between images and text tokens")

        # Convert images to torch.int64 for concatenation
        images = images.to(torch.int64)

        # Concatenate along the last dimension
        concatenated_input = torch.cat([model_input, images], dim=-1)
        print("Shape after concatenation:", concatenated_input.shape)
        print(f"Model input type after concatenation: {concatenated_input.dtype}")

        # Pass concatenated tensor through the decoder
        output = self.decoder(concatenated_input)[0]
        print(f"Output dtype: {output.dtype}")
        return output
