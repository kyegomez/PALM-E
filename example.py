import torch
from palme.model import PalmE

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = PalmE()
output = model(img, caption_tokens)