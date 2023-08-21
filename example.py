import torch
from palme.model import PalmE

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = PalmE()
output = model(text, img)