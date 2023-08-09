import torch
from palme.model import PALME

# Create a sample text token tensor
text_tokens = torch.randint(0, 32002, (1, 50), dtype=torch.LongTensor)

# Create a sample image tensor
images = torch.randn(1, 3, 224, 224)

# Instantiate the model
model = PALME()

# Pass the sample tensors to the model's forward function
output = model.forward(
    text_tokens=text_tokens,
    images=images
)

# Print the output from the model
print(f"Output: {output}")