import torch
from palme.model import PalmE

# Create a sample text token tensor
text_tokens = torch.randint(0, 32002, (1, 50), dtype=torch.long)

# Create a sample image tensor
images = torch.randn(1, 3, 224, 224)

# Instantiate the model
model = PalmE()

# Pass the sample tensors to the model's forward function
output = model.forward(
    text_tokens=text_tokens,
    images=images
)

# Print the output from the model
print(f"Output: {output}")




# ################################################################
# # Create dummy text token tensors.
# # Let's assume your text tokens are of size [1, 116, X], for demonstration purposes.
# dummy_text_tokens = torch.randint(0, 50304, (1, 2048)).cuda()

# # Create dummy image tensors.
# # The required size for images for the ViT model in CLIP is [batch_size, 3, 224, 224].
# dummy_images = torch.randn(1, 3, 224, 224)

# # Instantiate the PALME model
# palme_model = PALME()

# # Pass the dummy text tokens and image tensors to PALME's forward function
# output = palme_model(dummy_text_tokens, dummy_images)

# # Print the output
# print(output)