import torchvision.transforms as transforms
from model import PALME_Tokenizer, PALME
from PIL import Image

# Random text
text = "This is a sample text"

# Load a sample image
image_path = "galaxy-andromeda.jpeg"
image = Image.open(image_path)
transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Instantiate tokenizer and tokenize inputs
tokenizer = PALME_Tokenizer()
tokenized_inputs = tokenizer.tokenize({"target_text": text, "image": image})

# Instantiate model
model = PALME()

# Call the forward function and print the output
output = model.forward(
    text_tokens=tokenized_inputs["text_tokens"],
    images=tokenized_inputs["images"]
)

print(f'output: {output}')