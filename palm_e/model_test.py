import torchvision.transforms as transforms
from model import PALME_Tokenizer, PALME
from PIL import Image

# Random text
text = "This is a sample text"

# Load a sample image
image_path = "galaxy-andromeda.jpeg"
try:

    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

except Exception as e:
    print(f"Error loading and processing image: {e}")

# Instantiate tokenizer and tokenize inputs
try:
        
    tokenizer = PALME_Tokenizer()
    tokenized_inputs = tokenizer.tokenize({"target_text": text, "image": image})
except Exception as e:
    print(f"Error tokenzing inputs: {e}")

try:
        
    # Instantiate model
    model = PALME()
except Exception as e:
    print(f"Error initlizing model: {e}")

try:
        

    # Call the forward function and print the output
    output = model.forward(
        text_tokens=tokenized_inputs["text_tokens"],
        images=tokenized_inputs["images"]
    )

    print(f'output: {output}')

except Exception as e:
    print(f"Output call pass: {e}")