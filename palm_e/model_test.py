import torchvision.transforms as transforms
from model import PALME_Tokenizer, PALME
from PIL import Image

#random text
text = "This is a sample text"


#laod a sample image
image_path = "galaxy-andromeda.jpeg"
image = Image.open(image_path)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
image = transform(image)
image = image.unsqueeze(1) #add batch dimension


#instantiate tokenzier and tokenize inputs
tokenizer = PALME_Tokenizer()
tokenized_inputs = tokenizer.tokenize({"target_text": text, "image": image})



model = PALME()


#call the forward function and prunt the output
output = model.forward(
    text_tokens=tokenized_inputs["text_tokens"],
    images = tokenized_inputs["images"]
)

print(output)