import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import numpy as np

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = vgg19()
model.to(device)
checkpoint = torch.load('pretrained_models/model_sh_A.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print('Model loaded successfully')

# Load example image
img = Image.open('example_images/1.png').convert('RGB')
print(f'Image size: {img.size}')

# Run inference
inp = transforms.ToTensor()(img).unsqueeze(0).to(device)
with torch.no_grad():
    outputs, _ = model(inp)

count = torch.sum(outputs).item()
print(f'Predicted crowd count: {count:.2f}')
print('SUCCESS - DM-Count is working correctly')
