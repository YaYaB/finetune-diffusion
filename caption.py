import json
import requests
from tqdm import tqdm
from glob import glob
from PIL import Image
import os
import sys

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.append("")

from models.blip import blip_decoder


def load_image(image_size, device, path_image=None):
    if path_image == None:
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    else:
        raw_image = Image.open(path_image).convert('RGB')

    _,_ = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def load_blip_model(device):
    # Load BLIP model
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    return model, image_size


if __name__ == '__main__':
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, image_size = load_blip_model(device)

    # Loop over images, caption those and create a "./captions.json" 
    base_name = os.path.abspath(os.path.dirname(__file__))
    path_data, path_output = os.path.join(base_name, "./data"), os.path.join(base_name, "./captions.json")
    res = {}
    for elem in tqdm(glob(f"{path_data}/*")):
        image = load_image(image_size=image_size, device=device, path_image=elem)
        with torch.no_grad():
            # beam search
            res[elem] = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)[0]

    with open(path_output, "w") as f:
        json.dump(res, f)
