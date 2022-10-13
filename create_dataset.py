import json
from tqdm import tqdm
from PIL import Image
import argparse
import os

import datasets


def get_paramters():
    parser = argparse.ArgumentParser(description='Upload dataset')
    parser.add_argument('--save_to_disk', action="store_true", default=False,
                        help='Save to disk instead of in HuggingFace')
    parser.add_argument('--dataset_name', type=str,
                        help='name of the dataset')
    parser.add_argument('--user_name', type=str,
                        help='user name in huggingface')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_paramters()
    path_caption = "./captions.json"
    path_data = "./data"

    # Put the whole images in jpeg instead of png
    with open(path_caption, "r") as f:
        res = json.load(f)
    os.makedirs(path_data.replace("./data", "./data_"),exist_ok=True)
    for x,y in tqdm(res.items()):
        im1 = Image.open(x).convert('RGB')
        im1.save(x.replace("/data/", "/data_/"))
        
    res_ = {x.replace("/data/", "/data_/"):y for x,y in res.items()}
    dataset_dict = {
        'image': list(res_.keys()),
        'text': list(res_.values())
    }
    features = datasets.Features({
        'text': datasets.Value('string'),
        'image': datasets.Image(),
    })

    # Create dataset
    dataset = datasets.Dataset.from_dict(dataset_dict, features, split="train")

    # Save dataset
    # Localy
    if args.save_to_disk:
        dataset.save_to_disk(f'./datasets/{args.dataset_name}')
    # Push dataset to HuggingFace 
    else:
        dataset.push_to_hub(f'{args.user_name}/{args.dataset_name}')
