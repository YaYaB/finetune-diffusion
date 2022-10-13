---
license: cc-by-nc-sa-4.0
annotations_creators:
- machine-generated
language:
- en
language_creators:
- other
multilinguality:
- monolingual
pretty_name: 'Subset of Magic card (Creature only) BLIP captions'
size_categories:
- n<1K
source_datasets:
- YaYaB/magic-creature-blip-captions
tags: []
task_categories:
- text-to-image
task_ids: []
---

# Disclaimer
This was inspired from https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions

# Dataset Card for A subset of Magic card  BLIP captions

_Dataset used to train [Magic card text to image model](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning)_

BLIP generated captions for Magic Card images collected from the web. Original images were obtained from [Scryfall](https://scryfall.com/) and captioned with the [pre-trained BLIP model](https://github.com/salesforce/BLIP).

For each row the dataset contains `image` and `text` keys. `image` is a varying size PIL jpeg, and `text` is the accompanying text caption. Only a train split is provided.


## Examples


![pk1.jpg](https://api.scryfall.com/cards/354de08d-41a8-4d6c-85d6-2413393ac181?format=image)
> A woman holding a flower

![pk10.jpg](https://api.scryfall.com/cards/95608d51-9ec0-497c-a065-15adb7eff242?format=image)
> two knights fighting

![pk100.jpg](https://api.scryfall.com/cards/42d3de03-9c3d-42f6-af34-1e15afb10e4f?format=image)
> a card with a unicorn on it

## Citation

If you use this dataset, please cite it as:

```
@misc{yayab2022onepiece,
      author = {YaYaB},
      title = {Magic card creature split BLIP captions},
      year={2022},
      howpublished= {\url{https://huggingface.co/datasets/YaYaB/magic-blip-captions/}}
} 
```