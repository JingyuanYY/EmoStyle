import torch
import json
import numpy as np
import torchvision
from torchvision import transforms
from transformers import CLIPImageProcessor
from PIL import Image
import random


EMOTION2ID = {
    "amusement": 0,
    "anger": 1,
    "awe": 2,
    "contentment": 3,
    "disgust": 4,
    "excitement": 5,
    "fear": 6,
    "sadness": 7,
}

class MyDataset(torch.utils.data.Dataset):
    cls_num = 8
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "Emotion"}]
        self.data_store = []
        self.origin_img_path = []
        self.edited_img_path = []
        self.targets = []
        for item in self.data:
            self.targets.append(EMOTION2ID[item["edit_prompt"]])
            self.origin_img_path.append(item["original_image"])
            self.edited_img_path.append(item["edited_image"])
        # print(self.targets)
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.cls_num)]
        # print(cls_num_list_old)

        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.cls_num)]
        for i in range(self.cls_num):
            self.class_map[sorted_classes[i]] = i

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.cls_num)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.cls_num)]

        # for emotion in EMOTION_LIST:
        #     self.targets.extend([EMOTION2ID[emotion]] * EMOTION_NUM[emotion])
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

        # preprocess
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.preprocess = transforms.Compose([
                        transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalize,
                    ])

    def __len__(self):
        length = len(self.data)
        print(f"Dataset length: {length}")  # Debugging statement
        return len(self.data)
        
    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            print(f"Index {idx} is out of bounds for dataset of length {len(self.data)}")  # Debugging statement
            raise IndexError("Index out of bounds")
        item = self.data[idx] 
        text = item["edit_prompt"]
        
        origin_image_file = self.origin_img_path[idx]
        edited_image_file = self.edited_img_path[idx]
        
        # read image
        origin_raw_image = Image.open(origin_image_file)
        # crop image to 224x224
        # origin_raw_image = origin_raw_image.crop((0, 0, self.size, self.size))
        transformed_origin_image = self.transform(origin_raw_image.convert("RGB"))
        origin_image = origin_raw_image.convert("RGB")
        origin_image = origin_image.crop((0, 0, self.size, self.size))
        edited_raw_image = Image.open(edited_image_file)
        transformed_edited_image = self.transform(edited_raw_image.convert("RGB"))
        edited_image = edited_raw_image.convert("RGB")
        edited_image = edited_image.crop((0, 0, self.size, self.size))
        # gray_style_image = self.preprocess(raw_image.convert("L")).unsqueeze(0)
        origin_clip_image = self.clip_image_processor(images=origin_raw_image, return_tensors="pt").pixel_values
        edited_clip_image = self.clip_image_processor(images=edited_raw_image, return_tensors="pt").pixel_values

        emo_score = item["distribution_value"]
        # convert to tensor
        emo_score = torch.from_numpy(np.array(emo_score)).to(torch.bfloat16)
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        tgt_onehot = np.zeros(8)
        tgt_onehot[int(EMOTION2ID[item["edit_prompt"]])] = 1
        tgt_onehot = torch.from_numpy(tgt_onehot).to(torch.bfloat16)
        
        return {
            "origin_image": origin_image,
            "edited_image": edited_image,
            "transformed_origin_image": transformed_origin_image,
            "transformed_edited_image": transformed_edited_image,
            "text_input_ids": text_input_ids,
            "origin_clip_image": origin_clip_image,
            "edited_clip_image": edited_clip_image,
            "drop_image_embed": drop_image_embed,
            "txt": text,
            "tgt_onehot": tgt_onehot,
            "emo_score": emo_score,
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    transformed_origin_images = torch.stack([example["transformed_origin_image"] for example in data])
    origin_images = [example["origin_image"] for example in data]
    transformed_edited_images = torch.stack([example["transformed_edited_image"] for example in data])
    edited_images = [example["edited_image"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    origin_clip_images = torch.cat([example["origin_clip_image"] for example in data], dim=0)
    edited_clip_images = torch.cat([example["edited_clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    txts = [example["txt"] for example in data]
    tgt_onehots = torch.stack([example["tgt_onehot"] for example in data])
    emo_scores = torch.stack([example["emo_score"] for example in data])

    return {
        "origin_images": origin_images,
        "edited_images": edited_images,
        "transformed_origin_images": transformed_origin_images,
        "transformed_edited_images": transformed_edited_images,
        "text_input_ids": text_input_ids,
        "origin_clip_images": origin_clip_images,
        "edited_clip_images": edited_clip_images,
        "drop_image_embeds": drop_image_embeds,
        "txts": txts,
        "tgt_onehots": tgt_onehots,
        "emo_scores": emo_scores,
    }