import os
import datetime
import dataclasses
from typing import Literal

from accelerate import Accelerator
from transformers import HfArgumentParser, SiglipImageProcessor, SiglipVisionModel
from transformers import SiglipTextModel, AutoTokenizer
from PIL import Image
import json
import itertools

from flux.pipeline import USOPipeline, preprocess_ref
from emostyle.models import EmoStyle
from emostyle.datasets import EMOTION2ID
import torch
import numpy as np
from flux.modules.layers import SigLIPMultiFeatProjModel
from safetensors.torch import load_file
from pathlib import Path


def load_sft(ckpt_path: str | Path, target_device: torch.device, dtype: torch.dtype | None = None):
    ckpt_path = str(ckpt_path)
    # Always load on CPU to avoid safetensors device parsing issues
    state = load_file(ckpt_path, device="cpu")
    if dtype is not None:
        for k, v in state.items():
            state[k] = v.to(dtype)
    return {k: v.to(target_device) for k, v in state.items()}


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 1024
    height: int = 1024
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 45
    emo_strength: float = 1.0
    seed: int = 126
    emotion: str = "amusement"
    save_path: str = None
    mapper_ckpt: str = None
    dict_ckpt: str = None
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 128
    data_resolution: int = 1024
    pe: Literal['d', 'h', 'w', 'o'] = 'd'
    hf_download: bool = True
    query_mode: str = "transformer"  
    transformer_heads: int = 8
    transformer_layers: int = 4

def main(args: InferenceArgs):
    accelerator = Accelerator()

    pipeline = USOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank,
        hf_download=args.hf_download,
    )

    style_proj_model = SigLIPMultiFeatProjModel(
        siglip_token_nums=729,
        style_token_nums=64,
        siglip_token_dims=1152,
        hidden_size=3072,
        context_layer_norm=True,
    ).to(accelerator.device)
    ckpt_path = "USO/uso_flux_v1.0/projector.safetensors"
    raw_state = load_sft(ckpt_path, target_device=torch.device("cpu"))
    print(f"Loaded SigLIP projector state from {ckpt_path}, keys: {list(raw_state.keys())}")

    remapped = {}
    for k, v in raw_state.items():
        if k.startswith("feature_embedder."):
            remapped[k[len("feature_embedder."):]] = v.to(accelerator.device)
        else:
            remapped[k] = v.to(accelerator.device)
    style_proj_model.load_state_dict(remapped, strict=True)

    for p in style_proj_model.parameters():
        p.requires_grad_(False)

    # Load SigLIP vision
    siglip_path = "weights/siglip"
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(siglip_path)
    siglip_model.eval().to(accelerator.device)
    print("Loaded SigLip vision model from", siglip_path)

    mapper = EmoStyle(
        in_dim=8, img_dim=1152, token_nums=192, emb_dim=3072,
        codebook_size=60, query_dim=512,
        query_mode=args.query_mode,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
    )
    mapper.load_state_dict(torch.load(args.mapper_ckpt, map_location="cpu"))
    mapper.StyleDictionary.load_state_dict(torch.load(args.dict_ckpt, map_location="cpu"))
    mapper = mapper.to(accelerator.device).to(next(pipeline.model.parameters()).dtype)
    mapper.eval()

    save_path = f"{args.save_path}/inference"


    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
            
    if args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
    else:
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):
        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue
                
        image_name = os.path.basename(data_dict["image_paths"][0])
        image_name = image_name.split(".")[0]

        ref_imgs = [
            Image.open(img_path).resize((args.data_resolution, args.data_resolution))
            for img_path in data_dict["image_paths"]
        ]
        if args.ref_size==-1:
            args.ref_size = 512 if len(ref_imgs)==1 else 320

        ref_imgs = [preprocess_ref(img, args.ref_size) for img in ref_imgs]



        tgt_onehot = np.zeros(8)
        tgt_onehot[int(EMOTION2ID[data_dict["prompt"]])] = 1
        tgt_onehot = torch.from_numpy(tgt_onehot).to(torch.float).to(accelerator.device)
        emo_embeds = tgt_onehot.to(next(pipeline.model.parameters()).dtype)
        emo_embeds = emo_embeds.unsqueeze(0)  # (1, 8)  



        siglip_content_inputs = siglip_processor(
            images=ref_imgs[0], return_tensors="pt"
        ).to(accelerator.device)
        siglip_content_outputs = siglip_model(**siglip_content_inputs, output_hidden_states=True)
        content_tokens = siglip_content_outputs.last_hidden_state.to(next(pipeline.model.parameters()).dtype)  # (B,729,1152)


        emostyle_emb, indice, _ = mapper(emo_embeds, content_tokens)  # (B,192,3072)
        emostyle_emb = emostyle_emb * args.emo_strength  # scale up for stronger effect


        image_gen = pipeline(
            prompt=" ",
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs,
            pe=args.pe,
            emostyle_inputs=emostyle_emb,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        os.makedirs(save_path, exist_ok=True)

        image_gen.save(os.path.join(save_path, f"{image_name}_{args.emotion}_{indice.item()}_{args.seed}_{datetime.datetime.now().strftime('%Y%m%d')}.png"))
     

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)