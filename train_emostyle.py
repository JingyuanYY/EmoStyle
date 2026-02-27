import torch
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, set_seed
from transformers import CLIPTokenizer, SiglipVisionModel, SiglipImageProcessor
from emostyle.datasets import MyDataset, collate_fn
from emostyle.classifier_sampler import ClassAwareSampler
from emostyle.models import EmoStyle
import argparse
from pathlib import Path

import time
from torch.utils.tensorboard import SummaryWriter
from flux.modules.layers import SigLIPMultiFeatProjModel
from safetensors.torch import load_file
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from flux.util import load_ae, load_clip, load_t5, load_flow_model, load_flow_model_only_lora
from flux.sampling import prepare_multi_ip, get_noise, get_schedule
import torchvision.transforms.functional as TVF

def get_models(name: str, device, offload: bool = False):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model_only_lora(name, device="cpu", lora_rank=128)
    vae = load_ae(name, device="cpu" if offload else device)

    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser()
    emotion = "amusement"
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1019)
    parser.add_argument("--optimize_target", type=str, default="transformer", choices=["transformer", "codebook", "both"])
    parser.add_argument("--data_resolution", type=int, default=1024)
    parser.add_argument("--data_root", type=str, default="data/emostyle")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save checkpoints. Defaults to output_dir if not set.")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--init_folder", type=str, default=None,
                        help="Folder for pre-initialization. Supports image files or folders.")
    parser.add_argument("--init_batch", type=int, default=32, help="Batch size for initialization.")
    parser.add_argument("--init_mode", type=str, default="similarity")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--codebook_ckpt",
        type=str,
        default=f"weights/StyleDictionary/{emotion}.bin",
        help="Path to the pretrained codebook checkpoint.",
    )
    parser.add_argument("--lr_transformer", type=float, default=1e-4, help="LR for transformer/query path")
    parser.add_argument("--lr_codebook", type=float, default=1e-4, help="LR for codebook (tags/values)")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--lock_codebook_after_init", action="store_true",
                        help="Freeze codebook so only pre-initialized entries are selectable.")
    parser.add_argument("--init_sim_threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for selecting diverse initialization images.")
    parser.add_argument("--init_max_samples", type=int, default=5000,
                        help="Max number of images to scan from dataset for initialization.")
    parser.add_argument("--loss_type", type=str, default="combo",
                        choices=["mse", "cos", "combo"],
                        help="Use MSE, cosine similarity, or a weighted combination.")
    parser.add_argument("--mse_weight", type=float, default=0.0, help="Weight for MSE loss when using combo.")
    parser.add_argument("--cos_weight", type=float, default=1.0, help="Weight for cosine loss when using combo.")
    parser.add_argument("--no_norm_before_loss", action="store_true",
                        help="Disable LayerNorm before computing losses.")
    parser.add_argument("--cosine_mode", type=str, default="mean_token",
                        choices=["flat", "mean_token"],
                        help="Cosine across flattened sequence or mean over token-wise cosine.")
    args = parser.parse_args()
    return args


def load_sft(ckpt_path: str | Path, target_device: torch.device, dtype: torch.dtype | None = None):
    ckpt_path = str(ckpt_path)
    # Always load on CPU to avoid safetensors device parsing issues
    state = load_file(ckpt_path, device="cpu")
    if dtype is not None:
        for k, v in state.items():
            state[k] = v.to(dtype)
    return {k: v.to(target_device) for k, v in state.items()}

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    writer = SummaryWriter(f'runs/{args.output_dir}')
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],  
    )
    seed = args.seed
    set_seed(seed)
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("weights/clip-vit-l14")
    dataset = MyDataset(args.json_file, tokenizer, size=args.data_resolution)
    balance_sampler = ClassAwareSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=balance_sampler,
        num_workers=8,
        collate_fn=collate_fn
    )

    # Load SigLIP vision
    siglip_path = "weights/siglip"
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(siglip_path)
    siglip_model.eval().to(accelerator.device)
    print("Loaded SigLip vision model from", siglip_path)

    dit, vae, t5, clip = get_models("flux-dev", device=accelerator.device)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    dit.requires_grad_(False)
    
    dit = dit.to(accelerator.device).eval()

    model = EmoStyle(
        in_dim=8, img_dim=1152, token_nums=192, emb_dim=3072,
        codebook_size=60, query_dim=512,
        query_mode="transformer",
        transformer_heads=8,
        transformer_layers=4,
    ).to(accelerator.device)
    if args.checkpoint_path is not None:
        checkpoint_state = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_state, strict=False)
        print(f"Loaded model checkpoint from {args.checkpoint_path}")
    # Load pretrained codebook checkpoint
    if args.codebook_ckpt is not None:
        codebook_state = torch.load(args.codebook_ckpt, map_location="cpu")
        model.StyleDictionary.load_state_dict(codebook_state)
        if args.lock_codebook_after_init:
            model.lock_after_init()
            print(f"Codebook locked after loading checkpoint from {args.codebook_ckpt}")
    

    # Projector must also output 192 tokens to match mapper output
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
    style_proj_model.load_state_dict(remapped, strict=False)
    style_proj_model.eval()
    for p in style_proj_model.parameters():
        p.requires_grad_(False)

    codebook_params, transformer_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "StyleDictionary" in name:
            codebook_params.append(p)
        else:
            transformer_params.append(p)
    param_groups = []
    if transformer_params:
        param_groups.append({"params": transformer_params, "lr": args.lr_transformer})
    if codebook_params:
        param_groups.append({"params": codebook_params, "lr": args.lr_codebook})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    if args.optimize_target == "transformer":
        print("Optimizing only transformer/query parameters.")
        for p in codebook_params:
            p.requires_grad = False
    elif args.optimize_target == "codebook":
        print("Optimizing only codebook parameters.")
        for p in transformer_params:
            p.requires_grad = False

    if accelerator.is_main_process:
        print(f"[Opt] transformer params: {sum(p.numel() for p in transformer_params)} "
              f"codebook params: {sum(p.numel() for p in codebook_params)}")
        print(f"[Opt] lr_transformer={args.lr_transformer} lr_codebook={args.lr_codebook}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if args.init_mode == "similarity" and args.optimize_target in ("codebook", "both"):
        unwrapped = accelerator.unwrap_model(model)
        param_dtype = next(unwrapped.parameters()).dtype
        
        if accelerator.is_main_process:
            print(f"[Init] Scanning dataset for diverse images (threshold < {args.init_sim_threshold})...")
            
            # Create a temp loader for scanning
            scan_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.init_batch, shuffle=True, 
                num_workers=4, collate_fn=collate_fn
            )
            
            reference_style = None
            candidates_style = []
            candidates_content = []
            
            max_scan = args.init_max_samples
            scanned = 0
            
            unwrapped.eval()
            with torch.no_grad():
                for batch in scan_loader:
                    if scanned >= max_scan:
                        break
                    
                    imgs = batch["edited_images"]
                    if imgs is None: continue
                    
                    inputs = siglip_processor(images=imgs, return_tensors="pt").to(accelerator.device)
                    outputs = siglip_model(**inputs, output_hidden_states=True)
                    styles = style_proj_model(outputs).to(param_dtype)       # (B, T, D)
                    contents = outputs.last_hidden_state.to(param_dtype)     # (B, 729, 1152)
                    
                    bsz = styles.shape[0]
                    start_idx = 0
                    
                    if reference_style is None:
                        reference_style = styles[0].unsqueeze(0) # (1, T, D)
                        candidates_style.append(styles[0].unsqueeze(0).cpu())
                        candidates_content.append(contents[0].unsqueeze(0).cpu())
                        start_idx = 1
                        
                    if start_idx < bsz:
                        ref_vec = reference_style.mean(dim=1) # (1, D)
                        batch_vec = styles[start_idx:].mean(dim=1) # (B', D)
                        
                        sims = F.cosine_similarity(ref_vec, batch_vec, dim=-1) # (B',)
                        mask = sims < args.init_sim_threshold
                        
                        if mask.any():
                            sel_styles = styles[start_idx:][mask]
                            sel_contents = contents[start_idx:][mask]
                            candidates_style.append(sel_styles.cpu())
                            candidates_content.append(sel_contents.cpu())
                            
                    scanned += bsz
                    print(f"[Init] Scanned {scanned}/{max_scan} images...", end="\r")
            
            print("")
            
            if len(candidates_style) > 0:
                all_styles = torch.cat(candidates_style, dim=0).to(accelerator.device)
                all_contents = torch.cat(candidates_content, dim=0).to(accelerator.device)
                
                num_selected = all_styles.shape[0]
                print(f"[Init] Selected {num_selected} images with similarity < {args.init_sim_threshold}")
                
                emo_zeros = torch.zeros(num_selected, 8, device=accelerator.device, dtype=param_dtype)
                n = unwrapped.initialize_from_images(all_contents, all_styles, emo_vec=emo_zeros)
                print(f"[Init] Initialized {n} entries in codebook.")
                del all_styles, all_contents, emo_zeros
            else:
                print("[Init] No images found below similarity threshold.")

            unwrapped.train()
            state_cpu = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
        else:
            state_cpu = None

        # Broadcast
        accelerator.wait_for_everyone()
        if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            obj_list = [state_cpu] if accelerator.is_main_process else [None]
            dist.broadcast_object_list(obj_list, src=0)
            state_cpu = obj_list[0]
        accelerator.unwrap_model(model).load_state_dict(state_cpu)
        accelerator.wait_for_everyone()
        
        if args.lock_codebook_after_init:
            accelerator.unwrap_model(model).lock_after_init()
            if accelerator.is_main_process:
                print("[Init] Codebook locked.")


    total_steps = args.epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    if accelerator.is_main_process:
        print(f"Using CosineAnnealingLR: T_max={total_steps}, eta_min=1e-6")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total params: {total_params:.2f}M")


    global_step = 0
    param_dtype = next(model.parameters()).dtype
    dump_dir = Path(args.output_dir) / "tensor_dumps"

    timesteps = get_schedule(
        999,
        (args.data_resolution // 8) * (args.data_resolution // 8) // 4,
        shift=True,
    )
    timesteps = torch.tensor(timesteps, device=accelerator.device)

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    if accelerator.is_main_process:
        save_dir = os.path.join(args.save_path, f"checkpoint-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state(save_dir, safe_serialization=False)

    for epoch in range(args.epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(model):
                with torch.no_grad():
                    # Style target tokens (edited images)
                    siglip_inputs = siglip_processor(
                        images=batch["edited_images"], return_tensors="pt"
                    ).to(accelerator.device)
                    siglip_outputs = siglip_model(**siglip_inputs, output_hidden_states=True)
                    # Projector output: (B, style_token_nums, 3072) -> pool to single token
                    style_tokens = style_proj_model(siglip_outputs).to(param_dtype)
                    siglip_content_inputs = siglip_processor(
                        images=batch["origin_images"], return_tensors="pt"
                    ).to(accelerator.device)
                    siglip_content_outputs = siglip_model(**siglip_content_inputs, output_hidden_states=True)
                    content_tokens = siglip_content_outputs.last_hidden_state.to(param_dtype)  # (B,729,1152)

                emo_embeds = batch["tgt_onehots"].to(accelerator.device).to(param_dtype)

                pred_tokens, indice, align_loss = model(emo_embeds, content_tokens, style_tokens=style_tokens)  # (B,192,3072)
                assert pred_tokens.shape == style_tokens.shape, f"{pred_tokens.shape} vs {style_tokens.shape}"

                if args.optimize_target in ("transformer", "both"):
                    prompts = batch["txts"]
                    img = batch["transformed_edited_images"]
                    with torch.no_grad():
                        # Encode edited image to VAE latents; use same shape for noise/targets
                        x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                        ref_imgs = [vae.encode(
                            (TVF.to_tensor(ref_img) * 2.0 - 1.0)
                            .unsqueeze(0)
                            .to(accelerator.device, torch.float32)
                        ).to(torch.bfloat16) for ref_img in batch["origin_images"]]

                        bs = x_1.size(0)
                        t_idx = torch.randint(0, 1000, (bs,), device=accelerator.device)
                        t = timesteps[t_idx]
                        x_0 = torch.randn_like(x_1, device=accelerator.device)
                        x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                        guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)

                        # Tokenize x_t for the model input
                        inp_cond = prepare_multi_ip(
                            t5=t5,
                            clip=clip,
                            img=x_t,
                            prompt=prompts,
                            ref_imgs=ref_imgs,
                            pe="d"
                        )
                        # Align all inputs to the dit device/dtype
                        dit_dev = next(dit.parameters()).device
                        dit_dtype = next(dit.parameters()).dtype
                        def _to_dev_dtype(obj):
                            if torch.is_tensor(obj):
                                return obj.to(device=dit_dev, dtype=dit_dtype if obj.is_floating_point() else None)
                            if isinstance(obj, list):
                                return [_to_dev_dtype(o) for o in obj]
                            return obj
                        inp_cond = {k: _to_dev_dtype(v) for k, v in inp_cond.items()}
                        t = t.to(dit_dev)
                        guidance_vec = guidance_vec.to(device=dit_dev, dtype=dit_dtype)

                        inp_x1 = prepare_multi_ip(t5=t5, clip=clip, img=x_1, prompt=prompts, ref_imgs=ref_imgs, pe="d")
                        inp_x0 = prepare_multi_ip(t5=t5, clip=clip, img=x_0, prompt=prompts, ref_imgs=ref_imgs, pe="d")
                        inp_x1 = {k: _to_dev_dtype(v) for k, v in inp_x1.items()}
                        inp_x0 = {k: _to_dev_dtype(v) for k, v in inp_x0.items()}
                        target_tokens = (inp_x0['img'] - inp_x1['img']).to(dit_dtype)  # shape ~ (B, N_tokens, D_token)

                    model_pred = dit(
                        img=inp_cond['img'].to(weight_dtype),
                        img_ids=inp_cond['img_ids'].to(weight_dtype),
                        ref_img=[x.to(weight_dtype) for x in inp_cond['ref_img']],
                        ref_img_ids=[ref_img_id.to(weight_dtype) for ref_img_id in inp_cond['ref_img_ids']],
                        txt=inp_cond['txt'].to(weight_dtype),
                        txt_ids=inp_cond['txt_ids'].to(weight_dtype),
                        timesteps=t.to(weight_dtype),
                        y=inp_cond["vec"].to(weight_dtype),
                        guidance=guidance_vec.to(weight_dtype),
                        emostyle_inputs=style_tokens.to(dit_dev, dit_dtype),
                    )

                    assert model_pred.shape == target_tokens.shape, f"{model_pred.shape} vs {target_tokens.shape}"
                    diff_loss = F.mse_loss(model_pred.float(), target_tokens.float(), reduction="mean")

                # record indices for debugging
                if accelerator.is_main_process:
                    if indice.numel() == 1:
                        print(f"Step {global_step}: indice = {indice.item()}")
                    else:
                        print(f"Step {global_step}: indices = {indice.tolist()}")

                with torch.autocast(device_type=accelerator.device.type, enabled=False):
                    pred_f = pred_tokens.float()
                    tgt_f  = style_tokens.float()

                    if not args.no_norm_before_loss:
                        pred_use = F.layer_norm(pred_f, (pred_f.size(-1),))
                        tgt_use  = F.layer_norm(tgt_f, (tgt_f.size(-1),))
                    else:
                        pred_use, tgt_use = pred_f, tgt_f

                    mse_loss = None
                    cos_loss = None

                    if args.loss_type in ("mse", "combo"):
                        mse_loss = F.mse_loss(pred_use, tgt_use, reduction="mean")

                    if args.loss_type in ("cos", "combo"):
                        if args.cosine_mode == "flat":
                            p = F.normalize(pred_use.view(pred_use.size(0), -1), dim=-1)
                            t = F.normalize(tgt_use.view(tgt_use.size(0), -1), dim=-1)
                            cos_loss = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()
                        else:
                            p = F.normalize(pred_use, dim=-1)
                            t = F.normalize(tgt_use, dim=-1)
                            cos_loss = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()

                    if args.loss_type == "mse":
                        loss = mse_loss
                    elif args.loss_type == "cos":
                        loss = cos_loss
                    else:
                        loss = args.mse_weight * mse_loss + args.cos_weight * cos_loss

                if args.optimize_target == "both":
                    if align_loss is not None:
                        loss = loss + align_loss
                    if diff_loss is not None:
                        loss = loss + diff_loss
                elif args.optimize_target == "transformer":
                    if align_loss is not None:
                        loss = align_loss
                    if diff_loss is not None:
                        loss = loss + diff_loss
                elif args.optimize_target == "codebook":
                    loss = loss
                emo_score = batch["emo_scores"].to(accelerator.device).to(param_dtype)
                loss = loss * emo_score.mean()
                
                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step < 5 or global_step % 200 == 0:
                    if accelerator.is_main_process:
                        print(
                            f"[dbg] step {global_step} "
                            f"pred_mean {pred_f.mean():.3f} pred_std {pred_f.std():.3f} "
                            f"tgt_mean {tgt_f.mean():.3f} tgt_std {tgt_f.std():.3f} "
                            f"mse {mse_loss.item():.4f} cos {cos_loss.item():.4f}"
                        )

                if mse_loss is not None:
                    writer.add_scalar('Loss/mse', float(mse_loss.item()), global_step)
                if cos_loss is not None:
                    writer.add_scalar('Loss/cos', float(cos_loss.item()), global_step)
                if align_loss is not None:
                    writer.add_scalar('Loss/align', float(align_loss.item()), global_step)
                if diff_loss is not None:
                    writer.add_scalar('Loss/diff', float(diff_loss.item()), global_step)

                gathered = accelerator.gather(loss.detach().unsqueeze(0))
                avg_loss = gathered.mean().item()
                writer.add_scalar('Loss/seq', avg_loss, global_step)

                if accelerator.is_main_process:
                    print(f"Epoch {epoch} step {step} gstep {global_step} "
                          f"loss {avg_loss:.5f} data_t {load_data_time:.3f}")

                global_step += 1
                if accelerator.is_main_process and (global_step % args.save_steps == 0):
                    save_dir = os.path.join(args.save_path, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    accelerator.save_state(save_dir, safe_serialization=False)

            begin = time.perf_counter()

    writer.close()

if __name__ == "__main__":
    main()