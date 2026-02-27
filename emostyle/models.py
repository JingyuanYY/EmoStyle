import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EmoStyle(nn.Module):
    def __init__(self, 
                 in_dim: int = 8,
                 img_dim: int = 1152,
                 token_nums: int = 192,
                 emb_dim: int = 3072,
                 codebook_size: int = 512,
                 query_dim: int = 512, # Kept for signature compatibility, but effectively ignored/overridden
                 tau: float = 1.0,
                 hard: bool = True,
                 init_threshold: float = 0.8,
                 free_select: str = "sequential",
                 ema_momentum: float = 0.0,
                 style_sim_reduce: str = "mean",
                 query_mode: str = "concat",
                 transformer_heads: int = 8,
                 transformer_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.img_dim = img_dim
        self.token_nums = token_nums
        self.emb_dim = emb_dim
        # We now force query_dim to match emb_dim because we compare against value_table directly
        self.query_dim = emb_dim 
        self.codebook_size = codebook_size
        self.tau = tau
        self.hard = hard
        self.init_threshold = init_threshold
        self.free_select = free_select
        self.ema_m = ema_momentum
        self.style_sim_reduce = style_sim_reduce
        self.query_mode = query_mode
        self.transformer_heads = transformer_heads
        assert self.free_select in ("sequential", "random")
        assert self.query_mode in ("concat", "cross_attn", "transformer")

        # === Query builder modules (depend on query_mode) ===
        # Output dimension is now self.emb_dim (3072) instead of old query_dim (512)
        if self.query_mode == "concat":
            # Split emotion/image projections and concatenate (requires even split)
            assert self.emb_dim % 2 == 0, "emb_dim must be even for concat mode"
            self.emo_q = nn.Linear(in_dim, self.emb_dim // 2)
            self.img_pool = nn.Linear(img_dim, self.emb_dim // 2)
        elif self.query_mode == "cross_attn":
            # Emotion becomes a single query; image tokens supply keys/values
            self.emo_q = nn.Linear(in_dim, self.emb_dim)
            self.img_k = nn.Linear(img_dim, self.emb_dim)
            self.img_v = nn.Linear(img_dim, self.emb_dim)
            self.cross_ln = nn.LayerNorm(self.emb_dim)
        else:  # transformer
            self.emo_token = nn.Linear(in_dim, self.emb_dim)
            self.img_proj = nn.Linear(img_dim, self.emb_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.emb_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.tr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.tr_ln = nn.LayerNorm(self.emb_dim)

        # (legacy names used later conditionally; keep attributes for existing checkpoints)
        if not hasattr(self, "emo_q") and self.query_mode != "transformer":
            self.emo_q = nn.Linear(in_dim, self.emb_dim)
        if not hasattr(self, "img_pool") and self.query_mode == "concat":
            self.img_pool = nn.Linear(img_dim, self.emb_dim // 2)

        # Trainable codebook values ONLY (no separate tags)
        self.StyleDictionary = nn.Parameter(torch.zeros(codebook_size, token_nums, emb_dim))
        
        # Init flags
        self.register_buffer("value_init", torch.zeros(codebook_size, dtype=torch.bool), persistent=True)

        self.lock_codebook = False

        # Alignment config (no longer applicable since tags ARE values, but kept for API compat)
        self.align_tags_with_style = True 
        self.align_loss_weight = 1.0
        self._align_loss: torch.Tensor | None = None

    # === NEW: helpers for pre-initialization from tensors/images ===
    @torch.no_grad()
    def reset_codebook(self):
        """
        Zero all entries and clear init flags.
        """
        self.StyleDictionary.data.zero_()
        self.value_init[:] = False

    @torch.no_grad()
    def visualize_codebook_values(self) -> torch.Tensor:
        """
        Return codebook style tokens as a (codebook_size, token_nums, emb_dim) tensor.
        """
        return self.StyleDictionary.data.clone()

    @torch.no_grad()
    def initialize_from_images(
        self,
        image_tokens: torch.Tensor,               # (K, N, img_dim)
        style_tokens: torch.Tensor,               # (K, token_nums, emb_dim)
        emo_vec: torch.Tensor | None = None,      # optional (K, in_dim); if None -> zeros
        start_from: int = 0,                      # first code index to use
        reset: bool = False,                      # if True, clear existing entries first
    ) -> int:
        """
        Initialize entries using a batch/list of images and their style tokens.
        Returns: number of entries initialized.
        """
        device = self.StyleDictionary.device
        dtype_v = self.StyleDictionary.dtype

        if reset:
            self.reset_codebook()

        K = image_tokens.size(0)
        
        # Free slots
        free = torch.nonzero(~self.value_init, as_tuple=False).squeeze(1)
        # obey start_from offset
        free = free[free >= start_from]
        take = min(K, free.numel())
        if take <= 0:
            return 0

        # Move tensors to storage device/dtype
        v_store = style_tokens[:take].to(device=device, dtype=dtype_v)

        # Assign sequentially
        slots = free[:take]
        self.StyleDictionary.data.index_copy_(0, slots, v_store)
        self.value_init[slots] = True

        return int(take)

    def build_query(self, emo_vec: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        emo_vec: (B, in_dim)
        image_tokens: (B, N, img_dim)
        return: normalized query (B, emb_dim)
        """
        if self.query_mode == "concat":
            img_pooled = image_tokens.mean(dim=1)  # (B, img_dim)
            q = torch.cat([self.emo_q(emo_vec), self.img_pool(img_pooled)], dim=-1)
            q = F.layer_norm(q, (q.size(-1),))
        elif self.query_mode == "cross_attn":
            # Single-step attention: emotion as query, image tokens as context
            q0 = self.emo_q(emo_vec)  # (B,D)
            k = self.img_k(image_tokens)  # (B,N,D)
            v = self.img_v(image_tokens)  # (B,N,D)
            # scaled dot product: (B,1,N)
            attn_scores = torch.matmul(q0.unsqueeze(1), k.transpose(1, 2)) * (q0.size(-1) ** -0.5)
            attn_probs = attn_scores.softmax(dim=-1)
            ctx = torch.matmul(attn_probs, v).squeeze(1)  # (B,D)
            q = self.cross_ln(q0 + ctx)
        else:  # transformer
            emo_tok = self.emo_token(emo_vec).unsqueeze(1)  # (B,1,D)
            img_tok = self.img_proj(image_tokens)          # (B,N,D)
            seq = torch.cat([emo_tok, img_tok], dim=1)      # (B,1+N,D)
            enc = self.tr_encoder(seq)                     # (B,1+N,D)
            q = self.tr_ln(enc[:, 0])                      # take emotion (CLS-like) token

        # L2 normalize
        q = q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return q

    @staticmethod
    def _st_hard_softmax(logits: torch.Tensor, dim: int, hard: bool) -> torch.Tensor:
        # soft weights
        w_soft = torch.softmax(logits, dim=dim)
        if not hard:
            return w_soft
        # hard one-hot with straight-through estimator
        idx = w_soft.argmax(dim=dim, keepdim=True)
        w_hard = torch.zeros_like(w_soft).scatter_(dim, idx, 1.0)
        return (w_hard - w_soft).detach() + w_soft

    def lock_after_init(self):
        """
        Freeze codebook structure: no new entries will be initialized
        and selection will be restricted to already initialized indices.
        """
        self.lock_codebook = True

    def pop_align_loss(self) -> torch.Tensor:
        """
        Return the last stored alignment loss (or 0.) and clear it.
        Call this in your training step and add it to your main loss.
        """
        if self._align_loss is None:
            return torch.tensor(0.0, device=self.StyleDictionary.device, dtype=self.StyleDictionary.dtype)
        v = self._align_loss
        self._align_loss = None
        return v
    
    @torch.no_grad()
    def get_codebook_entries(self, initialized_only: bool = True, as_numpy: bool = False):
        """
        Return code indices and their stored values.
        Args:
            initialized_only: if True, return only entries where value is initialized.
            as_numpy: if True, return CPU numpy arrays; otherwise return torch tensors on current device.
        Returns:
            indices: (K,)
            values:  (K, token_nums, emb_dim)
        """
        if initialized_only:
            mask = self.value_init
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        else:
            idx = torch.arange(self.codebook_size, device=self.StyleDictionary.device)

        if idx.numel() == 0:
            # Create empty tensors with correct shapes
            empty_vals = self.StyleDictionary.new_empty((0, self.token_nums, self.emb_dim))
            if as_numpy:
                return (
                    idx.detach().cpu().numpy(),
                    empty_vals.detach().cpu().numpy(),
                )
            return idx, empty_vals

        values = self.StyleDictionary.index_select(0, idx).detach().clone()

        if as_numpy:
            return (
                idx.detach().cpu().numpy(),
                values.detach().cpu().numpy(),
            )
        return idx, values

    def forward(
        self,
        emo_vec: torch.Tensor,
        image_tokens: torch.Tensor,
        style_tokens: torch.Tensor | None = None,
        return_indices: bool = True,
    ):
        B = emo_vec.size(0)
        dtype = next(self.parameters()).dtype

        # Build query (B, emb_dim)
        q = self.build_query(emo_vec.to(dtype), image_tokens.to(dtype))

        # Prepare codebook representations for retrieval
        # We use mean-pooled value table as the "keys"
        code_repr = self.StyleDictionary.mean(dim=1) # (C, D)
        code_norm = code_repr / (code_repr.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        # Query logits: q vs code_norm
        logits_query = (q @ code_norm.t()) / max(1e-6, self.tau)  # (B,C)

        # Mask uninitialized codes
        mask = (~self.value_init).unsqueeze(0)

        logits = logits_query

        logits = logits.masked_fill(mask, -1e9)

        weights = self._st_hard_softmax(logits, dim=1, hard=True)
        out = torch.einsum("bc,ctd->btd", weights, self.StyleDictionary)
        argmax_idx = weights.argmax(dim=1)

        # Optional EMA update of selected code (always uses style_tokens if provided)
        if self.ema_m > 0.0 and style_tokens is not None:
            m = self.ema_m
            with torch.no_grad():
                for b, code in enumerate(argmax_idx.tolist()):
                    # Update value towards current style tokens
                    self.StyleDictionary.data[code].lerp_(style_tokens[b].to(self.StyleDictionary.dtype), 1.0 - m)
                    self.value_init[code] = True

        if return_indices:
            return out, argmax_idx, self._align_loss
        return out