import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16.):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(module, rank, alpha, targets):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += inject_lora(child, rank, alpha, targets)
    return count


class CLIPClassifier(nn.Module):
    def __init__(self, model_id, revision, class_names, device, rank, alpha,
                 prompt, checkpointing=False, local_files_only=False):
        super().__init__()
        from transformers import CLIPModel, AutoTokenizer
        clip = CLIPModel.from_pretrained(model_id, revision=revision,local_files_only=local_files_only)
        clip.requires_grad_(False)
        clip.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,local_files_only=local_files_only)
        # Text features are computed once on CPU; no text model is retained on GPU.
        texts = [prompt.format(c.replace('_', ' ')) for c in class_names]
        with torch.no_grad():
            vectors = []
            for i in range(0, len(texts), 32):
                tokens = tokenizer(texts[i:i+32], padding=True, return_tensors="pt")
                vectors.append(clip.get_text_features(**tokens).float())
            text = F.normalize(torch.cat(vectors), dim=-1)
        self.vision = clip.vision_model
        self.projection = clip.visual_projection
        self.register_buffer("text_features", text)
        self.register_buffer("logit_scale", clip.logit_scale.detach().exp().float())
        self.revision = getattr(clip.config, "_commit_hash", None) or revision
        targets = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        self.adapter_count = inject_lora(self.vision, rank, alpha, targets)
        if checkpointing:
            # Non-reentrant checkpointing supports autograd.grad and frozen inputs.
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            self.vision.encoder.gradient_checkpointing = True
            self.vision.encoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self.to(device)

    def forward(self, x):
        features = self.vision(pixel_values=x).pooler_output
        features = F.normalize(self.projection(features).float(), dim=-1)
        return self.logit_scale * features @ self.text_features.float().T


class TinyClassifier(nn.Module):
    """OFFLINE INTEGRATION FIXTURE ONLY; not a CLIP accuracy experiment."""
    def __init__(self, classes, rank=4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8,32), nn.Tanh(),
                                 nn.Linear(32,24), nn.Tanh())
        self.head = nn.Linear(24, classes, bias=False)
        self.requires_grad_(False)
        for i in (1,3):
            self.net[i] = LoRALinear(self.net[i], rank, rank)
        self.revision = "synthetic-fixture"
        self.adapter_count = 2

    def forward(self, x):
        return self.head(self.net(x))


def trainable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore_trainable(model, state):
    params = dict(model.named_parameters())
    expected = {n for n,p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint adapter keys do not match the configured model")
    for n,t in state.items():
        params[n].copy_(t.to(params[n].device))
