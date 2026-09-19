"""Bounded, class-balanced memory with IMMUTABLE per-example loss anchors.

Memory losses always use the fixed 100-class frozen CLIP head. This prevents
an expanding CIL softmax from silently moving the loss reference. All class
names are assumed public; no future-class training images are accessed.
Evaluation and new-task training still use only classes seen so far.
"""
import numpy as np
import torch
from torch.nn import functional as F


class Memory:
    def __init__(self, capacity, seed):
        self.capacity,self.seed = capacity,seed
        self.entries = []
        self.rng = np.random.default_rng(seed+909)

    def __len__(self):
        return len(self.entries)

    @property
    def anchor(self):
        return float(np.mean([x["anchor"] for x in self.entries])) if self.entries else 0.

    def indices(self):
        return [x["index"] for x in self.entries]

    def sample(self, n):
        if not self.entries or n <= 0:
            return []
        return self.rng.choice(self.indices(),size=n,replace=n>len(self.entries)).tolist()

    @torch.no_grad()
    def update(self, model, data, task, device, batch_size):
        if self.capacity <= 0:
            return
        seen = sum(data.task_classes[:task+1],[])
        if self.capacity < len(seen):
            raise ValueError("memory_size must be at least the number of seen classes")
        old = {e["index"]:e for e in self.entries}
        chosen = []
        for position,c in enumerate(seen):
            quota = self.capacity//len(seen) + int(position < self.capacity%len(seen))
            if c in data.task_classes[task]:
                candidates = data.train_indices[data.targets[data.train_indices] == c]
                rng = np.random.default_rng(self.seed+100003*c)
                ids = rng.permutation(candidates)[:quota].tolist()
            else:
                # Old data are selected ONLY from retained memory, never re-read
                # from the full previous-task training pool.
                ids = [e["index"] for e in self.entries if e["class"] == c][:quota]
            chosen.extend(ids)
        new_ids = [i for i in chosen if i not in old]
        training = model.training
        model.eval()
        for start in range(0,len(new_ids),batch_size):
            ids = new_ids[start:start+batch_size]
            x,y = data.batch(ids,device)
            losses = F.cross_entropy(model(x).float(),y,reduction="none").cpu().tolist()
            for i,loss in zip(ids,losses):
                old[i] = {"index":int(i),"class":int(data.targets[i]),"anchor":float(loss),"added_task":task}
        model.train(training)
        self.entries = [old[i] for i in chosen]

    def state_dict(self):
        return {"entries":self.entries,"rng":self.rng.bit_generator.state}

    def load_state_dict(self,state):
        self.entries = state["entries"]
        self.rng.bit_generator.state = state["rng"]


def memory_loss(model, data, indices, device, batch_size, gradients=False, parameters=None, head_classes=None):
    """Deterministic transforms, sample-weighted mean, no overwrite of new .grad."""
    if not indices:
        return (0., {}) if gradients else 0.
    training = model.training
    model.eval()
    total = 0.
    grads = {n:torch.zeros_like(p,dtype=torch.float32) for n,p in (parameters or [])}
    try:
        with torch.set_grad_enabled(gradients):
            for start in range(0,len(indices),batch_size):
                ids = indices[start:start+batch_size]
                x,y = data.batch(ids,device)
                logits = model(x).float()
                if head_classes is not None:
                    classes = torch.as_tensor(head_classes,device=device)
                    mapping = torch.full((logits.shape[-1],),-1,device=device,dtype=torch.long)
                    mapping[classes] = torch.arange(len(classes),device=device)
                    logits,y = logits[:,classes],mapping[y]
                loss = F.cross_entropy(logits,y,reduction="sum") / len(indices)
                total += float(loss.detach())
                if gradients:
                    gg = torch.autograd.grad(loss,[p for n,p in parameters],allow_unused=True)
                    for (n,p),g in zip(parameters,gg):
                        if g is not None:
                            grads[n].add_(g.detach().float())
    finally:
        model.train(training)
    return (total,grads) if gradients else total


def fisher_scores(model, data, indices, device, proposal, label_generator):
    """ONE image and ONE model-sampled label per score. Joint layer coordinates."""
    training = model.training
    model.eval()
    scores = []
    try:
        for i in indices:
            x,_ = data.batch([i],device)
            logits = model(x).float()
            target = torch.multinomial(logits.detach().softmax(-1),1,generator=label_generator).squeeze(-1)
            loss = F.cross_entropy(logits,target)
            gg = torch.autograd.grad(loss,[p for n,p in proposal.params],allow_unused=True)
            scores.append(proposal.coordinates({n:g for (n,p),g in zip(proposal.params,gg)}))
    finally:
        model.train(training)
    return np.asarray(scores)
