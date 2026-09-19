import numpy as np
import torch
from torch.utils.data import Dataset


class DataBundle:
    def __init__(self, cfg):
        self.synthetic = cfg.get("synthetic", False)
        if self.synthetic:
            self.classes = [f"class_{i}" for i in range(cfg.get("synthetic_classes",6))]
            rng = np.random.default_rng(12031)
            centroids = rng.normal(size=(len(self.classes),3,8,8)).astype(np.float32)
            def make(count):
                y = np.repeat(np.arange(len(self.classes)),count)
                x = centroids[y] + .15*rng.normal(size=(len(y),3,8,8)).astype(np.float32)
                return x.astype(np.float32), y
            self.train_images,self.targets = make(12)
            self.test_images,self.test_targets = make(6)
            self.train_transform = self.eval_transform = None
            holdout = 2
        else:
            from torchvision.datasets import CIFAR100
            from torchvision import transforms as T
            train = CIFAR100(cfg["data_root"], train=True, download=cfg.get("download",False))
            test = CIFAR100(cfg["data_root"], train=False, download=cfg.get("download",False))
            self.classes = train.classes
            self.train_images,self.targets = train.data,np.asarray(train.targets)
            self.test_images,self.test_targets = test.data,np.asarray(test.targets)
            norm = T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))
            self.eval_transform = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                                            T.CenterCrop(224), T.ToTensor(), norm])
            if cfg.get("augmentation", "random_crop") == "random_crop":
                self.train_transform = T.Compose([T.RandomResizedCrop(224, scale=(.8,1.),
                    interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(), T.ToTensor(), norm])
            else:
                self.train_transform = self.eval_transform
            holdout = cfg.get("validation_per_class",50)
        if holdout <= 0:
            raise ValueError("Keep a disjoint validation split; v0 requires validation_per_class > 0")
        split_rng = np.random.default_rng(cfg.get("split_seed",17))
        self.train_indices, self.val_indices = [], []
        for c in range(len(self.classes)):
            ids = np.flatnonzero(self.targets == c)
            split_rng.shuffle(ids)
            if holdout >= len(ids):
                raise ValueError("Validation allocation consumes an entire class")
            self.val_indices.extend(ids[:holdout].tolist())
            self.train_indices.extend(ids[holdout:].tolist())
        self.train_indices = np.asarray(self.train_indices)
        self.val_indices = np.asarray(self.val_indices)
        # Explicit real-data integration subset; never silently call it a benchmark.
        self.debug_subset = False
        for key,attribute in (("debug_train_per_class","train_indices"),
                              ("debug_validation_per_class","val_indices")):
            count = cfg.get(key)
            if count is not None:
                if not isinstance(count,int) or count <= 0:
                    raise ValueError(f"{key} must be a positive integer")
                pool = getattr(self,attribute)
                setattr(self,attribute,np.concatenate([
                    pool[self.targets[pool] == c][:count] for c in range(len(self.classes))]))
                self.debug_subset = True
        order = cfg.get("class_order")
        if order is None:
            order = np.random.default_rng(cfg.get("class_order_seed",1993)).permutation(len(self.classes)).tolist()
        if sorted(order) != list(range(len(self.classes))):
            raise ValueError("class_order must be a permutation of every class")
        tasks = cfg.get("tasks",10)
        if len(self.classes) % tasks:
            raise ValueError("Number of classes must be divisible by tasks")
        self.order = order
        self.task_classes = [v.tolist() for v in np.array_split(order,tasks)]

    def image(self, index, test=False, augment=False):
        array = (self.test_images if test else self.train_images)[index]
        if self.synthetic:
            return torch.from_numpy(array.copy())
        from PIL import Image
        transform = self.train_transform if augment else self.eval_transform
        return transform(Image.fromarray(array))

    def batch(self, indices, device, test=False, augment=False):
        x = torch.stack([self.image(int(i),test,augment) for i in indices]).to(device)
        target = self.test_targets if test else self.targets
        y = torch.as_tensor(target[np.asarray(indices,dtype=int)],dtype=torch.long,device=device)
        return x,y

    def pool(self, task):
        return self.train_indices[np.isin(self.targets[self.train_indices],self.task_classes[task])]


class IndexedImages(Dataset):
    def __init__(self, bundle, indices, augment=True):
        self.bundle, self.indices, self.augment = bundle, list(indices), augment
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        i = int(self.indices[index])
        return self.bundle.image(i,augment=self.augment), int(self.bundle.targets[i])
