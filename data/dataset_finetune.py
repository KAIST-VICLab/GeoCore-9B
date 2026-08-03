import math
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image


def XYZToLonLat(x, y, z):
    x, y, z = float(x), float(y), float(z)
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


class Git10MDataset(Dataset):
    def __init__(self, root, target_size=256, score_threshold=4.8):
        self.root = root
        self.dataset = load_from_disk(root)
        self.target_size = target_size

        print(f"Filtering dataset with img_quality_score >= {score_threshold}...")
        scores = self.dataset["img_quality_score"]
        self.valid_indices = [i for i, score in enumerate(scores) if score is not None and score >= score_threshold]
        print(f"Filtered dataset size: {len(self.valid_indices):,} / {len(self.dataset):,}")

    def __len__(self):
        return len(self.valid_indices)

    def process_image(self, img):
        w, h = img.size
        min_side = min(w, h)
        scale_factor = 1.0

        if min_side < self.target_size:
            scale_factor = self.target_size / min_side
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))
            img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BICUBIC)

        i, j, h_crop, w_crop = T.RandomCrop.get_params(img, output_size=(self.target_size, self.target_size))
        img = TF.crop(img, i, j, h_crop, w_crop)

        img = TF.to_tensor(img)
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=-1.0)
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return img, scale_factor

    def __getitem__(self, index):
        # map filtered index back to the underlying dataset
        actual_index = self.valid_indices[index]
        item = self.dataset[actual_index]
        
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image, scale_factor = self.process_image(image)

        google_loc = item.get('Google_location', None)
        try:
            z_level, x_tile, y_tile = google_loc.split('_')
            lon, lat = XYZToLonLat(x_tile, y_tile, z_level)
            calc_res = (17 - int(z_level))

            if scale_factor > 1.0:
                calc_res -= math.log2(scale_factor)

        except Exception:
            lon, lat, calc_res = -999.0, -999.0, -999.0

        meta = {
            "res": torch.tensor(calc_res, dtype=torch.float32),
            "lon": torch.tensor(lon, dtype=torch.float32),
            "lat": torch.tensor(lat, dtype=torch.float32),
            "score": torch.tensor(item.get("img_quality_score", 0.0) or 0.0, dtype=torch.float32),
            "loc_raw": google_loc if google_loc is not None else ""
        }

        caption = str(item.get("caption", "") or "")

        return image, caption, meta


class CFGDataset(Dataset):
    def __init__(self, dataset, p_uncond=0.1, p_drop_text=0.1, p_drop_res=0.1, p_drop_loc=0.1, empty_token=""):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.p_drop_text = p_drop_text
        self.p_drop_res = p_drop_res
        self.p_drop_loc = p_drop_loc
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, cap, meta = self.dataset[item]

        drop_text = False
        drop_res = False
        drop_loc = False

        if random.random() < self.p_uncond:
            drop_text = drop_res = drop_loc = True
        else:
            if random.random() < self.p_drop_text:
                drop_text = True
            if random.random() < self.p_drop_res:
                drop_res = True
            if random.random() < self.p_drop_loc:
                drop_loc = True

        if drop_text:
            cap = self.empty_token

        if drop_res:
            meta["res"] = torch.tensor(-999.0, dtype=torch.float32)

        if drop_loc:
            meta["lon"] = torch.tensor(-999.0, dtype=torch.float32)
            meta["lat"] = torch.tensor(-999.0, dtype=torch.float32)
            meta["loc_raw"] = ""

        return x, cap, meta


class Git10M_T2I:
    def __init__(self, path, cfg=True, p_uncond=0.1, score_threshold=4.8):
        print(f'Prepare Git-10M dataset from {path}...')

        self.train = Git10MDataset(path, target_size=256, score_threshold=score_threshold)

        if cfg:
            print(f'Setting up CFG for Git-10M with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond=p_uncond, empty_token="")

    @property
    def data_shape(self):
        return 32, 16, 16