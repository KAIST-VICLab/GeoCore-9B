"""Discriminative linear probes on frozen GeoCore-9B DiT features (pre-registered).

Method (SatDiFuser/DIFT-style): z0 = E_VAE(x); z_tau = (1-tau) z0 + tau eps (linear path,
tau=0.25, per-image seed 3000+idx); one forward of the frozen DiT with empty text and null
geospatial metadata; image-token features (16x16x4096) hooked at global blocks {4, 8, 16, 24}
(block 8 = the GSA-aligned block; 16/24 are single-stream blocks, image-token slice).

Tasks / probes (all linear, features frozen):
  loveda  : semantic segmentation, 7 classes (labels 1..7, 0=ignore). Per-pixel linear probe
            on bilinearly x4-upsampled features (64x64 grid), labels nearest-downsampled to
            64x64. Report mIoU at the 64x64 probe grid.
  eurosat : scene classification, 10 classes. Token-mean-pooled features -> linear classifier,
            fixed 80/20 split (numpy seed 0). Report top-1.
  bright  : building-damage change detection. Siamese frozen features of pre/post;
            per-pixel linear probe on [f_pre, f_post, |f_pre-f_post|]; labels
            nearest-downsampled to 64x64. Report mIoU over present classes.
Usage:
  python3 discriminative_probe.py --task loveda  --gpu 1
  python3 discriminative_probe.py --task eurosat --gpu 7
  python3 discriminative_probe.py --task bright  --gpu 1
"""
import argparse, glob, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.flux2 import Flux2, TerraNova9BParams
from models.vae_flux2 import AutoEncoder, AutoEncoderParams
from models.text_encoder import TextEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
TAU, BS, GRID = 0.25, 16, 16
TAU_SWEEP = [0.05, 0.1, 0.25, 0.5, 0.75]   # EuroSAT-subset ablation only
BLOCKS = {"g4": ("double", 3), "g8": ("double", 7), "g16": ("single", 7), "g24": ("single", 15)}


class FeatExtractor:
    def __init__(self, dev, ckpt, vae_dir):
        self.dev = dev
        self.model = Flux2(TerraNova9BParams()).to(dev, torch.bfloat16)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["ema"] if "ema" in ck else ck.get("model", ck)
        self.model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=True)
        self.model.eval()
        del ck, sd
        self.vae = AutoEncoder(AutoEncoderParams()).to(dev, torch.bfloat16)
        self.vae.load_state_dict(load_file(vae_dir))
        self.vae.eval()
        self.te = TextEncoder(device=dev, dtype=torch.bfloat16)
        self.te.encoder_clip.eval(); self.te.encoder_t5.eval()
        self.cap = {}
        # Flux2.forward calls block.forward_kv_extract directly (not __call__), so nn forward
        # hooks never fire; wrap the method instead.
        for name, (kind, idx) in BLOCKS.items():
            blk = (self.model.double_blocks if kind == "double" else self.model.single_blocks)[idx]
            self._wrap(blk, name, kind)
        ids = torch.zeros(GRID, GRID, 3)
        ids[..., 1] += torch.arange(GRID)[:, None]
        ids[..., 2] += torch.arange(GRID)[None, :]
        self.img_ids = ids.reshape(GRID * GRID, 3).to(dev, torch.bfloat16)
        self.n_txt = None

    def _wrap(self, blk, name, kind):
        orig = blk.forward_kv_extract
        def wrapped(*a, **k):
            out = orig(*a, **k)
            t = out[0]  # double: (img, txt, kv); single: (x, kv)
            if kind == "single":
                t = t[:, self.n_txt:, :]
            self.cap[name] = t.detach()
            return out
        blk.forward_kv_extract = wrapped

    @torch.no_grad()
    def __call__(self, imgs, seed0, tau=TAU):
        """imgs: [B,3,256,256] in [-1,1] float. Returns dict name->[B,256,4096] fp16 cpu."""
        n = imgs.shape[0]
        x = imgs.to(self.dev, torch.bfloat16)
        z0 = self.vae.encode(x)
        eps = torch.stack([torch.randn(z0.shape[1:], generator=torch.Generator().manual_seed(seed0 + j))
                           for j in range(n)]).to(self.dev, torch.bfloat16)
        zt = (1 - tau) * z0 + tau * eps
        ctx, pooled, ctx_ids = self.te([""] * n)
        self.n_txt = ctx.shape[1]
        t = torch.full((n,), tau, device=self.dev, dtype=torch.bfloat16)
        null = torch.full((n,), -999.0, device=self.dev, dtype=torch.bfloat16)
        self.cap.clear()
        self.model(x=zt, x_ids=self.img_ids.unsqueeze(0).repeat(n, 1, 1), timesteps=t,
                   ctx=ctx, ctx_ids=ctx_ids, y=pooled, res=null, lon=null, lat=null)
        return {k: v.float().half().cpu() for k, v in self.cap.items()}


def to_input(pil):
    pil = pil.convert("RGB").resize((256, 256), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return x * 2 - 1


def mask64(pil):
    return torch.from_numpy(np.asarray(pil.resize((64, 64), Image.NEAREST)).astype(np.int64))


def extract_set(fx, items, feat_path, seed0):
    """items: list of dicts with load() -> ([-1,1] tensor) images list. Saves [N,4,256,4096]."""
    if os.path.exists(feat_path):
        return np.load(feat_path, mmap_mode="r")
    n_img = len(items[0]["imgs"])
    arr = np.lib.format.open_memmap(feat_path + ".tmp", mode="w+", dtype=np.float16,
                                    shape=(len(items), n_img, 4, GRID * GRID, 4096))
    for i in range(0, len(items), BS):
        chunk = items[i:i + BS]
        for v in range(n_img):
            batch = torch.stack([c["imgs"][v] for c in chunk])
            f = fx(batch, seed0 + i)
            for k, name in enumerate(["g4", "g8", "g16", "g24"]):
                arr[i:i + len(chunk), v, k] = f[name].numpy()
        if (i // BS) % 20 == 0:
            print(f"extract {i}/{len(items)}", flush=True)
    arr.flush()
    os.replace(feat_path + ".tmp", feat_path)
    return np.load(feat_path, mmap_mode="r")


def train_pixel_probe(Xtr, Ytr, Xva, Yva, ncls, dev, ignore=0, epochs=8, dim=4 * 4096):
    """X*: memmap [N, 4, 256, 4096] (or [N,2imgs,...] pre-flattened by caller into dim);
       Y*: [N,64,64] int labels. Streams batches; features upsampled 16->64 bilinear."""
    W = torch.nn.Linear(dim, ncls).to(dev)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    N = Xtr.shape[0]
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(N)
        for bi in range(0, N, 8):
            idx = order[bi:bi + 8]
            feats = torch.from_numpy(np.array(Xtr[idx])).to(dev).float()   # [b,...]
            f = feats.reshape(len(idx), -1, GRID, GRID, 4096)
            f = f.permute(0, 1, 4, 2, 3).reshape(len(idx), -1, GRID, GRID)
            f = F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False)
            f = f.permute(0, 2, 3, 1).reshape(-1, dim)
            y = torch.from_numpy(np.array(Ytr[idx])).to(dev).reshape(-1)
            m = y != ignore
            if m.sum() == 0:
                continue
            sel = torch.randperm(int(m.sum()), device=dev)[:16384]
            loss = F.cross_entropy(W(f[m][sel]), (y[m] - (1 if ignore == 0 else 0))[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"probe ep{ep} loss {loss.item():.3f}", flush=True)
    inter = torch.zeros(ncls); union = torch.zeros(ncls)
    with torch.no_grad():
        for bi in range(0, Xva.shape[0], 8):
            feats = torch.from_numpy(np.array(Xva[bi:bi + 8])).to(dev).float()
            b = feats.shape[0]
            f = feats.reshape(b, -1, GRID, GRID, 4096).permute(0, 1, 4, 2, 3).reshape(b, -1, GRID, GRID)
            f = F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False)
            pred = W(f.permute(0, 2, 3, 1).reshape(-1, dim)).argmax(1)
            y = torch.from_numpy(np.array(Yva[bi:bi + 8])).to(dev).reshape(-1)
            m = y != ignore
            yv = (y[m] - (1 if ignore == 0 else 0)); pv = pred[m]
            for c in range(ncls):
                inter[c] += ((pv == c) & (yv == c)).sum().item()
                union[c] += (((pv == c) | (yv == c)).sum().item())
    present = union > 0
    iou = (inter[present] / union[present].clamp(min=1)).numpy()
    return {"miou": float(iou.mean()), "per_class_iou": iou.round(4).tolist(),
            "n_classes_present": int(present.sum())}


def run_loveda(fx, dev, data_dir, OUT):
    from datasets import load_dataset
    dd = os.path.join(data_dir, "LoveDA", "data")
    ds = load_dataset("parquet", data_files={
        "train": sorted(glob.glob(f"{dd}/train-*.parquet")),
        "validation": sorted(glob.glob(f"{dd}/validation-*.parquet"))})
    cols = ds["train"].column_names
    ikey = "image" if "image" in cols else cols[0]
    mkey = "mask" if "mask" in cols else ("label" if "label" in cols else cols[1])
    print("loveda cols:", cols, "->", ikey, mkey, flush=True)

    def prep(split, cap):
        items, ys = [], []
        n = min(len(ds[split]), cap)
        idxs = np.linspace(0, len(ds[split]) - 1, n, dtype=int)
        for i in idxs:
            r = ds[split][int(i)]
            items.append({"imgs": [to_input(r[ikey])]})
            ys.append(mask64(r[mkey]))
        return items, torch.stack(ys).numpy()

    tr_items, Ytr = prep("train", 3000)
    va_items, Yva = prep("validation", 1200)
    Xtr = extract_set(fx, tr_items, f"{OUT}/loveda_train.npy", 3000)
    Xva = extract_set(fx, va_items, f"{OUT}/loveda_val.npy", 90000)
    res = train_pixel_probe(Xtr, Ytr, Xva, Yva, ncls=7, dev=dev, ignore=0)
    json.dump(res, open(f"{OUT}/loveda_results.json", "w"), indent=1)
    print("LOVEDA", res, flush=True)


def run_eurosat(fx, dev, data_dir, OUT):
    from datasets import load_dataset
    ds = load_dataset(os.path.join(data_dir, "EuroSAT_RGB"))["train"]
    cols = ds.column_names
    ikey = "image" if "image" in cols else cols[0]
    lkey = "label" if "label" in cols else cols[1]
    feats, labels = [], []
    for i in range(0, len(ds), BS):
        rows = ds[i:i + BS]
        batch = torch.stack([to_input(p) for p in rows[ikey]])
        f = fx(batch, 5000 + i)
        pooled = torch.cat([f[n].mean(dim=1) for n in ["g4", "g8", "g16", "g24"]], dim=1)
        feats.append(pooled); labels.extend(rows[lkey])
        if (i // BS) % 50 == 0:
            print(f"eurosat {i}/{len(ds)}", flush=True)
    X = torch.cat(feats).float(); y = torch.tensor(labels)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X)); ntr = int(0.8 * len(X))
    tr, va = order[:ntr], order[ntr:]
    W = torch.nn.Linear(X.shape[1], int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3)
    Xd, yd = X.to(dev), y.to(dev)
    for ep in range(40):
        for bi in range(0, ntr, 1024):
            idx = torch.from_numpy(tr[bi:bi + 1024]).to(dev)
            loss = F.cross_entropy(W(Xd[idx]), yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((W(Xd[va]).argmax(1) == yd[va]).float().mean())
    res = {"top1": acc, "n_train": ntr, "n_val": len(X) - ntr}

    # Pre-registered tau-sweep ablation on a fixed 5,000-image subset (same 80/20 split rule)
    sub = np.linspace(0, len(ds) - 1, 5000, dtype=int)
    res["tau_sweep_5k"] = {}
    for tau in TAU_SWEEP:
        fs, ls = [], []
        for i in range(0, len(sub), BS):
            rows = ds[[int(v) for v in sub[i:i + BS]]]
            batch = torch.stack([to_input(p) for p in rows[ikey]])
            f = fx(batch, 5000 + i, tau=tau)
            fs.append(torch.cat([f[n_].mean(dim=1) for n_ in ["g4", "g8", "g16", "g24"]], dim=1))
            ls.extend(rows[lkey])
        Xs = torch.cat(fs).float().to(dev); ys = torch.tensor(ls).to(dev)
        o = np.random.default_rng(0).permutation(len(Xs)); k = int(0.8 * len(Xs))
        Ws = torch.nn.Linear(Xs.shape[1], int(ys.max()) + 1).to(dev)
        opts = torch.optim.Adam(Ws.parameters(), lr=1e-3)
        for ep in range(40):
            for bi in range(0, k, 1024):
                idx = torch.from_numpy(o[bi:bi + 1024]).to(dev)
                l = F.cross_entropy(Ws(Xs[idx]), ys[idx])
                opts.zero_grad(); l.backward(); opts.step()
        with torch.no_grad():
            va_idx = torch.from_numpy(o[k:]).to(dev)
            res["tau_sweep_5k"][str(tau)] = float((Ws(Xs[va_idx]).argmax(1) == ys[va_idx]).float().mean())
        print("tau", tau, res["tau_sweep_5k"][str(tau)], flush=True)

    json.dump(res, open(f"{OUT}/eurosat_results.json", "w"), indent=1)
    print("EUROSAT", res, flush=True)


def run_bright(fx, dev, data_dir, OUT):
    base = os.path.join(data_dir, "BRIGHT", "prepared")

    def prep(split, cap):
        pres = sorted(glob.glob(f"{base}/{split}/pre-event/*"))
        items, ys = [], []
        idxs = np.linspace(0, len(pres) - 1, min(len(pres), cap), dtype=int)
        for i in idxs:
            p = pres[int(i)]
            b = os.path.basename(p)
            post = p.replace("pre-event", "post-event").replace("_pre_", "_post_")
            tgt = p.replace("pre-event", "target").replace("_pre_", "_target_")
            if not os.path.exists(post):
                cand = glob.glob(f"{base}/{split}/post-event/{b.split('_pre')[0]}*")
                post = cand[0] if cand else None
            if not os.path.exists(tgt):
                cand = glob.glob(f"{base}/{split}/target/{b.split('_pre')[0]}*")
                tgt = cand[0] if cand else None
            if post is None or tgt is None:
                continue
            items.append({"imgs": [to_input(Image.open(p)), to_input(Image.open(post))]})
            ys.append(mask64(Image.open(tgt)))
        return items, torch.stack(ys).numpy()

    tr_items, Ytr = prep("train", 2500)
    va_items, Yva = prep("val", 1000)
    print(f"bright pairs: train {len(tr_items)} val {len(va_items)}; label values "
          f"{np.unique(Ytr)[:10]}", flush=True)
    Xtr = extract_set(fx, tr_items, f"{OUT}/bright_train.npy", 7000)
    Xva = extract_set(fx, va_items, f"{OUT}/bright_val.npy", 95000)

    class SiamView:
        """View memmap [N,2,4,256,4096] as [N, 3*4*256*4096-equivalent] via __getitem__."""
        def __init__(self, X):
            self.X = X
            self.shape = (X.shape[0],)

        def __getitem__(self, idx):
            a = np.array(self.X[idx])                      # [b,2,4,256,4096]
            pre, post = a[:, 0], a[:, 1]
            return np.concatenate([pre, post, np.abs(pre.astype(np.float32) -
                                                     post.astype(np.float32)).astype(np.float16)],
                                  axis=1)                  # [b,12,256,4096]

    ig = 255 if Ytr.max() > 10 else -1
    valid_max = int(max(Ytr[Ytr != ig].max() if ig != -1 else Ytr.max(),
                        Yva[Yva != ig].max() if ig != -1 else Yva.max()))
    res = train_pixel_probe(SiamView(Xtr), Ytr, SiamView(Xva), Yva, ncls=valid_max + 1,
                            dev=dev, ignore=ig, dim=12 * 4096)
    json.dump(res, open(f"{OUT}/bright_results.json", "w"), indent=1)
    print("BRIGHT", res, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Frozen linear probes on GeoCore-9B DiT features")
    ap.add_argument("--task", required=True, choices=["loveda", "eurosat", "bright"])
    ap.add_argument("--ckpt", required=True, help="Pre-trained GeoCore-9B checkpoint (.pt)")
    ap.add_argument("--vae-dir", required=True, help="Path to the Flux2 VAE ae.safetensors")
    ap.add_argument("--data-root", required=True,
                    help="Directory holding LoveDA/, EuroSAT_RGB/ and BRIGHT/")
    ap.add_argument("--out", default=os.path.join(HERE, "disc_probe"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    fx = FeatExtractor(dev, args.ckpt, args.vae_dir)
    {"loveda": run_loveda, "eurosat": run_eurosat, "bright": run_bright}[args.task](
        fx, dev, args.data_root, args.out)


if __name__ == "__main__":
    main()
