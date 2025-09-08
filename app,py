import io, time, os, hashlib, pathlib, requests
import timm
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    vit_b_16, ViT_B_16_Weights,
    densenet121, DenseNet121_Weights
)

# ===== Lightning allowlist for torch.load =====
from contextlib import contextmanager
try:
    from torch.serialization import add_safe_globals, safe_globals
    HAVE_SAFE_GLOBALS = True
except Exception:
    HAVE_SAFE_GLOBALS = False
    def add_safe_globals(_): return
    @contextmanager
    def safe_globals(_=None): yield

try:
    import lightning.fabric.wrappers as lwrap
    try:
        add_safe_globals([lwrap._FabricModule])
    except Exception:
        pass
except Exception:
    lwrap = None

# ===== Streamlit header =====
st.set_page_config(page_title="WBC Classifier Demo", page_icon="🧪", layout="wide")
st.title("🧪 White Blood Cell Classifier – 5 Models")
st.caption("โหลด checkpoint จาก Hugging Face (fold=0) → เลือกสถาปัตยกรรม → อัปโหลดรูปทำนายได้เลย")

# ===== Labels (5 WBC types) =====
LABELS = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

# ===== URLs ของน้ำหนัก (เฉพาะ fold 0) =====
WEIGHT_URLS = {
    "MobileNetV3-Large-100": "https://huggingface.co/thakchinan/weather-ckptss/resolve/main/mobilenetv3_large_100_fold0.pt",
    "EfficientNet-B0":        "https://huggingface.co/thakchinan/weather-ckptss/resolve/main/efficientnet_b0_fold0.pt",
    "ResNet50":               "https://huggingface.co/thakchinan/weather-ckptss/resolve/main/resnet50_fold0.pt",
    "ViT-Base-Patch16-224":   "https://huggingface.co/thakchinan/weather-ckptss/resolve/main/vit_b16_fold0.pt",
    "DenseNet121":            "https://huggingface.co/thakchinan/weather-ckptss/resolve/main/densenet121_fold0.pt",
}

# ===== Sidebar =====
st.sidebar.header("Model Settings")
arch = st.sidebar.selectbox("Architecture", list(WEIGHT_URLS.keys()), index=2)
device_opt = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
device = torch.device(device_opt if (device_opt == "cuda" and torch.cuda.is_available()) else "cpu")
st.sidebar.caption(f"Using **{device}**")

topk = st.sidebar.slider("Top-K", 1, 5, 3)
use_builtin_transforms = st.sidebar.checkbox("Use pretrained built-in transforms (recommended)", value=True)

st.write(f"Torch: {torch.__version__} | timm: {timm.__version__} | CUDA: {torch.cuda.is_available()} | device: {device}")

# ===== Utils =====
def default_imagenet_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def ensure_weight(url: str) -> str:
    os.makedirs("weights", exist_ok=True)
    fname = pathlib.Path("weights") / (hashlib.md5(url.encode()).hexdigest() + ".pt")
    if not fname.exists():
        with st.spinner("Downloading weights…"):
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            fname.write_bytes(r.content)
    return str(fname)

def build_model_and_preprocess(arch_name: str, num_classes: int):
    if arch_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "MobileNetV3-Large-100":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "EfficientNet-B0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "ViT-Base-Patch16-224":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    elif arch_name == "DenseNet121":
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        preprocess = weights.transforms() if use_builtin_transforms else default_imagenet_transform(224)

    else:
        raise ValueError("Unknown architecture")

    return model, preprocess

def load_checkpoint_auto(local_path: str, arch_name: str, num_classes: int):
    if lwrap is not None and HAVE_SAFE_GLOBALS:
        with safe_globals([lwrap._FabricModule]):
            obj = torch.load(local_path, map_location="cpu", weights_only=False)
    else:
        obj = torch.load(local_path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model = obj.module if hasattr(obj, "module") and isinstance(obj.module, torch.nn.Module) else obj
        preprocess = default_imagenet_transform(224)
        return model, preprocess, "full_model"

    if isinstance(obj, dict):
        state = obj
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ("model.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_state[nk] = v

        model, preprocess = build_model_and_preprocess(arch_name, len(LABELS))
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        return model, preprocess, f"state_dict (missing {len(missing)}, unexpected {len(unexpected)})"

    raise ValueError("Unknown checkpoint format")

# ===== Load model =====
ckpt_url = WEIGHT_URLS[arch]
ckpt_path = ensure_weight(ckpt_url)
st.sidebar.caption(f"Checkpoint (cached): {ckpt_path}")

try:
    model, preprocess, how = load_checkpoint_auto(ckpt_path, arch, len(LABELS))
    st.sidebar.success(f"✅ Loaded checkpoint ({how}) for {arch}")
except Exception as e:
    st.sidebar.error(f"Load error: {e}")
    st.stop()

model.to(device).eval()

# ===== Prediction =====
st.subheader("Upload images to predict")
files = st.file_uploader("อัปโหลดรูปภาพ (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

def predict_one(pil_img: Image.Image):
    t0 = time.time()
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    dt = time.time() - t0
    k = min(topk, probs.shape[0])
    top_p, top_i = torch.topk(probs, k=k)
    results = [(LABELS[idx], float(p)) for p, idx in zip(top_p.tolist(), top_i.tolist())]
    return results, dt

if not files:
    st.info("⬆️ ลากรูปมาวางเพื่อทำนาย (เลือกได้หลายไฟล์)")
else:
    cols = st.columns(st.slider("Columns", 1, 5, 3))
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        res, elapsed = predict_one(img)
        with cols[i % len(cols)]:
            st.image(img, caption=f.name, use_column_width=True)
            st.caption(f"⏱ {elapsed:.3f}s")
            for r, (name, p) in enumerate(res, start=1):
                st.write(f"{r}. **{name}** — {p*100:.2f}%")
