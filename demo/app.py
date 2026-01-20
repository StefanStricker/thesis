import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match training class order (ImageFolder is alphabetical by default)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# Path to your photo model
MODEL_OPTIONS = {
    "ResNet50 – Baseline": {
        "path": "trained_models/resnet50_baseline_seed64.pth",
        "description": "Trained on source domain without photometric augmentations."
    },
    "ResNet50 – Photo Augmentations": {
        "path": "trained_models/resnet50_photo_seed64.pth",
        "description": "Trained with photometric augmentations to improve robustness under domain shift."
    }
}



NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])

st.subheader("Model selection")

model_name = st.selectbox(
    "Choose a model",
    options=list(MODEL_OPTIONS.keys())
)

st.caption(MODEL_OPTIONS[model_name]["description"])


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model(weights_path: str):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(weights_path, map_location="cpu")

    # If you saved a checkpoint dict, pull out the state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        # already a raw state_dict
        state = checkpoint

    # Handle "module." prefix (DataParallel)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model

def predict_topk(model, img: Image.Image, k: int = 3):
    x = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu()

    top_probs, top_idx = torch.topk(probs, k=min(k, NUM_CLASSES))
    results = [(CLASS_NAMES[i], float(p)) for p, i in zip(top_probs, top_idx)]
    return results, probs

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Waste Classification Demo", page_icon="🗑️", layout="centered")

st.title("🗑️ Waste Classification Demo (ResNet50 + Photo Augmentations)")
st.write(
    "Upload an image and the model will predict one of: "
    + ", ".join([f"**{c}**" for c in CLASS_NAMES]) + "."
)
st.caption("Inference-only demo. No target-domain training or domain adaptation is performed.")


weights_path = MODEL_OPTIONS[model_name]["path"]

try:
    model = load_model(weights_path)
except Exception as e:
    st.error(f"Could not load model from `{weights_path}`.\n\nError: {e}")
    st.stop()


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    results, probs = predict_topk(model, img, k=3)

    st.subheader("Prediction")
    st.markdown(f"### Results using **{model_name}**")
    st.write(f"**{results[0][0]}** (confidence: {results[0][1]:.3f})")

    st.subheader("Top-3 probabilities")
    for name, p in results:
        st.write(f"- {name}: {p:.3f}")

    with st.expander("Show full probability vector"):
        for cls, p in zip(CLASS_NAMES, probs.tolist()):
            st.write(f"{cls}: {p:.4f}")
