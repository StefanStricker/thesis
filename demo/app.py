import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import math
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Waste Classification Demo", layout="centered")

# Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_REPO_ID = "StefanStricker/waste_resnet50_models"

MODEL_OPTIONS = {
    "ResNet50 – Baseline": {
        "filename": "resnet50_baseline_seed64.pth",
        "description": "Trained on source domain without photometric augmentations."
    },
    "ResNet50 – Photo Augmentations": {
        "filename": "resnet50_photo_seed64.pth",
        "description": "Trained with photometric augmentations to improve robustness under domain shift."
    },
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


# Helpers

@st.cache_resource
def get_weights_path(filename: str) -> str:
    return hf_hub_download(repo_id=MODEL_REPO_ID, filename=filename)

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
        state = checkpoint

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

def compute_uncertainty_metrics(probs: torch.Tensor):
    p = probs.detach().cpu().float()
    top2_vals, top2_idx = torch.topk(p, k=min(2, p.numel()))
    top1_p = float(top2_vals[0])
    top1_i = int(top2_idx[0])
    top2_p = float(top2_vals[1]) if p.numel() > 1 else 0.0

    margin = top1_p - top2_p

    eps = 1e-12
    entropy = float(-(p * (p + eps).log()).sum())

    max_entropy = math.log(len(p)) if len(p) > 1 else 1.0
    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "top1_prob": top1_p,
        "top1_idx": top1_i,
        "top2_prob": top2_p,
        "margin": float(margin),
        "entropy": entropy,
        "entropy_norm": entropy_norm
    }

def plot_prob_bar_chart(class_names, probs_list, top_idx=None):
    fig = plt.figure(figsize=(7, 3.6))
    xs = np.arange(len(class_names))
    plt.bar(xs, probs_list)
    plt.xticks(xs, class_names, rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Probability")
    plt.title("Class probability distribution")
    if top_idx is not None:
        plt.axvline(top_idx, linestyle="--", linewidth=1)
    plt.tight_layout()
    return fig

# Grad-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward()

        # Global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

def overlay_cam_on_image(img_pil, cam, alpha=0.6):

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_resized.resize(img_pil.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_resized).astype(np.float32) / 255.0

    img = np.array(img_pil).astype(np.float32) / 255.0

    heatmap = np.zeros_like(img)
    heatmap[..., 0] = cam_resized  

    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


# UI

st.title("Waste Classification Demo (ResNet50)")
st.write(
    "Upload an image and the model will predict one of: "
    + ", ".join([f"**{c}**" for c in CLASS_NAMES]) + "."
)
st.caption("Inference-only demo. No target-domain training or domain adaptation is performed.")

st.subheader("Model selection")

model_name = st.selectbox("Choose a model", options=list(MODEL_OPTIONS.keys()))
st.caption(MODEL_OPTIONS[model_name]["description"])

filename = MODEL_OPTIONS[model_name]["filename"]
try:
    weights_path = get_weights_path(filename)
    model = load_model(weights_path)
except Exception as e:
    st.error(f"Could not load model `{filename}` from `{MODEL_REPO_ID}`.\n\nError: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=False)

    results, probs = predict_topk(model, img, k=3)

    # Confidence / uncertainty controls
    st.subheader("Confidence & uncertainty")

    confidence_threshold = st.slider(
        "Confidence threshold (flag low-confidence predictions)",
        min_value=0.00,
        max_value=0.99,
        value=0.60,
        step=0.01
    )

    show_bar_chart = st.checkbox("Show probability bar chart", value=True)

    metrics = compute_uncertainty_metrics(probs)
    top1_name = CLASS_NAMES[metrics["top1_idx"]]
    top1_conf = metrics["top1_prob"]

    low_conf = top1_conf < confidence_threshold
    low_margin = metrics["margin"] < 0.15
    high_entropy = metrics["entropy_norm"] > 0.75

    if low_conf or low_margin or high_entropy:
        st.warning(
            "Model uncertainty is high for this image.\n\n"
            f"- Top-1 confidence: {top1_conf:.3f}\n"
            f"- Margin (top1 - top2): {metrics['margin']:.3f}\n"
            f"- Normalized entropy: {metrics['entropy_norm']:.3f}\n\n"
            "Consider trying a closer crop, better lighting, or a simpler background."
        )
    else:
        st.success(
            "Model confidence looks reasonable.\n\n"
            f"- Top-1 confidence: {top1_conf:.3f}\n"
            f"- Margin (top1 - top2): {metrics['margin']:.3f}\n"
            f"- Normalized entropy: {metrics['entropy_norm']:.3f}"
        )

    # Main prediction display
    st.subheader("Prediction")
    st.markdown(f"### Results using **{model_name}**")
    st.write(f"**{top1_name}** (confidence: {top1_conf:.3f})")

    if show_gradcam:
        st.subheader("Grad-CAM explanation")
    
        # Prepare input tensor
        input_tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        class_idx = metrics["top1_idx"]
    
        # ResNet50 last convolutional block
        target_layer = model.layer4[-1]
    
        cam_generator = GradCAM(model, target_layer)
        cam = cam_generator.generate(input_tensor, class_idx)
    
        cam_overlay = overlay_cam_on_image(img.resize((224, 224)), cam)
    
        st.image(
            cam_overlay,
            caption="Grad-CAM highlights regions contributing most to the predicted class",
            use_container_width=True
        )
    
        st.caption(
            "Grad-CAM provides a qualitative indication of spatial regions influencing the prediction "
        )


    st.subheader("Top-3 probabilities")
    for name, p in results:
        st.write(f"- {name}: {p:.3f}")

    if show_bar_chart:
        fig = plot_prob_bar_chart(CLASS_NAMES, probs.tolist(), top_idx=metrics["top1_idx"])
        st.pyplot(fig)

    with st.expander("Show full probability vector"):
        for cls, p in zip(CLASS_NAMES, probs.tolist()):
            st.write(f"{cls}: {p:.4f}")