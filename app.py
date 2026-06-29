import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from ultralytics import YOLO
from PIL import Image
import json, time, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.set_page_config(page_title="DL Model Comparison", page_icon="🤖", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    }
    .title-box h1 { color: white; font-size: 2.5em; margin: 0; }
    .title-box p  { color: #e0e0e0; font-size: 1.1em; margin: 5px 0 0 0; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-card h3 { color: #a0aec0; font-size: 0.9em; margin: 0; text-transform: uppercase; }
    .metric-card h2 { color: white; font-size: 1.8em; margin: 5px 0 0 0; }
    .model-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .vit-badge  { background: #4C9BE8; color: white; }
    .yolo-badge { background: #E8774C; color: white; }
    .result-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #2d3250;
        margin-top: 10px;
    }
    div[data-testid="stTabs"] button {
        font-size: 1em;
        font-weight: bold;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🤖 Deep Learning Model Comparison</h1>
    <p>Fine-tuned YOLOv8 vs ViT-Base on Cats & Dogs Dataset | CC3 Project 2025/2026</p>
</div>
""", unsafe_allow_html=True)

# ── Load Models ─────────────────────────────────────────────────
@st.cache_resource
def load_vit():
    model = ViTForImageClassification.from_pretrained("models/vit_finetuned")
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    weights = "runs/detect/models/yolo_finetuned/train/weights/best.pt"
    return YOLO(weights)

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Model Status")
    try:
        vit_model = load_vit()
        st.success("✅ ViT-Base loaded")
    except Exception as e:
        st.error(f"❌ ViT error: {e}")
        vit_model = None

    try:
        yolo_model = load_yolo()
        st.success("✅ YOLOv8 loaded")
    except Exception as e:
        st.error(f"❌ YOLO error: {e}")
        yolo_model = None

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Dataset:** Cats vs Dogs  
    **Classes:** 2 (cats, dogs)  
    **Train images:** 180  
    **Val images:** 60  
    **Test images:** 60  
    """)
    st.markdown("---")
    st.markdown("## 🎓 Project Info")
    st.markdown("""
    **Filière:** IA - TS  
    **Groupe:** TS-DIA-IA-102  
    **Formateur:** Mr. SABER  
    **Année:** 2025/2026  
    """)

# ── Tabs ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Test Models", "📊  Comparison Dashboard"])

# ════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload an image to test both models")
    
    col_upload, col_options = st.columns([2, 1])
    
    with col_upload:
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                     label_visibility="collapsed")
    with col_options:
        st.markdown("<br>", unsafe_allow_html=True)
        model_choice = st.selectbox("🤖 Select Model", 
                                     ["Both Models", "ViT only", "YOLO only"])
        run_btn = st.button("🚀 Run Inference", use_container_width=True)

    if uploaded and run_btn:
        image = Image.open(uploaded).convert("RGB")
        
        st.markdown("---")
        
        if model_choice == "Both Models":
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
            col3 = col2

        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(image, use_column_width=True)

        # ── ViT ──
        if model_choice in ["Both Models", "ViT only"] and vit_model:
            col = col2
            with col:
                st.markdown('<p class="model-badge vit-badge">ViT-Base Result</p>', 
                           unsafe_allow_html=True)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                ])
                tensor = transform(image).unsqueeze(0)
                start = time.time()
                with torch.no_grad():
                    outputs = vit_model(pixel_values=tensor).logits
                latency = (time.time() - start) * 1000
                probs = torch.softmax(outputs, dim=1)[0]
                pred = outputs.argmax(dim=1).item()
                classes = ["cats", "dogs"]
                emoji = "🐱" if classes[pred] == "cats" else "🐶"
                
                st.image(image, use_column_width=True)
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="color:white">{emoji} {classes[pred].upper()}</h3>
                    <p style="color:#a0aec0">Confidence: <b style="color:#4C9BE8">{probs[pred].item()*100:.1f}%</b></p>
                    <p style="color:#a0aec0">Latency: <b style="color:#48BB78">{latency:.1f} ms</b></p>
                </div>
                """, unsafe_allow_html=True)

        # ── YOLO ──
        if model_choice in ["Both Models", "YOLO only"] and yolo_model:
            col = col3
            with col:
                st.markdown('<p class="model-badge yolo-badge">YOLOv8 Result</p>', 
                           unsafe_allow_html=True)
                start = time.time()
                results = yolo_model(image, verbose=False)
                latency = (time.time() - start) * 1000
                result_img = results[0].plot()
                result_img = Image.fromarray(result_img[:, :, ::-1])
                boxes = results[0].boxes
                
                st.image(result_img, use_column_width=True)
                if len(boxes) > 0:
                    conf = float(boxes.conf[0]) * 100
                    cls = int(boxes.cls[0])
                    classes = ["cats", "dogs"]
                    emoji = "🐱" if classes[cls] == "cats" else "🐶"
                    st.markdown(f"""
                    <div class="result-box">
                        <h3 style="color:white">{emoji} {classes[cls].upper()}</h3>
                        <p style="color:#a0aec0">Confidence: <b style="color:#E8774C">{conf:.1f}%</b></p>
                        <p style="color:#a0aec0">Latency: <b style="color:#48BB78">{latency:.1f} ms</
                        ></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No detection found")
                    st.markdown(f"Latency: **{latency:.1f} ms**")

# ════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Performance Dashboard")

    vit_metrics = {}
    yolo_metrics = {}

    if os.path.exists("models/vit_finetuned/metrics_vit.json"):
        with open("models/vit_finetuned/metrics_vit.json") as f:
            vit_metrics = json.load(f)
    if os.path.exists("models/yolo_finetuned/metrics_yolo.json"):
        with open("models/yolo_finetuned/metrics_yolo.json") as f:
            yolo_metrics = json.load(f)

    # Top metric cards
    st.markdown("#### 🏆 Key Metrics at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>ViT F1 Score</h3>
            <h2 style="color:#4C9BE8">{vit_metrics.get('f1', 'N/A')}</h2>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h3>YOLO F1 Score</h3>
            <h2 style="color:#E8774C">{yolo_metrics.get('f1', 'N/A')}</h2>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h3>ViT Train Time</h3>
            <h2 style="color:#48BB78">{vit_metrics.get('training_time_seconds', 'N/A')}s</h2>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h3>YOLO Train Time</h3>
            <h2 style="color:#48BB78">{yolo_metrics.get('training_time_seconds', 'N/A')}s</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table + Chart side by side
    col_table, col_chart = st.columns(2)

    with col_table:
        st.markdown("#### 📋 Full Metrics Table")
        data = {
            "Metric":    ["Precision", "Recall", "F1 Score", "Training Time (s)", "Epochs"],
            "ViT 🔵":   [vit_metrics.get("precision","N/A"),  vit_metrics.get("recall","N/A"),
                          vit_metrics.get("f1","N/A"),         vit_metrics.get("training_time_seconds","N/A"),
                          vit_metrics.get("epochs","N/A")],
            "YOLO 🟠":  [yolo_metrics.get("precision","N/A"), yolo_metrics.get("recall","N/A"),
                          yolo_metrics.get("f1","N/A"),        yolo_metrics.get("training_time_seconds","N/A"),
                          yolo_metrics.get("epochs","N/A")]
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("#### 📈 Visual Comparison")
        metrics_names = ["Precision", "Recall", "F1 Score"]
        vit_vals  = [vit_metrics.get("precision",0),  vit_metrics.get("recall",0),  vit_metrics.get("f1",0)]
        yolo_vals = [yolo_metrics.get("precision",0), yolo_metrics.get("recall",0), yolo_metrics.get("f1",0)]

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1e2130")
        ax.set_facecolor("#1e2130")
        x = np.arange(len(metrics_names))
        bars1 = ax.bar(x - 0.2, vit_vals,  0.4, label="ViT",  color="#4C9BE8", zorder=3)
        bars2 = ax.bar(x + 0.2, yolo_vals, 0.4, label="YOLO", color="#E8774C", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, color="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2d3250", labelcolor="white")
        ax.grid(axis="y", alpha=0.2, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3250")
        st.pyplot(fig)

    # Learning curves
    st.markdown("#### 📉 Learning Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ViT Training Curves**")
        if os.path.exists("models/vit_finetuned/training_curves.png"):
            st.image("models/vit_finetuned/training_curves.png", use_column_width=True)
    with col2:
        st.markdown("**YOLO Training Curves**")
        if os.path.exists("runs/detect/models/yolo_finetuned/train/results.png"):
            st.image("runs/detect/models/yolo_finetuned/train/results.png", use_column_width=True)

    # Model sizes
    st.markdown("#### 💾 Model Sizes")
    def get_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return round(total / (1024*1024), 1)

    c1, c2 = st.columns(2)
    c1.metric("ViT Model Size",  str(get_size("models/vit_finetuned")) + " MB",  "Classification")
    c2.metric("YOLO Model Size", str(get_size("models/yolo_finetuned")) + " MB", "Detection")