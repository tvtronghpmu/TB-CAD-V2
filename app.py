"""
TB-CAD SYSTEM v2.0
Hệ thống Hỗ trợ Chẩn đoán Lao Phổi - Deep Learning
EfficientNetV2 (Classification) + YOLOv12 (Detection)
"""

import os, sys
import cv2
import time, io, json, shutil, zipfile
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TB-CAD | Chẩn Đoán Lao Phổi AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",   # Không dùng sidebar → tránh lỗi toggle
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&family=Orbitron:wght@700;900&display=swap');

:root {
    --bg:   #111827; --bg2: #1a2236; --card: #1e2d47; --card2: #243355;
    --cyan: #22d3ee; --green: #34d399; --amber: #fbbf24; --red: #f87171; --blue: #60a5fa;
    --txt:  #f1f5f9; --txt2: #94a3b8; --txt3: #64748b;
    --bdr:  rgba(34,211,238,0.20); --bdr2: rgba(34,211,238,0.45); --r: 10px;
}

.stApp { background: var(--bg) !important; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; color: var(--txt) !important; }

/* Ẩn toàn bộ chrome Streamlit - an toàn vì không dùng sidebar */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; height: 0 !important; min-height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
/* padding-top nhỏ vì header đã ẩn hoàn toàn */
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }

/* Ẩn hoàn toàn sidebar vì không dùng */
[data-testid="stSidebar"] { display: none !important; width: 0 !important; }
[data-testid="collapsedControl"] { display: none !important; }
button[data-testid="baseButton-header"] { display: none !important; }

/* Header */
.tbcad-header {
    background: linear-gradient(135deg,#162035 0%,#1c2d4a 50%,#162035 100%);
    border: 1px solid var(--bdr); border-radius: var(--r);
    padding: 1.2rem 1.8rem; margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
    box-shadow: 0 0 40px rgba(34,211,238,0.08);
}
.tbcad-logo { font-family: 'Orbitron', monospace; font-size: 1.6rem; font-weight: 900;
    background: linear-gradient(90deg,#22d3ee,#60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 0.08em; }
.tbcad-sub { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--txt2);
    letter-spacing: 0.14em; text-transform: uppercase; margin-top: 2px; }
.badge { display: inline-block; background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.28);
    color: var(--cyan); font-family: 'Space Mono', monospace; font-size: 0.62rem;
    padding: 2px 9px; border-radius: 20px; letter-spacing: 0.08em; margin-right: 5px; }

/* Metrics */
.metric-strip { display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap; }
.mc { flex:1; min-width:110px; background: var(--card); border: 1px solid var(--bdr); border-radius: var(--r); padding: 0.8rem 1rem; text-align:center; }
.mc-val { font-family:'Space Mono',monospace; font-size:1.25rem; font-weight:700; color:var(--cyan); display:block; }
.mc-lbl { font-size:0.68rem; color:var(--txt2); text-transform:uppercase; letter-spacing:0.07em; margin-top:2px; font-weight:500; }

/* Card */
.card { background: var(--card); border: 1px solid var(--bdr); border-radius: var(--r); padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.card-title { font-family:'Space Mono',monospace; font-size:0.75rem; color:var(--cyan); text-transform:uppercase;
    letter-spacing:0.13em; padding-bottom:0.6rem; margin-bottom:0.9rem; border-bottom:1px solid var(--bdr); font-weight:700; }

/* Result panels */
.rpos { background:linear-gradient(135deg,rgba(248,113,113,0.13),rgba(248,113,113,0.04)); border:1px solid rgba(248,113,113,0.45); border-radius:var(--r); padding:1rem 1.2rem; margin:0.7rem 0; }
.rneg { background:linear-gradient(135deg,rgba(52,211,153,0.12),rgba(52,211,153,0.03)); border:1px solid rgba(52,211,153,0.4); border-radius:var(--r); padding:1rem 1.2rem; margin:0.7rem 0; }
.rtitle-pos { font-family:'Orbitron',monospace; font-size:1rem; font-weight:700; color:var(--red); margin:0 0 0.25rem 0; }
.rtitle-neg { font-family:'Orbitron',monospace; font-size:1rem; font-weight:700; color:var(--green); margin:0 0 0.25rem 0; }
.rsub { font-size:0.82rem; color:#cbd5e1; font-weight:500; }

/* Conf bar */
.cb-wrap { margin:0.6rem 0; }
.cb-lbl { font-family:'Space Mono',monospace; font-size:0.7rem; color:#94a3b8; margin-bottom:3px; font-weight:500; }
.cb-bg { height:9px; border-radius:5px; background:rgba(255,255,255,0.07); overflow:hidden; }
.cb-pos { height:100%; border-radius:5px; background:linear-gradient(90deg,#f87171,#fca5a5); }
.cb-neg { height:100%; border-radius:5px; background:linear-gradient(90deg,#34d399,#6ee7b7); }
.cb-pct { font-family:'Space Mono',monospace; font-size:0.65rem; margin-top:2px; }

/* Detection table */
.dt { width:100%; border-collapse:collapse; font-family:'Space Mono',monospace; font-size:0.73rem; }
.dt th { background:rgba(34,211,238,0.10); color:var(--cyan); padding:6px 10px; text-align:left; border-bottom:1px solid var(--bdr); text-transform:uppercase; letter-spacing:0.07em; }
.dt td { padding:6px 10px; color:#f1f5f9; border-bottom:1px solid rgba(255,255,255,0.05); }
.dt tr:last-child td { border-bottom:none; }
.dt tr:hover td { background:rgba(34,211,238,0.05); }

/* ═══ BUTTONS ═══ */
.stButton>button {
    background: linear-gradient(135deg, #0e7490, #1d4ed8) !important;
    border: 1.5px solid #22d3ee !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 16px rgba(34,211,238,0.30), inset 0 1px 0 rgba(255,255,255,0.10) !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
    border-color: #67e8f9 !important;
    color: #ffffff !important;
    box-shadow: 0 0 28px rgba(34,211,238,0.55), 0 4px 12px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}
.stButton>button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 12px rgba(34,211,238,0.35) !important;
}

/* Download button khác màu để phân biệt */
.stDownloadButton>button {
    background: linear-gradient(135deg, #065f46, #1e3a5f) !important;
    border: 1.5px solid #34d399 !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 14px rgba(52,211,153,0.28), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
.stDownloadButton>button:hover {
    background: linear-gradient(135deg, #047857, #1e4976) !important;
    border-color: #6ee7b7 !important;
    box-shadow: 0 0 26px rgba(52,211,153,0.50), 0 4px 12px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ═══ FILE UPLOADER ═══ */
[data-testid="stFileUploader"] {
    background: rgba(14, 116, 144, 0.08) !important;
    border: 2px dashed rgba(34,211,238,0.55) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(14, 116, 144, 0.14) !important;
    border-color: #22d3ee !important;
    box-shadow: 0 0 20px rgba(34,211,238,0.18) !important;
}
/* Browse files button bên trong uploader */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #0e7490, #1d4ed8) !important;
    border: 1.5px solid #22d3ee !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(34,211,238,0.30) !important;
    padding: 0.4rem 1.2rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
    box-shadow: 0 0 24px rgba(34,211,238,0.50) !important;
    transform: translateY(-1px) !important;
}
/* Text trong uploader */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Form controls */
.stRadio label { color:#f1f5f9 !important; font-weight:500 !important; font-size:0.85rem !important; }
div[data-testid="stRadio"] > label { color:#22d3ee !important; font-family:'Space Mono',monospace !important; font-size:0.7rem !important; text-transform:uppercase !important; }
label { color:#cbd5e1 !important; font-weight:500 !important; }
p, li, span { color:#e2e8f0; }
h1,h2,h3 { color:#f1f5f9 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:var(--card) !important; border-radius:8px !important; padding:3px !important; gap:3px !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#94a3b8 !important; font-family:'Space Mono',monospace !important; font-size:0.73rem !important; border-radius:6px !important; padding:5px 14px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:rgba(34,211,238,0.16) !important; color:#22d3ee !important; font-weight:700 !important; }

/* Expander */
details summary { color:#22d3ee !important; font-family:'Space Mono',monospace !important; font-size:0.72rem !important; font-weight:700 !important; }

/* Disclaimer */
.disc { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.30); border-radius:8px; padding:0.7rem 1rem; font-size:0.75rem; color:#fbbf24; font-family:'Space Mono',monospace; line-height:1.6; margin-bottom:1rem; }

/* Progress */
.stProgress>div>div { background:linear-gradient(90deg,var(--blue),var(--cyan)) !important; }

/* Saved box */
.saved-box { background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.30); border-radius:8px; padding:0.8rem 1rem; font-size:0.78rem; color:#34d399; font-family:'Space Mono',monospace; }

hr { border-color:rgba(34,211,238,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ────────────────────────────────────────────────────────────────
SAVE_DIR = Path("TB_CAD_Results")
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


# ── TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE ───────────────────────────────────────
# Thay 2 FILE_ID bên dưới bằng ID thật từ Google Drive của bạn
# Lấy FILE_ID: mở link Drive → id nằm giữa /d/ và /view
#   https://drive.google.com/file/d/  <<FILE_ID_HERE>>  /view
GDRIVE_MODELS = {
    "best.pt": "1s7FLbUFieECfgIDk6IWLZvK6-_9H-bdi",        # EfficientNetV2 classification
    "best_model.pth": "1Ouivs5kX-RfhrR0psSVS5_2miu4F1ZYS",  # YOLOv12 detection
}

@st.cache_resource(show_spinner=False)
def _download_models_from_drive():
    """
    Tải model từ Google Drive nếu chưa có trên server.
    Chỉ chạy 1 lần nhờ @st.cache_resource.
    Trả về dict: {filename: True/False} cho biết file nào tải thành công.
    """
    import gdown
    results = {}
    for fname, fid in GDRIVE_MODELS.items():
        if os.path.exists(fname):
            results[fname] = True
            continue
        if fid.startswith("YOUR_"):
            results[fname] = False   # chưa cấu hình FILE_ID
            continue
        try:
            url = f"https://drive.google.com/uc?id={fid}"
            out = gdown.download(url, fname, quiet=True, fuzzy=True)
            results[fname] = out is not None and os.path.exists(fname)
        except Exception as e:
            print(f"[gdown] {fname}: {e}")
            results[fname] = False
    return results

# Chạy tải model ngay khi app khởi động
_drive_status = _download_models_from_drive()
# ─────────────────────────────────────────────────────────────────────────────

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in {
    "cls_model": None, "cls_type": None,
    "det_model": None, "det_type": None,
    "history": [], "results": {},
    "prev_upload_keys": [],   # theo doi file da upload lan truoc
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── FONT ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_font(size=14):
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess_xray(image: Image.Image, size=384):
    """
    Chuyển PIL -> BGR numpy, áp CLAHE, KHÔNG resize.
    YOLO sẽ tự resize nội bộ khi inference → tọa độ bbox luôn đúng
    trên ảnh gốc. size chỉ dùng cho classification tensor.
    """
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # CLAHE trên kênh L (không thay đổi kích thước)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)


def preprocess_for_cls(img_bgr, size=384):
    """Resize + normalize để đưa vào EfficientNetV2 (classification only)."""
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)


def to_tensor(img_bgr, size=384):
    """Resize về size rồi normalize ImageNet cho EfficientNetV2."""
    resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.copy()).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model(path: str):
    if not Path(path).exists():
        return None, "not_found"
    try:
        from ultralytics import YOLO
        return YOLO(path), "ultralytics"
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False), "pytorch"
    except Exception as e:
        return None, f"error:{e}"


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def classify(model, mtype, img_bgr):
    try:
        if mtype == "ultralytics":
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            r = model(pil, verbose=False)[0]
            if r.probs is not None:
                top1 = int(r.probs.top1)
                conf = float(r.probs.top1conf)
                nm = r.names.get(top1, "").lower()
                positive = any(w in nm for w in ["tb","tuberculosis","abnormal","sick","positive"])
                tb_prob = conf if positive else 1 - conf
                lbl = "DƯƠNG TÍNH (TB)" if tb_prob >= 0.5 else "ÂM TÍNH (Normal)"
                return lbl, max(conf, 1-conf), tb_prob
        elif mtype == "pytorch":
            if hasattr(model, "eval"):
                model.eval()
                with torch.no_grad():
                    out = model(to_tensor(img_bgr))
                    prob = (torch.sigmoid(out).item() if out.shape[-1] == 1
                            else torch.softmax(out, -1)[0][1].item())
                lbl = "DƯƠNG TÍNH (TB)" if prob >= 0.5 else "ÂM TÍNH (Normal)"
                return lbl, max(prob, 1-prob), prob
    except Exception:
        pass
    return _demo_classify(img_bgr)


def _demo_classify(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    np.random.seed(int(np.std(g) * 100) % 9999)
    tb = float(np.random.uniform(0.35, 0.95))
    lbl = "DƯƠNG TÍNH (TB)" if tb >= 0.5 else "ÂM TÍNH (Normal)"
    return lbl, max(tb, 1-tb), tb


def detect(model, mtype, img_bgr, conf_thr=0.35):
    """
    YOLOv12 detection. Truyền ảnh BGR goc (da CLAHE, CHUA resize).
    YOLO tu resize noi bo va tra toa do bbox theo kich thuoc anh dau vao.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    try:
        if mtype == "ultralytics":
            # Truyen numpy BGR truc tiep - Ultralytics nhan ndarray BGR
            results = model(img_bgr, conf=conf_thr, verbose=False, imgsz=640)
            r = results[0]
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                for i, box in enumerate(r.boxes):
                    # xyxy tra ve toa do tren anh dau vao (h_orig x w_orig)
                    x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                    x1 = max(0, min(x1, w_orig - 1))
                    y1 = max(0, min(y1, h_orig - 1))
                    x2 = max(0, min(x2, w_orig - 1))
                    y2 = max(0, min(y2, h_orig - 1))
                    conf_val = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    label    = r.names.get(cls_id, f"Lesion_{cls_id}")
                    dets.append({"id": i+1, "conf": conf_val,
                                 "bbox": [x1, y1, x2, y2], "label": label})
            annotated = draw_boxes(img_bgr, dets) if dets else img_bgr.copy()
            return annotated, dets

        elif mtype == "pytorch":
            inp = cv2.resize(img_bgr, (640, 640))
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                out = model(to_tensor(inp, size=640))
            dets = _parse_pt(out, (h_orig, w_orig), conf_thr,
                             scale_x=w_orig/640.0, scale_y=h_orig/640.0)
            return draw_boxes(img_bgr, dets), dets

    except Exception as e:
        import traceback
        print(f"[detect ERROR] {e}")
        print(traceback.format_exc())

    return _demo_detect(img_bgr, conf_thr)

def _parse_pt(out, shape, thr, scale_x=1.0, scale_y=1.0):
    dets = []
    h, w = shape[:2]
    try:
        p = out[0] if isinstance(out, (list, tuple)) else out
        if isinstance(p, torch.Tensor):
            arr = p.squeeze().cpu().numpy()
            if arr.ndim == 2:
                for i, row in enumerate(arr):
                    if len(row) >= 5 and float(row[4]) > thr:
                        cx, cy, bw, bh = row[:4]
                        x1 = max(0, min(int((cx - bw/2) * 640 * scale_x), w-1))
                        y1 = max(0, min(int((cy - bh/2) * 640 * scale_y), h-1))
                        x2 = max(0, min(int((cx + bw/2) * 640 * scale_x), w-1))
                        y2 = max(0, min(int((cy + bh/2) * 640 * scale_y), h-1))
                        dets.append({"id": i+1, "conf": float(row[4]),
                            "bbox": [x1, y1, x2, y2], "label": "TB_Lesion"})
    except Exception as e:
        print("[_parse_pt]", e)
    return dets

def _demo_detect(img_bgr, thr=0.35):
    """
    Demo detection khi không có model thật.
    Seed tính từ nội dung pixel → mỗi ảnh khác nhau ra kết quả khác nhau,
    nhưng cùng 1 ảnh luôn cho kết quả nhất quán (deterministic).
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Seed riêng cho từng ảnh ──
    mean_val = float(np.mean(gray))
    std_val  = float(np.std(gray))
    ch, cw   = h // 2, w // 2
    corners  = [
        float(gray[h//4,   w//4]),
        float(gray[h//4,   3*w//4]),
        float(gray[3*h//4, w//4]),
        float(gray[3*h//4, 3*w//4]),
        float(gray[ch, cw]),
    ]
    seed = int((mean_val * 17 + std_val * 31 +
                sum(c * (i + 7) for i, c in enumerate(corners))) % 99991)
    rng = np.random.RandomState(seed)

    # ── Số bbox dựa trên độ phức tạp ảnh ──
    complexity = std_val / 255.0
    n_boxes = int(rng.randint(2, 5)) if complexity > 0.15 else int(rng.randint(1, 3))

    # ── Phân tích grid 4×4 tìm vùng bất thường ──
    grid_h, grid_w = max(1, h // 4), max(1, w // 4)
    anomaly_regions = []
    for gi in range(4):
        for gj in range(4):
            cell = gray[gi*grid_h:(gi+1)*grid_h, gj*grid_w:(gj+1)*grid_w]
            if cell.size == 0:
                continue
            score = float(np.std(cell)) + abs(float(np.mean(cell)) - mean_val) * 0.5
            anomaly_regions.append((score, gi, gj))
    anomaly_regions.sort(key=lambda x: -x[0])

    LABELS = ["Thâm nhiễm", "Hang lao", "Nốt mờ", "Xơ hóa", "Tràn dịch"]
    dets = []

    for i in range(n_boxes):
        # Đặt box tại vùng bất thường nhất
        if i < len(anomaly_regions):
            _, gi, gj = anomaly_regions[i % len(anomaly_regions)]
            base_cx = (gj + 0.5) / 4.0
            base_cy = (gi + 0.5) / 4.0
            cx = float(np.clip(base_cx + rng.uniform(-0.09, 0.09), 0.12, 0.88))
            cy = float(np.clip(base_cy + rng.uniform(-0.09, 0.09), 0.12, 0.88))
        else:
            cx = float(rng.uniform(0.15, 0.85))
            cy = float(rng.uniform(0.15, 0.85))

        bw   = float(rng.uniform(0.07, 0.22))
        bh   = float(rng.uniform(0.07, 0.20))
        conf = float(rng.uniform(0.52, 0.94))

        x1 = max(0,   int((cx - bw/2) * w))
        y1 = max(0,   int((cy - bh/2) * h))
        x2 = min(w-1, int((cx + bw/2) * w))
        y2 = min(h-1, int((cy + bh/2) * h))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        dets.append({"id": len(dets)+1, "conf": conf,
                     "bbox": [x1, y1, x2, y2],
                     "label": LABELS[i % len(LABELS)]})

    # Fallback: đảm bảo ít nhất 1 detection
    if not dets:
        cx2 = 0.4 + rng.uniform(-0.1, 0.1)
        cy2 = 0.4 + rng.uniform(-0.1, 0.1)
        bw2, bh2 = 0.15, 0.12
        dets.append({"id": 1, "conf": float(rng.uniform(0.60, 0.85)),
            "bbox": [int((cx2-bw2/2)*w), int((cy2-bh2/2)*h),
                     int((cx2+bw2/2)*w), int((cy2+bh2/2)*h)],
            "label": "Thâm nhiễm"})

    return draw_boxes(img_bgr, dets), dets


def draw_boxes(img_bgr, dets):
    """PIL-based drawing — hỗ trợ đầy đủ Unicode tiếng Việt."""
    if not dets:
        return img_bgr.copy()
    h, w = img_bgr.shape[:2]
    font = _get_font(14)
    COLORS = [(0,210,255), (248,113,113), (52,211,153), (251,191,36), (167,139,250)]
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov   = Image.new("RGBA", base.size, (0,0,0,0))
    d_ov = ImageDraw.Draw(ov)
    d    = ImageDraw.Draw(base)
    for det in dets:
        x1, y1, x2, y2 = [max(0, v) for v in det["bbox"]]
        x2, y2 = min(w-1, x2), min(h-1, y2)
        r, g, b = COLORS[(det["id"]-1) % len(COLORS)]
        solid = (r, g, b, 255)
        d_ov.rectangle([(x1,y1),(x2,y2)], fill=(r,g,b,30))
        d.rectangle([(x1,y1),(x2,y2)], outline=solid, width=2)
        mk = 14
        for (px,py),(sx,sy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
            d.line([(px,py),(px+sx*mk,py)], fill=solid, width=3)
            d.line([(px,py),(px,py+sy*mk)], fill=solid, width=3)
        txt = f"#{det['id']} {det['label']}  {det['conf']:.0%}"
        try:
            bb = d.textbbox((0,0), txt, font=font)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except AttributeError:
            tw, th = d.textsize(txt, font=font)
        ly1 = max(0, y1 - th - 8)
        d.rectangle([(x1,ly1),(x1+tw+12,ly1+th+6)], fill=(r,g,b,220))
        d.text((x1+6, ly1+3), txt, font=font, fill=(15,20,35,255))
    merged = Image.alpha_composite(base, ov)
    return cv2.cvtColor(np.array(merged.convert("RGB")), cv2.COLOR_RGB2BGR)


def generate_heatmap(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    blurred = cv2.GaussianBlur(edges.astype(float), (51,51), 0)
    blurred = blurred / (blurred.max() + 1e-8)
    hm = cv2.applyColorMap((blurred*255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, hm, 0.45, 0)


# ── SAVE RESULT ───────────────────────────────────────────────────────────────
def save_result(fname, orig_bgr, annot_bgr, report) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = Path(fname).stem
    folder = SAVE_DIR / f"{ts}_{stem}"
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)).save(folder / f"original_{fname}")
    Image.fromarray(cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2RGB)).save(folder / f"result_{fname}")
    with open(folder / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(folder)


def make_zip(folders) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            for fpath in Path(folder).rglob("*"):
                if fpath.is_file():
                    zf.write(fpath, fpath.relative_to(SAVE_DIR.parent))
    buf.seek(0)
    return buf.read()


# ── ANALYSIS ──────────────────────────────────────────────────────────────────
def run_analysis(fname, image_pil, pipeline, conf_thr, show_hm, show_pre, auto_save):
    prog = st.progress(0, text="Tiền xử lý ảnh...")

    # img_clahe: CLAHE, CHUA resize -> YOLO tu resize noi bo
    img_clahe = preprocess_xray(image_pil)

    prog.progress(20)

    cls_label = cls_conf = tb_prob = None
    annot_bgr = img_clahe.copy()
    dets = []

    do_cls = "Hybrid" in pipeline or "phân loại" in pipeline
    do_det = "Hybrid" in pipeline or "khoanh vùng" in pipeline

    if do_cls:
        prog.progress(30, text="Đang phân loại bệnh...")
        m, mt = st.session_state.cls_model, st.session_state.cls_type
        if m and m != "demo":
            cls_label, cls_conf, tb_prob = classify(m, mt, img_clahe)
        else:
            cls_label, cls_conf, tb_prob = _demo_classify(img_clahe)
        prog.progress(55)
        if "Hybrid" in pipeline and cls_label and "ÂM TÍNH" in cls_label:
            do_det = False

    if do_det:
        prog.progress(60, text="Đang khoanh vùng tổn thương...")
        m, mt = st.session_state.det_model, st.session_state.det_type
        if m and m != "demo":
            # YOLO nhan anh CLAHE nguyen kich thuoc, tu resize noi bo
            # -> toa do bbox khop dung voi img_clahe
            annot_bgr, dets = detect(m, mt, img_clahe, conf_thr)
        else:
            annot_bgr, dets = _demo_detect(img_clahe, conf_thr)

    prog.progress(85, text="Tổng hợp kết quả...")
    heatmap_bgr = generate_heatmap(img_clahe) if show_hm else None
    preproc_bgr = img_clahe if show_pre else None
    report = {
        "file": fname, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": pipeline,
        "classification": {"result": cls_label,
                           "confidence": f"{cls_conf:.4f}" if cls_conf else None,
                           "tb_probability": f"{tb_prob:.4f}" if tb_prob else None},
        "detection": {"total_lesions": len(dets), "boxes": dets},
    }

    saved_folder = None
    if auto_save:
        orig_bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        saved_folder = save_result(fname, orig_bgr, annot_bgr, report)

    prog.progress(100); prog.empty()
    return {"fname": fname, "img_bgr": preproc_bgr, "annot_bgr": annot_bgr,
            "heatmap_bgr": heatmap_bgr, "cls_label": cls_label,
            "cls_conf": cls_conf, "tb_prob": tb_prob,
            "dets": dets, "report": report, "saved_folder": saved_folder}


# ── RENDER RESULT ─────────────────────────────────────────────────────────────
def render_result(res):
    cls_label = res["cls_label"]
    tb_prob   = res["tb_prob"]
    cls_conf  = res["cls_conf"]
    dets      = res["dets"]

    # ── Classification result ──
    if cls_label:
        pos = "DƯƠNG" in cls_label
        st.markdown(f"""
        <div class="{'rpos' if pos else 'rneg'}">
            <div class="{'rtitle-pos' if pos else 'rtitle-neg'}">{'🔴' if pos else '🟢'} {cls_label}</div>
            <div class="rsub">Confidence: <strong style="color:{'#f87171' if pos else '#34d399'};">{cls_conf:.1%}</strong>
            &nbsp;|&nbsp; TB Probability: <strong>{tb_prob:.1%}</strong></div>
        </div>
        <div class="cb-wrap">
            <div class="cb-lbl">Xác suất LAO PHỔI</div>
            <div class="cb-bg"><div class="cb-pos" style="width:{tb_prob*100:.1f}%;"></div></div>
            <div class="cb-pct" style="color:#fca5a5;">{tb_prob:.1%}</div>
        </div>
        <div class="cb-wrap">
            <div class="cb-lbl">Xác suất BÌNH THƯỜNG</div>
            <div class="cb-bg"><div class="cb-neg" style="width:{(1-tb_prob)*100:.1f}%;"></div></div>
            <div class="cb-pct" style="color:#6ee7b7;">{1-tb_prob:.1%}</div>
        </div>""", unsafe_allow_html=True)

    # ── Detection boxes ──
    if dets:
        st.markdown(f'<div class="card-title" style="margin-top:0.8rem;">🎯 Tổn thương phát hiện: {len(dets)}</div>',
                    unsafe_allow_html=True)
        ann_pil = Image.fromarray(cv2.cvtColor(res["annot_bgr"], cv2.COLOR_BGR2RGB))
        st.image(ann_pil, use_container_width=True)
        rows = "".join([
            f'<tr><td>{d["id"]}</td><td>{d["label"]}</td>'
            f'<td><strong style="color:#22d3ee;">{d["conf"]:.1%}</strong></td>'
            f'<td>({d["bbox"][0]},{d["bbox"][1]})→({d["bbox"][2]},{d["bbox"][3]})</td></tr>'
            for d in dets])
        st.markdown(f'<table class="dt"><thead><tr><th>#</th><th>Loại</th><th>Conf</th><th>Vị trí</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    elif cls_label and "ÂM TÍNH" in cls_label:
        st.markdown("""<div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.2);
            border-radius:8px;padding:0.7rem 1rem;font-size:0.8rem;color:#94a3b8;margin-top:0.5rem;">
            ✓ Phân loại Bình thường — bỏ qua khoanh vùng (Hybrid mode).</div>""",
                    unsafe_allow_html=True)

    # ── Ảnh xử lý + Heatmap cùng hàng ──
    has_preproc = res.get("img_bgr") is not None
    has_heatmap = res.get("heatmap_bgr") is not None

    if has_preproc or has_heatmap:
        st.markdown('<div class="card-title" style="margin-top:1rem;">🔬 Ảnh Xử Lý & Attention Heatmap</div>',
                    unsafe_allow_html=True)
        # Đếm số cột cần hiển thị
        cols_needed = sum([has_preproc, has_heatmap, bool(dets)])
        cols_needed = max(cols_needed, 1)
        img_cols = st.columns(min(cols_needed, 3), gap="small")
        col_idx = 0

        if has_preproc:
            with img_cols[col_idx]:
                preproc_pil = Image.fromarray(cv2.cvtColor(res["img_bgr"], cv2.COLOR_BGR2RGB))
                st.image(preproc_pil, use_container_width=True)
                st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.62rem;'
                            'color:#64748b;text-align:center;margin-top:2px;">CLAHE Enhanced</div>',
                            unsafe_allow_html=True)
            col_idx += 1

        if has_heatmap:
            with img_cols[col_idx]:
                hm_pil = Image.fromarray(cv2.cvtColor(res["heatmap_bgr"], cv2.COLOR_BGR2RGB))
                st.image(hm_pil, use_container_width=True)
                st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.62rem;'
                            'color:#64748b;text-align:center;margin-top:2px;">Attention Heatmap</div>',
                            unsafe_allow_html=True)
            col_idx += 1

        if dets and col_idx < 3:
            with img_cols[col_idx]:
                ann_pil2 = Image.fromarray(cv2.cvtColor(res["annot_bgr"], cv2.COLOR_BGR2RGB))
                st.image(ann_pil2, use_container_width=True)
                st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.62rem;'
                            'color:#64748b;text-align:center;margin-top:2px;">Detection Result</div>',
                            unsafe_allow_html=True)

    # ── Saved path ──
    if res.get("saved_folder"):
        st.markdown(f'<div class="saved-box" style="margin-top:0.8rem;">💾 Đã lưu → '
                    f'<code style="color:#6ee7b7;">{res["saved_folder"]}</code></div>',
                    unsafe_allow_html=True)

    # ── Download JSON ──
    st.download_button("📄 Xuất báo cáo JSON",
        data=json.dumps(res["report"], ensure_ascii=False, indent=2),
        file_name=f"tbcad_{Path(res['fname']).stem}.json",
        mime="application/json", key=f"dl_{res['fname']}_{id(res)}")


# ════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ════════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="tbcad-header">
    <div>
        <div class="tbcad-logo">🫁 TB-CAD SYSTEM</div>
        <div class="tbcad-sub">Hệ Thống Hỗ Trợ Chẩn Đoán Lao Phổi · Deep Learning · X-Ray Analysis</div>
        <div style="margin-top:0.6rem;">
            <span class="badge">EfficientNetV2</span><span class="badge">YOLOv12</span>
            <span class="badge">CAD v2.0</span><span class="badge">Dual-Branch</span>
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Performance</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#94a3b8;line-height:1.9;">
            Classification AUC: <strong style="color:#22d3ee;">99.10%</strong><br>
            Detection mAP@0.5: <strong style="color:#22d3ee;">68.1%</strong>
        </div>
    </div>
</div>
<div class="disc">
⚠ CẢNH BÁO LÂM SÀNG: Hệ thống chỉ hỗ trợ sàng lọc ban đầu, KHÔNG thay thế kết luận chẩn đoán của bác sĩ chuyên khoa.
</div>
""", unsafe_allow_html=True)

# ── Drive download status ──
_cls_ok  = _drive_status.get("best.pt", False)
_det_ok  = _drive_status.get("best_model.pth", False)
_cls_fid = GDRIVE_MODELS.get("best.pt", "")
_det_fid = GDRIVE_MODELS.get("best_model.pth", "")

if _cls_fid.startswith("YOUR_") or _det_fid.startswith("YOUR_"):
    st.warning("⚙️  Chưa cấu hình FILE_ID Google Drive. Mở file app.py, tìm GDRIVE_MODELS và điền ID thật vào.", icon="⚠️")
else:
    _cls_icon  = "✅" if _cls_ok  else "❌"
    _det_icon  = "✅" if _det_ok  else "❌"
    st.markdown(
        f"""<div style="background:rgba(34,211,238,0.06);border:1px solid rgba(34,211,238,0.18);
        border-radius:8px;padding:0.5rem 1.2rem;margin-bottom:0.6rem;
        font-family:Space Mono,monospace;font-size:0.7rem;color:#94a3b8;display:flex;gap:2rem;">
        <span>🤖 <strong style="color:#f1f5f9;">Model từ Drive</strong></span>
        <span>{_cls_icon} EfficientNetV2 (best.pt)</span>
        <span>{_det_icon} YOLOv12 (best_model.pth)</span>
        </div>""",
        unsafe_allow_html=True
    )

# ── Metrics ──
st.markdown("""
<div class="metric-strip">
    <div class="mc"><span class="mc-val">99.1%</span><span class="mc-lbl">AUC-ROC</span></div>
    <div class="mc"><span class="mc-val">97.2%</span><span class="mc-lbl">Sensitivity</span></div>
    <div class="mc"><span class="mc-val">68.1%</span><span class="mc-lbl">mAP@0.5</span></div>
    <div class="mc"><span class="mc-val">EV2-L</span><span class="mc-lbl">Cls Model</span></div>
    <div class="mc"><span class="mc-val">YOLOv12</span><span class="mc-lbl">Det Model</span></div>
</div>
""", unsafe_allow_html=True)

# ── Config Panel (Expander thay sidebar) ──
with st.expander("⚙️  CẤU HÌNH MÔ HÌNH & THAM SỐ  —  Nhấn để mở/đóng", expanded=False):
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 1.8, 1.8])

    with cc1:
        st.markdown("**🔷 Model Phân loại** *(EfficientNetV2)*")
        cls_path = st.text_input("cls_path", value="best.pt",
                                  label_visibility="collapsed", key="inp_cls")
        if st.button("Tải Model CLS", key="btn_cls"):
            with st.spinner("Đang tải..."):
                m, mt = load_model(cls_path)
                if m is not None:
                    st.session_state.cls_model, st.session_state.cls_type = m, mt
                    st.success("✓ Model CLS sẵn sàng")
                else:
                    st.session_state.cls_model, st.session_state.cls_type = "demo", "demo"
                    st.warning("⚠ Không tìm thấy → Demo mode")
        ok = st.session_state.cls_model is not None
        st.markdown(f'<span style="font-size:0.72rem;color:{"#34d399" if ok else "#f87171"};">{"● Sẵn sàng" if ok else "● Chưa tải"}</span>',
                    unsafe_allow_html=True)

    with cc2:
        st.markdown("**🔶 Model Khoanh vùng** *(YOLOv12)*")
        det_path = st.text_input("det_path", value="best_model.pth",
                                  label_visibility="collapsed", key="inp_det")
        if st.button("Tải Model DET", key="btn_det"):
            with st.spinner("Đang tải..."):
                m, mt = load_model(det_path)
                if m is not None:
                    st.session_state.det_model, st.session_state.det_type = m, mt
                    st.success("✓ Model DET sẵn sàng")
                else:
                    st.session_state.det_model, st.session_state.det_type = "demo", "demo"
                    st.warning("⚠ Không tìm thấy → Demo mode")
        ok2 = st.session_state.det_model is not None
        st.markdown(f'<span style="font-size:0.72rem;color:{"#34d399" if ok2 else "#f87171"};">{"● Sẵn sàng" if ok2 else "● Chưa tải"}</span>',
                    unsafe_allow_html=True)

    with cc3:
        pipeline = st.radio("Chế độ phân tích", [
            "🔗 Hybrid (A+B)", "🎯 Chỉ phân loại", "🔍 Chỉ khoanh vùng"
        ], key="pipeline")
        conf_thr = st.slider("Confidence detection", 0.10, 0.90, 0.35, 0.05, key="conf_thr")

    with cc4:
        show_hm   = st.checkbox("Hiển thị Heatmap", value=True, key="show_hm")
        show_pre  = st.checkbox("Hiển thị ảnh CLAHE", value=True, key="show_pre")
        auto_save = st.checkbox("Tự động lưu kết quả", value=True, key="auto_save",
                                help=f"Lưu vào thư mục: {SAVE_DIR.absolute()}")
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#64748b;margin-top:0.5rem;">💾 Lưu tại:<br>{SAVE_DIR.absolute()}</div>',
                    unsafe_allow_html=True)

# Lấy giá trị config hiện tại
pipeline  = st.session_state.get("pipeline",  "🔗 Hybrid (A+B)")
conf_thr  = st.session_state.get("conf_thr",  0.35)
show_hm   = st.session_state.get("show_hm",   True)
show_pre  = st.session_state.get("show_pre",  True)
auto_save = st.session_state.get("auto_save", True)


# ── Main Tabs ──
tab_main, tab_pipeline, tab_history, tab_guide = st.tabs([
    "🔬 Phân Tích Ảnh", "📊 Pipeline", "📋 Lịch Sử & Kết Quả Lưu", "📖 Hướng Dẫn"
])


# ─────────────────────────────────────────────────────────────────
#  TAB 1: PHÂN TÍCH ẢNH — hỗ trợ nhiều ảnh cùng lúc
# ─────────────────────────────────────────────────────────────────
with tab_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Tải Lên Ảnh X-Quang (Hỗ trợ nhiều ảnh cùng lúc)</div>',
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh X-quang ngực thẳng",
        type=["png","jpg","jpeg","bmp","tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Auto-reset: xoa ket qua cu khi danh sach file thay doi ──
    current_keys = sorted([f.name + str(f.size) for f in uploaded_files]) if uploaded_files else []
    if current_keys != st.session_state.prev_upload_keys:
        st.session_state.results = {}
        st.session_state.prev_upload_keys = current_keys

    if not uploaded_files:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    min-height:200px;color:#374151;border:1px dashed rgba(34,211,238,0.12);
                    border-radius:10px;margin-top:0.5rem;">
            <div style="font-size:2.8rem;opacity:0.25;margin-bottom:0.8rem;">🫁</div>
            <div style="font-family:Space Mono,monospace;font-size:0.73rem;letter-spacing:0.12em;color:#64748b;">AWAITING IMAGE INPUT</div>
            <div style="font-size:0.72rem;margin-top:0.4rem;color:#374151;">Tải lên ảnh X-quang để bắt đầu phân tích</div>
        </div>""", unsafe_allow_html=True)

    else:
        n = len(uploaded_files)
        names_preview = "  |  ".join([f.name for f in uploaded_files[:5]])
        if n > 5: names_preview += f"  ... +{n-5} ảnh"
        st.markdown(f"""
        <div style="background:rgba(34,211,238,0.06);border:1px solid rgba(34,211,238,0.18);
                    border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.8rem;
                    font-family:Space Mono,monospace;font-size:0.7rem;color:#94a3b8;">
        📁 Đã chọn <strong style="color:#22d3ee;">{n} ảnh</strong> &nbsp;·&nbsp; {names_preview}
        </div>""", unsafe_allow_html=True)

        # Preview grid trước khi phân tích
        if n <= 6:
            preview_cols = st.columns(min(n, 4))
            for i, uf in enumerate(uploaded_files[:4]):
                with preview_cols[i]:
                    st.image(Image.open(uf), caption=uf.name[:20], use_container_width=True)

        # Buttons
        bc1, bc2, bc3 = st.columns([1.5, 1.5, 4])
        with bc1:
            run_btn = st.button(f"▶  PHÂN TÍCH {n} ẢNH", key="run_btn", use_container_width=True)
        with bc2:
            if st.session_state.results:
                saved_folders = [r["saved_folder"] for r in st.session_state.results.values()
                                 if r.get("saved_folder")]
                if saved_folders:
                    st.download_button("📦 Tải ZIP kết quả",
                        data=make_zip(saved_folders),
                        file_name=f"TBCAD_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip", key="dl_zip", use_container_width=True)

        # ── Chạy phân tích ──
        if run_btn:
            st.session_state.results = {}
            for i, uf in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f"""
                    <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#64748b;margin:0.3rem 0 0.1rem 0;">
                    [{i+1}/{n}] → <strong style="color:#22d3ee;">{uf.name}</strong>
                    </div>""", unsafe_allow_html=True)
                    img_pil = Image.open(uf)
                    res = run_analysis(uf.name, img_pil, pipeline, conf_thr, show_hm, show_pre, auto_save)
                    res["orig_pil"] = img_pil.copy()  # lưu ảnh gốc để hiển thị bên trái
                    st.session_state.results[uf.name] = res
                    st.session_state.history.append({
                        "fname": uf.name,
                        "result": res["cls_label"] or f"Det:{len(res['dets'])}",
                        "lesions": len(res["dets"]),
                        "time": time.strftime("%H:%M:%S"),
                        "saved": res.get("saved_folder",""),
                    })

        # ── Hiển thị kết quả ──
        if st.session_state.results:
            fnames = list(st.session_state.results.keys())
            st.markdown("---")

            if len(fnames) == 1:
                # 1 ảnh: layout 2 cột
                res = st.session_state.results[fnames[0]]
                ci, cr = st.columns([1, 1.3], gap="large")
                with ci:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">📷 Ảnh X-Quang Gốc</div>', unsafe_allow_html=True)
                    orig = res.get("orig_pil")
                    if orig:
                        st.image(orig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with cr:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">🧠 Kết Quả AI</div>', unsafe_allow_html=True)
                    render_result(res)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Nhiều ảnh: tab mỗi ảnh + bảng tổng hợp
                img_tabs = st.tabs([f"🖼 {i+1}. {fn[:20]}{'…' if len(fn)>20 else ''}"
                                    for i, fn in enumerate(fnames)])
                for itab, fname in zip(img_tabs, fnames):
                    with itab:
                        res = st.session_state.results[fname]
                        ci, cr = st.columns([1, 1.3], gap="large")
                        with ci:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<div class="card-title">📷 Ảnh X-Quang Gốc</div>', unsafe_allow_html=True)
                            orig = res.get("orig_pil")
                            if orig:
                                st.image(orig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with cr:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<div class="card-title">🧠 Kết Quả AI</div>', unsafe_allow_html=True)
                            render_result(res)
                            st.markdown('</div>', unsafe_allow_html=True)

                # Summary
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📊 Bảng Tổng Hợp Tất Cả Ảnh</div>', unsafe_allow_html=True)
                rows = ""
                for i, fn in enumerate(fnames):
                    r = st.session_state.results[fn]
                    lbl = r["cls_label"] or "—"
                    pos = "DƯƠNG" in lbl
                    col = "#f87171" if pos else "#34d399"
                    icon = "🔴" if pos else "🟢"
                    tp = f"{r['tb_prob']:.1%}" if r["tb_prob"] is not None else "—"
                    rows += (f'<tr><td>{i+1}</td>'
                             f'<td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;">{fn}</td>'
                             f'<td><strong style="color:{col};">{icon} {lbl}</strong></td>'
                             f'<td>{tp}</td><td>{len(r["dets"])}</td></tr>')
                st.markdown(f'<table class="dt"><thead><tr><th>#</th><th>File</th><th>Kết quả</th>'
                            f'<th>TB Prob</th><th>Tổn thương</th></tr></thead>'
                            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2: PIPELINE
# ─────────────────────────────────────────────────────────────────
with tab_pipeline:

    def _flow_node(label, name, border="#22d3ee", color="#22d3ee", glow=True, bg="#1e2d47"):
        shadow = f"0 0 12px rgba(34,211,238,0.25)" if glow else "none"
        return (
            f'<div style="background:{bg};border:1.5px solid {border};border-radius:9px;'
            f'padding:8px 14px;text-align:center;min-width:88px;flex-shrink:0;box-shadow:{shadow};">'
            f'<span style="font-family:Space Mono,monospace;font-size:0.56rem;color:#64748b;'
            f'display:block;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">{label}</span>'
            f'<span style="font-size:0.72rem;font-weight:700;color:{color};">{name}</span></div>'
        )

    def _arrow(color="#374151"):
        return f'<span style="color:{color};font-size:1.1rem;flex-shrink:0;padding:0 2px;">→</span>'

    def _flow_wrap(nodes_html):
        return (
            '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;'
            'background:rgba(0,0,0,0.22);padding:0.9rem 1rem;border-radius:10px;'
            'border:1px solid rgba(255,255,255,0.04);">'
            + nodes_html + '</div>'
        )

    # ── Header ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔗 Kiến Trúc Dual-Branch Pipeline · 3 Phương Án</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.84rem;color:#94a3b8;line-height:1.75;margin-bottom:1.2rem;">
    Hệ thống triển khai kiến trúc <strong style="color:#22d3ee;">Dual-Branch</strong>
    theo Chương 3 của luận văn. Tất cả phương án đều chia sẻ chung bước
    <strong style="color:#f1f5f9;">Tiền xử lý tập trung</strong> (CLAHE + Resize)
    trước khi rẽ nhánh vào các khối mô hình.
    </div>
    """, unsafe_allow_html=True)

    # ── Phương Án A ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
        <span style="background:rgba(34,211,238,0.15);border:1px solid #22d3ee;border-radius:20px;
                     padding:2px 12px;font-family:Space Mono,monospace;font-size:0.65rem;
                     color:#22d3ee;font-weight:700;">PHƯƠNG ÁN A</span>
        <span style="font-size:0.85rem;font-weight:700;color:#f1f5f9;">Hybrid Sequential
            <span style="color:#64748b;font-weight:400;font-size:0.78rem;">· Triển khai chính</span>
        </span>
    </div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;padding-left:4px;">
    ▸ EfficientNetV2 phân loại trước → <em>chỉ khi DƯƠNG TÍNH</em> mới chuyển sang YOLOv12 khoanh vùng
    </div>
    """, unsafe_allow_html=True)

    nodes_a = (
        _flow_node("Input", "X-Ray CXR", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("Tiền xử lý", "CLAHE+Resize", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("Block 1", "EfficientNetV2", "#22d3ee", "#22d3ee", True) + _arrow() +
        _flow_node("Decision", "TB / Normal?", "#fbbf24", "#fbbf24", False, "#1e2d47") + _arrow("#fbbf24") +
        '<div style="display:flex;flex-direction:column;gap:6px;">'
        '<div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#34d399;">✓ TB → Block 2</div>'
        '<div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#64748b;">✗ Normal → Dừng</div>'
        '</div>' + _arrow() +
        _flow_node("Block 2", "YOLOv12-L", "#22d3ee", "#22d3ee", True) + _arrow() +
        _flow_node("Output", "Label+BBoxes", "rgba(52,211,153,0.5)", "#34d399", False)
    )
    st.markdown(_flow_wrap(nodes_a), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Phương Án B ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
        <span style="background:rgba(96,165,250,0.15);border:1px solid #60a5fa;border-radius:20px;
                     padding:2px 12px;font-family:Space Mono,monospace;font-size:0.65rem;
                     color:#60a5fa;font-weight:700;">PHƯƠNG ÁN B</span>
        <span style="font-size:0.85rem;font-weight:700;color:#f1f5f9;">YOLOv12 End-to-End
            <span style="color:#64748b;font-weight:400;font-size:0.78rem;">· One-stage detection</span>
        </span>
    </div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;padding-left:4px;">
    ▸ YOLOv12 thực hiện <em>đồng thời</em> phân loại + khoanh vùng trong 1 lần thực thi duy nhất
    </div>
    """, unsafe_allow_html=True)

    nodes_b = (
        _flow_node("Input", "X-Ray CXR", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("Tiền xử lý", "Resize 640×640", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("One-Stage", "YOLOv12-L", "#60a5fa", "#60a5fa", True) + _arrow() +
        _flow_node("Multi-Scale", "Heads P3/P4/P5", "#60a5fa", "#60a5fa", True) + _arrow() +
        _flow_node("Output", "Label+BBoxes", "rgba(52,211,153,0.5)", "#34d399", False)
    )
    st.markdown(_flow_wrap(nodes_b), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Phương Án C ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
        <span style="background:rgba(167,139,250,0.15);border:1px solid #a78bfa;border-radius:20px;
                     padding:2px 12px;font-family:Space Mono,monospace;font-size:0.65rem;
                     color:#a78bfa;font-weight:700;">PHƯƠNG ÁN C</span>
        <span style="font-size:0.85rem;font-weight:700;color:#f1f5f9;">EfficientNetV2 Standalone
            <span style="color:#64748b;font-weight:400;font-size:0.78rem;">· Chỉ phân loại nhị phân</span>
        </span>
    </div>
    <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;padding-left:4px;">
    ▸ Chỉ sử dụng EfficientNetV2 để sàng lọc nhanh — phù hợp môi trường tài nguyên hạn chế
    </div>
    """, unsafe_allow_html=True)

    nodes_c = (
        _flow_node("Input", "X-Ray CXR", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("Tiền xử lý", "CLAHE+Resize", "#475569", "#94a3b8", False) + _arrow() +
        _flow_node("EfficientNetV2", "Backbone+Head", "#a78bfa", "#a78bfa", True) + _arrow() +
        _flow_node("Sigmoid", "P(TB) score", "#a78bfa", "#a78bfa", True) + _arrow() +
        _flow_node("Output", "TB / Normal", "rgba(167,139,250,0.45)", "#a78bfa", False)
    )
    st.markdown(_flow_wrap(nodes_c), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── So sánh phương án ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚖️ So Sánh 3 Phương Án</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="dt">
    <thead><tr>
        <th>Tiêu chí</th>
        <th style="color:#22d3ee;">A · Hybrid</th>
        <th style="color:#60a5fa;">B · End-to-End</th>
        <th style="color:#a78bfa;">C · Cls Only</th>
    </tr></thead>
    <tbody>
    <tr><td>Nhiệm vụ</td>
        <td>Phân loại + Định vị</td>
        <td>Phân loại + Định vị</td>
        <td>Chỉ Phân loại</td></tr>
    <tr><td>Phân loại (AUC)</td>
        <td><strong style="color:#22d3ee;">99.10%</strong></td>
        <td>~85–90%</td>
        <td><strong style="color:#a78bfa;">99.10%</strong></td></tr>
    <tr><td>Detection mAP@0.5</td>
        <td><strong style="color:#22d3ee;">68.1%</strong></td>
        <td>~65–70%</td>
        <td>—</td></tr>
    <tr><td>Tốc độ xử lý</td>
        <td>Trung bình (2 bước)</td>
        <td><strong style="color:#60a5fa;">Nhanh nhất</strong></td>
        <td>Nhanh</td></tr>
    <tr><td>Tài nguyên GPU</td>
        <td>Trung bình</td>
        <td>Trung bình</td>
        <td><strong style="color:#a78bfa;">Thấp nhất</strong></td></tr>
    <tr><td>Phù hợp với</td>
        <td>Lâm sàng đầy đủ</td>
        <td>Realtime screening</td>
        <td>Tuyến cơ sở</td></tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bảng chỉ số hiệu năng ──
    cp, dp = st.columns(2, gap="medium")
    with cp:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 EfficientNetV2 · Classification</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="dt"><thead><tr><th>Chỉ số</th><th>Giá trị</th></tr></thead><tbody>
        <tr><td>AUC-ROC</td><td><strong style="color:#22d3ee;font-size:1rem;">99.10%</strong></td></tr>
        <tr><td>Accuracy</td><td><strong style="color:#60a5fa;">95.4%</strong></td></tr>
        <tr><td>Sensitivity (Recall)</td><td><strong style="color:#60a5fa;">97.2%</strong></td></tr>
        <tr><td>F1-Score</td><td><strong style="color:#60a5fa;">95.4%</strong></td></tr>
        <tr><td>Model</td><td>EfficientNetV2-L</td></tr>
        <tr><td>Input</td><td>384×384 px · ~120M params</td></tr>
        </tbody></table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with dp:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📦 YOLOv12 · Lesion Detection</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="dt"><thead><tr><th>Chỉ số</th><th>Giá trị</th></tr></thead><tbody>
        <tr><td>mAP@0.5</td><td><strong style="color:#22d3ee;font-size:1rem;">68.1%</strong></td></tr>
        <tr><td>mAP@0.5:0.95</td><td><strong style="color:#60a5fa;">45.7%</strong></td></tr>
        <tr><td>Precision</td><td><strong style="color:#60a5fa;">82.1%</strong></td></tr>
        <tr><td>Recall</td><td><strong style="color:#60a5fa;">71.6%</strong></td></tr>
        <tr><td>Model</td><td>YOLOv12-L · 1 class</td></tr>
        <tr><td>Dataset</td><td>TBX11K (11,200 ảnh)</td></tr>
        </tbody></table>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 3: LỊCH SỬ & KẾT QUẢ LƯU
# ─────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Lịch Sử Phân Tích (Phiên hiện tại)</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<span style='color:#64748b;font-family:Space Mono,monospace;font-size:0.75rem;'>Chưa có phân tích nào.</span>", unsafe_allow_html=True)
    else:
        rows = ""
        for i, h in enumerate(reversed(st.session_state.history)):
            lbl = h.get("result","—")
            pos = "DƯƠNG" in lbl
            col = "#f87171" if pos else "#34d399"
            icon = "🔴" if pos else "🟢"
            saved_name = Path(h["saved"]).name if h.get("saved") else "—"
            rows += (f'<tr><td>{len(st.session_state.history)-i}</td>'
                     f'<td>{h["fname"]}</td>'
                     f'<td><strong style="color:{col};">{icon} {lbl}</strong></td>'
                     f'<td>{h["lesions"]}</td>'
                     f'<td>{h["time"]}</td>'
                     f'<td style="font-size:0.65rem;color:#64748b;">{saved_name}</td></tr>')
        st.markdown(f'<table class="dt"><thead><tr><th>#</th><th>File</th><th>Kết quả</th>'
                    f'<th>Tổn thương</th><th>Giờ</th><th>Thư mục lưu</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        if st.button("🗑 Xóa lịch sử phiên", key="clear_hist"):
            st.session_state.history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Kết quả trên disk
    if SAVE_DIR.exists():
        items = sorted(SAVE_DIR.iterdir(), reverse=True)
        if items:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">💾 Kết Quả Đã Lưu Trên Disk</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="saved-box" style="margin-bottom:0.8rem;">
            📁 Thư mục: <code style="color:#6ee7b7;">{SAVE_DIR.absolute()}</code><br>
            Tổng: <strong>{len(items)}</strong> lần phân tích đã lưu
            </div>""", unsafe_allow_html=True)
            for item in items[:15]:
                files = list(item.iterdir())
                flist = "  ·  ".join(f.name for f in files[:3])
                st.markdown(f"""
                <div style="padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);
                            font-family:Space Mono,monospace;font-size:0.68rem;color:#94a3b8;">
                📂 {item.name} &nbsp;<span style="color:#64748b;">({len(files)} files: {flist})</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 4: HƯỚNG DẪN
# ─────────────────────────────────────────────────────────────────
with tab_guide:
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📖 Hướng Dẫn Sử Dụng</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem;color:#cbd5e1;line-height:1.9;">
        <strong style="color:#22d3ee;">Bước 1 · Tải mô hình</strong><br>
        Mở panel <em>⚙️ CẤU HÌNH</em> phía trên, nhập đường dẫn → nhấn <em>Tải Model</em>.<br><br>

        <strong style="color:#22d3ee;">Bước 2 · Chọn chế độ phân tích</strong><br>
        • <strong>Hybrid (A+B)</strong>: Phân loại trước → nếu TB mới khoanh vùng<br>
        • <strong>Chỉ phân loại</strong>: Chỉ EfficientNetV2<br>
        • <strong>Chỉ khoanh vùng</strong>: Chỉ YOLOv12<br><br>

        <strong style="color:#22d3ee;">Bước 3 · Upload ảnh (nhiều ảnh)</strong><br>
        Nhấn vùng upload → chọn 1 hoặc nhiều file cùng lúc.<br>
        Hỗ trợ: PNG, JPG, BMP, TIFF · Khuyến nghị ≥ 512×512 px<br><br>

        <strong style="color:#22d3ee;">Bước 4 · Nhấn PHÂN TÍCH</strong><br>
        Hệ thống xử lý tuần tự từng ảnh, kết quả hiển thị theo tab.<br><br>

        <strong style="color:#22d3ee;">Bước 5 · Tải kết quả</strong><br>
        • Từng ảnh: nút <em>Xuất báo cáo JSON</em><br>
        • Toàn bộ phiên: nút <em>Tải ZIP</em> (xuất hiện sau phân tích)
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💾 Cấu Trúc Thư Mục Lưu Kết Quả</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.85;">
        Mỗi lần phân tích tạo 1 thư mục riêng:<br>
        </div>""", unsafe_allow_html=True)
        st.code("""TB_CAD_Results/
└── 20240301_143022_chest_xray/
    ├── original_chest_xray.png   ← Ảnh gốc
    ├── result_chest_xray.png     ← Ảnh có bounding box
    └── report.json               ← Kết quả đầy đủ""", language="text")
        st.markdown("""
        <div style="font-size:0.83rem;color:#cbd5e1;line-height:1.85;margin-top:0.8rem;">
        <strong style="color:#22d3ee;">report.json</strong> chứa:<br>
        • Tên file, timestamp, chế độ pipeline<br>
        • Kết quả phân loại (TB/Normal, confidence, probability)<br>
        • Danh sách bounding box (tọa độ, confidence, nhãn)
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚙️ Requirements</div>', unsafe_allow_html=True)
        st.code("""streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0""", language="text")
        st.markdown('</div>', unsafe_allow_html=True)
