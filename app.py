# Chiller Failure Prediction -- Streamlit Demo (Phase 4)
# ======================================================
# HBO-ICT Machine Learning semester project -- AMD methodology
#
# Install:  pip install streamlit xgboost shap pandas numpy matplotlib scikit-learn
# Run:      streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chiller FDD", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "xgb_model.json"
SCALER_PATH = BASE / "data" / "prepared" / "scaler.pkl"
FEATURES_PATH = BASE / "data" / "prepared" / "feature_names.csv"
X_TRAIN_PATH = BASE / "data" / "prepared" / "X_train_raw.npy"
Y_MULTI_PATH = BASE / "data" / "prepared" / "y_train_multi.npy"

SENSOR_DESCRIPTIONS = {
    "TEI": "Evaporator water inlet temperature (\u00b0F)",
    "TEO": "Evaporator water outlet temperature (\u00b0F)",
    "TCI": "Condenser water inlet temperature (\u00b0F)",
    "TCO": "Condenser water outlet temperature (\u00b0F)",
    "kW": "Compressor power consumption (kW)",
    "TEA": "Evaporator approach temperature (\u00b0F)",
    "TCA": "Condenser approach temperature (\u00b0F)",
    "TRE": "Refrigerant evaporating temperature (\u00b0F)",
    "TRC": "Refrigerant condensing temperature (\u00b0F)",
    "TRC_sub": "Refrigerant condenser subcooling (\u00b0F)",
    "T_suc": "Compressor suction temperature (\u00b0F)",
    "Tsh_suc": "Suction superheat (\u00b0F)",
    "TR_dis": "Compressor discharge temperature (\u00b0F)",
    "Tsh_dis": "Discharge superheat (\u00b0F)",
    "TO_sump": "Oil sump temperature (\u00b0F)",
    "PO_net": "Net oil pressure (PSI)",
}

PRESET_CLASSES = {
    "Normal operation": 1,
    "Condenser Fouling (CF)": 2,
    "Refrigerant Leak (RL)": 5,
    "Refrigerant Overcharge (RO)": 6,
}

# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
def _check_file(path: Path) -> None:
    if not path.exists():
        st.error(f"Required file not found: {path.name}. "
                 "Make sure you've run Phase 2 and Phase 3 notebooks first.")
        st.stop()


@st.cache_resource
def load_model():
    _check_file(MODEL_PATH)
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_resource
def load_scaler():
    _check_file(SCALER_PATH)
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_feature_names():
    _check_file(FEATURES_PATH)
    return pd.read_csv(FEATURES_PATH)["feature"].tolist()


@st.cache_data
def load_presets_and_ranges():
    """Pick a representative real sample per fault class (closest to median).

    Using a real sample instead of the raw median guarantees the preset is an
    actual operating point the model has seen, avoiding the 'no-man's-land'
    problem where the median across mixed conditions falls between modes.
    """
    _check_file(X_TRAIN_PATH)
    _check_file(Y_MULTI_PATH)

    X = np.load(str(X_TRAIN_PATH))
    y = np.load(str(Y_MULTI_PATH))
    features = load_feature_names()

    # Slider ranges: min/max with 10 % padding
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    padding = (col_max - col_min) * 0.10
    ranges = {
        f: (float(round(col_min[i] - padding[i], 2)),
            float(round(col_max[i] + padding[i], 2)))
        for i, f in enumerate(features)
    }

    # Representative real sample per class (closest to that class's median)
    presets = {}
    for name, cls in PRESET_CLASSES.items():
        class_samples = X[y == cls]
        class_median = np.median(class_samples, axis=0)
        distances = np.linalg.norm(class_samples - class_median, axis=1)
        representative = class_samples[np.argmin(distances)]
        presets[name] = {
            f: float(round(representative[i], 2)) for i, f in enumerate(features)
        }

    return presets, ranges


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
model = load_model()
scaler = load_scaler()
explainer = load_explainer(model)
features = load_feature_names()
presets, ranges = load_presets_and_ranges()


# ---------------------------------------------------------------------------
# Sidebar -- sensor inputs
# ---------------------------------------------------------------------------
st.sidebar.title("Chiller Sensor Inputs")

preset_options = list(PRESET_CLASSES.keys()) + ["Custom"]
preset = st.sidebar.selectbox("Preset scenario", preset_options, index=0)

defaults = presets["Normal operation"] if preset == "Custom" else presets[preset]

sensor_values = {}
for f in features:
    lo, hi = ranges[f]
    sensor_values[f] = st.sidebar.number_input(
        f,
        min_value=lo,
        max_value=hi,
        value=defaults[f],
        step=0.1,
        format="%.2f",
        help=SENSOR_DESCRIPTIONS[f],
        key=f"input_{f}_{preset}",  # reset when preset changes
    )

predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Chiller Failure Prediction System")
st.caption("ML-powered predictive maintenance with SHAP explainability")

if not predict_clicked:
    st.info("Adjust the sensor values in the sidebar and click **Predict** to see the result.")
    st.stop()

# --- Run prediction pipeline ---
raw = np.array([[sensor_values[f] for f in features]])
scaled = scaler.transform(raw)
proba = model.predict_proba(scaled)[0]
fault_prob = float(proba[1])
is_fault = fault_prob >= 0.5

# Prediction banner + probability side by side
banner_col, prob_col = st.columns([2, 1])
with banner_col:
    if is_fault:
        st.error("## \u26a0\ufe0f FAULT DETECTED")
    else:
        st.success("## \u2705 NORMAL OPERATION")
with prob_col:
    st.metric("Fault probability", f"{fault_prob:.1%}")
    st.progress(min(fault_prob, 1.0))

st.divider()

# --- SHAP explanation (raw values shown) ---
st.subheader("SHAP Explanation")
st.caption(
    "Each bar shows how a sensor reading pushed the prediction toward Fault (red) "
    "or Normal (blue), starting from the model's average prediction."
)

shap_values = explainer(scaled)
sv = shap_values.values[0]
base = float(shap_values.base_values[0])

# Smaller plot, centred in a column to reduce on-screen size
plot_col = st.columns([1, 3, 1])[1]
with plot_col:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=base,
            data=raw[0],            # show raw sensor values, not scaled
            feature_names=features,
        ),
        show=False,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()

# --- Plain-English summary ---
st.subheader("Plain-English Summary")

abs_sv = np.abs(sv)
top_idx = np.argsort(abs_sv)[::-1][:3]

if is_fault:
    intro = (
        f"The model flagged this reading as a **fault** "
        f"(probability {fault_prob:.1%}). The top contributing sensors are:"
    )
else:
    intro = (
        f"The model considers this reading **normal** "
        f"(fault probability {fault_prob:.1%}). The most influential sensors are:"
    )

lines = []
for idx in top_idx:
    fname = features[idx]
    raw_val = sensor_values[fname]
    shap_val = sv[idx]
    desc = SENSOR_DESCRIPTIONS[fname]
    direction = "pushing toward **Fault**" if shap_val > 0 else "pushing toward **Normal**"
    lines.append(
        f"- **{fname}** = {raw_val:.1f} \u2014 {desc} \u2014 "
        f"SHAP {shap_val:+.3f}, {direction}."
    )

st.markdown(intro)
st.markdown("\n".join(lines))

# --- Collapsible input table ---
with st.expander("View all sensor inputs"):
    st.dataframe(
        pd.DataFrame({
            "Sensor": features,
            "Value": [sensor_values[f] for f in features],
            "Description": [SENSOR_DESCRIPTIONS[f] for f in features],
        }),
        use_container_width=True,
        hide_index=True,
    )