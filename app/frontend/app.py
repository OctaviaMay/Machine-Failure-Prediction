"""
Streamlit frontend for the Machine-Failure Prediction API (binary + multi-class).
"""
import os
from dotenv import load_dotenv

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

TIMEOUT = 15

st.set_page_config(page_title="Predicting Machine Failures", page_icon="⚙️", layout="wide")

# CSV column name -> API field name (accepts raw AI4I headers or API names)
CSV_TO_API = {
    "Type": "load_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}
API_FIELDS = list(CSV_TO_API.values())


# --------------------------------------------------------------------------- #
# API client — one place for HTTP + error handling
# --------------------------------------------------------------------------- #
class APIError(Exception):
    pass


def call_api(method: str, path: str, **kwargs) -> dict:
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(f"Cannot connect to FastAPI at {API_URL}. Is the backend running?")
    except requests.exceptions.Timeout:
        raise APIError("The request timed out.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # pydantic 422 validation errors
            detail = "; ".join(f"{'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail)
        raise APIError(f"API error {r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def get_metadata() -> dict | None:
    try:
        return call_api("GET", "/metadata")
    except APIError:
        return None


def get_health() -> dict | None:
    try:
        return call_api("GET", "/health")
    except APIError:
        return None


# --------------------------------------------------------------------------- #
# Renderers — each consumes the shared PredictionOutput envelope
# --------------------------------------------------------------------------- #
def render_warnings(result: dict):
    for w in result.get("warnings", []):
        st.warning(w, icon="⚠️")


def render_binary(result: dict):
    p_fail = result["failure_probability"]
    thr = result["threshold"]

    if result["is_failure"]:
        st.error(f"### ⚠️ {result['name']}")
    else:
        st.success(f"### ✅ {result['name']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure probability", f"{p_fail:.1%}")
    c2.metric("Decision threshold", f"{thr:.0%}")
    c3.metric("Margin to threshold", f"{p_fail - thr:+.1%}")

    st.progress(min(max(p_fail, 0.0), 1.0), text=f"Risk {p_fail:.1%} (alarm at ≥ {thr:.0%})")
    st.caption(f"Model: {result['model_name']}")
    render_warnings(result)


def render_multiclass(result: dict):
    probs = pd.DataFrame(result["probabilities"])

    if result["is_failure"]:
        st.error(f"### ⚠️ Predicted mode: {result['code']} — {result['name']}")
    else:
        st.success("### ✅ No Failure")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top class confidence", f"{result['confidence']:.1%}")
    c2.metric("P(any failure)", f"{result['failure_probability']:.1%}")
    # Most likely *failure* mode, even when the top class is "No Failure"
    fail_only = probs[probs["class_id"] != "0"]
    if not fail_only.empty:
        top_fail = fail_only.iloc[0]
        c3.metric("Most likely failure mode", top_fail["code"], f"{top_fail['probability']:.1%}", delta_color="off")

    chart_df = probs.sort_values("probability")
    chart_df["label"] = chart_df["code"] + " · " + chart_df["name"]
    chart_df["predicted"] = chart_df["class_id"] == result["class_id"]
    fig = px.bar(
        chart_df,
        x="probability",
        y="label",
        orientation="h",
        color="predicted",
        color_discrete_map={True: "#d62728", False: "#9aa5b1"},
        text=chart_df["probability"].map(lambda p: f"{p:.1%}"),
        range_x=[0, 1],
    )
    fig.update_layout(
        showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Model: {result['model_name']}")
    render_warnings(result)


def render_combined(result: dict):
    if result["agreement"]:
        st.info(f"**Models agree.** {result['summary']}")
    else:
        st.warning(f"**Models disagree.** {result['summary']}")
    left, right = st.columns(2)
    with left:
        st.subheader("Binary — will it fail?")
        render_binary(result["binary"])
    with right:
        st.subheader("Multi-class — how will it fail?")
        render_multiclass(result["multiclass"])


RENDERERS = {"binary": render_binary, "multiclass": render_multiclass, "combined": render_combined}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Modelling mode",
        options=["binary", "multiclass", "combined"],
        format_func={
            "binary": "Binary — failure / no failure",
            "multiclass": "Multi-class — failure mode",
            "combined": "Both (side by side)",
        }.get,
        index=2,
    )

    st.divider()
    health = get_health()
    if health is None:
        st.error(f"Backend offline\n\n{API_URL}")
    else:
        for m, ok in health["models_loaded"].items():
            st.write(f"{'🟢' if ok else '🔴'} {m} model")
        for m, e in health.get("errors", {}).items():
            st.caption(f"{m}: {e}")

    meta = get_metadata()
    if meta:
        with st.expander("Model metadata"):
            st.json(meta)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("⚙️ Predicting Machine Failures")
tab_single = st.tabs(["Single prediction"])[0]

# ---- Single ----------------------------------------------------------------
with tab_single:
    with st.form("prediction_form"):
        load_type = st.selectbox(
            "Load type", ["L", "M", "H"], help="L = Low, M = Medium, H = High quality variant"
        )
        c1, c2, c3 = st.columns(3)
        air = c1.number_input("Air temperature [K]", 250.0, 350.0, 298.0, 0.1)
        proc = c2.number_input("Process temperature [K]", 250.0, 400.0, 308.0, 0.1)
        rpm = c3.number_input("Rotational speed [rpm]", 1, 5000, 1500, 10)
        torque = c1.number_input("Torque [Nm]", 0.0, 150.0, 40.0, 0.1)
        wear = c2.number_input("Tool wear [min]", 0, 400, 100, 1)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        payload = {
            "load_type": load_type,
            "air_temperature_k": air,
            "process_temperature_k": proc,
            "rotational_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": wear,
        }
        try:
            with st.spinner("Predicting..."):
                st.session_state["single"] = {
                    "mode": mode,
                    "payload": payload,
                    "result": call_api("POST", f"/predict/{mode}", json=payload),
                }
        except APIError as e:
            st.session_state.pop("single", None)
            st.error(str(e))

    # Render from session_state so results survive widget reruns
    last = st.session_state.get("single")
    if last:
        if last["mode"] != mode:
            st.caption(f"Showing last **{last['mode']}** result — click Predict to refresh for **{mode}**.")
        RENDERERS[last["mode"]](last["result"])
        with st.expander("Input data"):
            st.dataframe(pd.DataFrame([last["payload"]]), width="stretch", hide_index=True)
        with st.expander("Raw API response"):
            st.json(last["result"])