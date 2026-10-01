import os
import pandas as pd
import streamlit as st
import requests


API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
    )


st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Predictive Maintenance Predictor")

st.write(
    "Enter the machine operating conditions and click "
    "**Predict failure risk**."
)

with st.form("prediction_form"):
    load_type = st.selectbox(
        "Load type",
        options=["L", "M", "H"],
        help="L = Low, M = Medium, H = High quality",
    )

    col1, col2 = st.columns(2)

    with col1:
        air_temperature_k = st.number_input(
            "Air temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=298.0,
            step=0.1,
        )

        rotational_speed_rpm = st.number_input(
            "Rotational speed [rpm]",
            min_value=0,
            value=1500,
            step=10,
        )

        tool_wear_min = st.number_input(
            "Tool wear [min]",
            min_value=0,
            value=100,
            step=1,
        )

    with col2:
        process_temperature_k = st.number_input(
            "Process temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=308.0,
            step=0.1,
        )

        torque_nm = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            value=40.0,
            step=0.1,
        )

    submitted = st.form_submit_button(
        "Predict failure risk",
        type="primary",
        use_container_width=True,
    )

    submitted_fmode = st.form_submit_button(
        "Predict process failure mode",
        type="primary",
        use_container_width=True
    )


if submitted:
    payload = {
        "load_type": load_type,
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    try:
        with st.spinner("Making prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        # confidence = result.get("confidence")

        if prediction in [1, "1", "Failure", "Machine failure"]:
            st.error("⚠️ Machine failure risk detected")
        else:
            st.success("✅ No machine failure detected")

        # if confidence is not None:
        #     st.metric(
        #         "Model confidence",
        #         f"{confidence:.1%}",
        #     )

        with st.expander("Input data"):
            st.dataframe(
                pd.DataFrame([payload]),
                use_container_width=True,
            )

        with st.expander("API response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the backend is running on port 8000."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                "Prediction request failed.",
            )
        except ValueError:
            detail = "Prediction request failed."

        st.error(detail)

    except requests.exceptions.RequestException as exc:
        st.error(f"Request error: {exc}")



if submitted_fmode:
    payload = {
        "load_type": load_type,
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    try:
        with st.spinner("Making prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        # confidence = result.get("confidence")

        if prediction in [1, "1", "Failure", "Machine failure"]:
            st.error("⚠️ Machine failure risk detected")
        else:
            st.success("✅ No machine failure detected")

        # if confidence is not None:
        #     st.metric(
        #         "Model confidence",
        #         f"{confidence:.1%}",
        #     )

        with st.expander("Input data"):
            st.dataframe(
                pd.DataFrame([payload]),
                use_container_width=True,
            )

        with st.expander("API response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Make sure the backend is running on port 8000."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                "Prediction request failed.",
            )
        except ValueError:
            detail = "Prediction request failed."

        st.error(detail)

    except requests.exceptions.RequestException as exc:
        st.error(f"Request error: {exc}")