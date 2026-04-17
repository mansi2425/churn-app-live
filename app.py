import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, encoders, features

def preprocess_input(raw, encoders):
    df = pd.DataFrame([raw])
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = 0
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    service_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["ChargePerService"] = df["MonthlyCharges"] / (df[service_cols].sum(axis=1) + 1)
    tenure_val = df["tenure"].values[0]
    if tenure_val <= 12:
        df["TenureGroup"] = 0
    elif tenure_val <= 24:
        df["TenureGroup"] = 1
    elif tenure_val <= 48:
        df["TenureGroup"] = 2
    else:
        df["TenureGroup"] = 3
    return df

def main():
    with st.sidebar:
        st.title("📡 Telco Churn Analytics")
        st.caption("Powered by Random Forest Classifier")
        st.divider()
        st.info("Fill in the customer details and click Predict Churn.")
        st.divider()
        st.markdown("**Artifact Status:**")
        for f in ["churn_model.pkl","scaler.pkl","label_encoders.pkl","feature_columns.pkl"]:
            path = os.path.join(ARTIFACTS_DIR, f)
            icon = "✅" if os.path.exists(path) else "❌"
            st.markdown(f"{icon} `{f}`")

    st.title("📡 Customer Churn Prediction System")
    st.markdown("Predict whether a telecom customer is likely to **churn** based on their profile.")
    st.divider()

    for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            st.error(f"Artifact not found: `{path}`. Please run train_model.py first.")
            st.stop()

    model, scaler, encoders, feature_columns = load_artifacts()

    st.subheader("🧾 Enter Customer Details")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**🏦 Account Details**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(monthly_charges * tenure))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col3:
            st.markdown("**📶 Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_svc = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.divider()
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        senior_val = 1 if senior_citizen == "Yes" else 0
        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior_val,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_svc,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Running prediction..."):
            input_df = preprocess_input(raw_input, encoders)
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]
            num_cols = [c for c in input_df.columns if input_df[c].dtype in ["float64","int64","int32"]]
            input_scaled = input_df.copy()
            try:
                input_scaled[num_cols] = scaler.transform(input_df[num_cols])
            except Exception:
                pass
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        churn_prob = probability[1] * 100
        no_churn_prob = probability[0] * 100

        res1, res2, res3 = st.columns([1.2, 1, 1])

        with res1:
            if prediction == 1:
                st.error("### ⚠️ HIGH CHURN RISK\nThis customer is **likely to churn.**")
            else:
                st.success("### ✅ LOW CHURN RISK\nThis customer is **likely to stay.**")

        with res2:
            st.metric("🔴 Churn Probability", f"{churn_prob:.1f}%")
            st.metric("🟢 Retention Probability", f"{no_churn_prob:.1f}%")

        with res3:
            st.markdown("**Risk Gauge**")
            st.progress(int(churn_prob), text=f"Churn Risk: {churn_prob:.1f}%")
            if churn_prob < 30:
                st.markdown("**Status: 🟢 Low Risk**")
            elif 30 <= churn_prob < 60:
                st.markdown("**Status: 🟡 Medium Risk**")
            else:
                st.markdown("**Status: 🔴 High Risk**")

        st.divider()
        st.subheader("💡 Retention Recommendations")
        if prediction == 1:
            recs = []
            if contract == "Month-to-month":
                recs.append("🔄 Offer a discounted 1-year or 2-year contract.")
            if internet_svc == "Fiber optic" and online_security == "No":
                recs.append("🔒 Bundle Online Security & Tech Support at a promotional rate.")
            if payment_method == "Electronic check":
                recs.append("💳 Incentivise auto-payment with a monthly discount.")
            if tenure < 12:
                recs.append("🎁 Trigger a loyalty reward for first-year customers.")
            if not recs:
                recs.append("📞 Schedule a proactive customer success call.")
            for r in recs:
                st.markdown(f"- {r}")
        else:
            st.success("Customer has a strong retention profile. Consider a loyalty reward to solidify the relationship.")

        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame(raw_input.items(), columns=["Feature", "Value"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()