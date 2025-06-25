import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load combined model and scaler
combined = joblib.load("combined_model.pkl")
model = combined['model']
scaler = combined['scaler']

# Judul dan layout
st.set_page_config(page_title="Cyber Attack Prediction", layout="wide")
st.title("🔐 Cyber Attack Prediction")
st.markdown("Analyze and predict whether a cyber threat is Normal or an Attack using machine learning.")
st.write("---")

# Layout 2 kolom
col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ Threat Parameters")
    flow_duration = st.number_input("Flow Duration (ms) [0 - 1,000,000]", min_value=0.0, max_value=1000000.0, help="e.g. 1520")
    packet_size = st.number_input("Packet Size (bytes) [32 - 1,514]", min_value=32.0, max_value=1514.0, help="e.g. 64")
    flow_bytes = st.number_input("Flow Bytes/s [0 - 950,000]", min_value=0.0, max_value=950000.0, help="e.g. 1024")
    flow_packets = st.number_input("Flow Packets/s [0 - 2,000]", min_value=0.0, max_value=2000.0, help="e.g. 50")
    cpu_util = st.number_input("CPU Utilization (%) [0 - 100]", min_value=0.0, max_value=100.0, help="e.g. 45")
    mem_util = st.number_input("Memory Utilization (%) [0 - 100]", min_value=0.0, max_value=100.0, help="e.g. 65")
    anomaly_index = st.number_input("Anomaly Severity Index [0 - 100]", min_value=0.0, max_value=100.0, help="e.g. 3.4")
    normalized_flow = st.number_input("Normalized Packet Flow [0 - 1.0]", min_value=0.0, max_value=1.0, help="e.g. 0.85")
    attack_severity = st.selectbox("Attack Severity (as input feature)", [0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])

    if st.button("⚡ Predict Cyber Attack"):
        input_data = np.array([[flow_duration, packet_size, flow_bytes, flow_packets,
                                cpu_util, mem_util, anomaly_index,
                                normalized_flow, attack_severity]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        with col2:
            st.header("📊 Threat Analysis")
            st.subheader("Prediction Result")
            st.success(f"Predicted Threat Label: **{['Normal', 'Attack'][prediction]}**")

            if prediction == 1:
                st.warning("⚠️ Recommended Actions:")
                st.markdown("""
                - **Disconnect immediately** from suspicious network sources or systems.
                - **Check system logs** for unusual activity matching the time of attack.
                - **Run antivirus or intrusion detection scans** on the system.
                - **Notify your IT/security team** to perform deeper investigation.
                - **Isolate affected devices** to prevent spread if attack seems severe.
                - **Ensure backups** are up to date in case data recovery is needed.
                """)

            st.subheader("Confidence Metrics")
            fig, ax = plt.subplots()
            sns.barplot(x=["Normal", "Attack"], y=prediction_proba[:2], palette="coolwarm", ax=ax)
            ax.set_ylabel("Probability")
            ax.set_ylim([0, 1])
            st.pyplot(fig)

            st.markdown("**Confidence metrics reflect likelihood of Normal vs Attack based on XGBoost model.**")