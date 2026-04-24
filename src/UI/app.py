import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="IT Asset Scanner",
    page_icon="📷",
    layout="wide"
)

st.title("📷 IT Asset Scanner")
st.caption("Photograph any hardware — model and category appear instantly.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    worker_id = st.text_input("Worker ID", value="worker-01")
    confidence_threshold = st.slider("Flag below confidence", 0.0, 1.0, 0.75)
    st.divider()
    if st.button("📋 View Scan History"):
        resp = requests.get(f"{API_URL}/scans")
        if resp.status_code == 200:
            scans = resp.json()["scans"]
            st.write(f"Total scans: {len(scans)}")
            for s in scans[:5]:
                st.write(f"• {s['detected']} — {s['clip_score']:.2f}")

# Main
col1, col2 = st.columns(2)

with col1:
    source = st.radio("Input Source", ["📁 Upload File", "📸 Camera"])
    if source == "📁 Upload File":
        file = st.file_uploader("Upload hardware image", type=["jpg","jpeg","png"])
    else:
        file = st.camera_input("Take a photo")

with col2:
    if file:
        st.image(file, caption="Input Image", use_column_width=True)

if file and st.button("🔍 Scan Asset", type="primary", use_container_width=True):
    with st.spinner("Analysing hardware..."):
        resp = requests.post(
            f"{API_URL}/scan",
            files={"file": file.getvalue()},
            params={"worker_id": worker_id}
        )

    if resp.status_code == 200:
        data = resp.json()
        st.success(f"Found {data['items_found']} item(s)")

        for item in data["results"]:
            top = item.get("top_match")
            score = top["score"] if top else 0
            category = top["metadata"].get("category") if top else "Unknown"
            flagged = score < confidence_threshold

            with st.expander(
                f"{'⚠️' if flagged else '✅'} {item['detected_class']} — CLIP Score: {score:.2f}",
                expanded=True
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Detected", item["detected_class"])
                c2.metric("YOLO Confidence", f"{item['yolo_confidence']:.0%}")
                c3.metric("CLIP Score", f"{score:.3f}")

                st.write(f"**Category:** {category}")
                st.write(f"**Asset ID:** {top['metadata'].get('asset_id', '—') if top else '—'}")

                if flagged:
                    st.warning("⚠️ Low confidence — please verify manually")
                else:
                    st.success("✅ High confidence match")
    else:
        st.error(f"API Error: {resp.status_code}")