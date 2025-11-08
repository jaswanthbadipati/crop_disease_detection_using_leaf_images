import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from huggingface_hub import hf_hub_download

# ===== Page Setup =====
st.set_page_config(page_title="CropGuard AI", layout="centered")

# ===== Hugging Face Model Files =====
REPO_ID = "jaswanth5472/crop-disease-detection-model"
MODEL_FILENAME = "plant_disease_prediction_model.h5"
CLASS_FILENAME = "class_indices.json"

# Download model + classes from Hugging Face cache if not exists
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
class_path = hf_hub_download(repo_id=REPO_ID, filename=CLASS_FILENAME)

# Load model & class indices
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_path))
inv_class_indices = {int(k): v for k, v in class_indices.items()}

# ===== Remedies =====
remedies = {

    # APPLE
    "Apple___Apple_scab": """
• Apply fungicides containing Mancozeb, Captan, or Chlorothalonil at early leaf stages.
• Remove and destroy fallen leaves to prevent reinfection.
• Improve air circulation by pruning dense branches.
• Water early in the morning to allow leaves to dry.
""",

    "Apple___Black_rot": """
• Remove infected fruits and prune infected twigs 10–12 inches below damage.
• Disinfect pruning tools with 10% bleach.
• Apply copper-based fungicides at bloom and petal fall stage.
• Avoid overhead watering and maintain orchard hygiene.
""",

    "Apple___Cedar_apple_rust": """
• Remove nearby juniper/cedar trees if possible (main source of spores).
• Apply Myclobutanil or Sulfur-based fungicides during spring.
• Improve spacing between plants to support airflow.
""",

    "Apple___healthy": "✅ Your apple leaf is healthy! Maintain regular watering and pruning.",

    # BLUEBERRY
    "Blueberry___healthy": "✅ No disease detected. Maintain mulch, drip irrigation, and remove weeds regularly.",

    # CHERRY
    "Cherry_(including_sour)___Powdery_mildew": """
• Spray neem oil, sulfur dust, or potassium bicarbonate solution weekly.
• Avoid excess nitrogen fertilizers.
• Prune overcrowded branches to increase ventilation.
""",

    "Cherry_(including_sour)___healthy": "✅ Plant is healthy. Keep soil slightly moist and ensure sunlight exposure.",

    # CORN / MAIZE
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": """
• Rotate crops every 2–3 seasons.
• Use resistant seed varieties.
• Apply fungicides like Propiconazole only if disease severity increases.
""",

    "Corn_(maize)___Common_rust_": """
• Use rust-resistant hybrid seeds.
• Apply Mancozeb or Propiconazole if rust covers >10% leaf area.
""",

    "Corn_(maize)___Northern_Leaf_Blight": """
• Remove infected leaves and crop residues.
• Avoid late-season irrigation.
• Apply Azoxystrobin or Trifloxystrobin-based fungicides early.
""",

    "Corn_(maize)___healthy": "✅ Good condition. Continue timely irrigation and use organic compost.",

    # GRAPE
    "Grape___Black_rot": """
• Remove mummified fruits & trim infected leaves immediately.
• Spray Mancozeb or Captan during early season.
• Maintain air movement by canopy training.
""",

    "Grape___Esca_(Black_Measles)": """
• Avoid waterlogging.
• Apply Trichoderma-based biofungicides to soil.
• Remove severely infected vines to prevent spread.
""",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": """
• Apply Bordeaux mixture (1%) or copper oxychloride.
• Ensure wide spacing for ventilation.
""",

    "Grape___healthy": "✅ No disease. Maintain trellis structure and prune regularly.",

    # ORANGE
    "Orange___Haunglongbing_(Citrus_greening)": """
⚠ No cure exists.
• Immediately remove infected trees to prevent spread.
• Control psyllid insects using imidacloprid spray.
• Use certified disease-free nursery plants.
""",

    # PEACH
    "Peach___Bacterial_spot": """
• Use copper hydroxide fungicide.
• Avoid wetting foliage — use drip irrigation.
• Select resistant cultivars where possible.
""",

    "Peach___healthy": "✅ Healthy. Maintain sunlight exposure and balanced fertilizer.",

    # BELL PEPPER
    "Pepper,_bell___Bacterial_spot": """
• Remove infected leaves.
• Spray copper bactericides weekly.
• Avoid touching plants when leaves are wet.
""",

    "Pepper,_bell___healthy": "✅ Healthy plant. Maintain warm temperature and avoid overwatering.",

    # POTATO
    "Potato___Early_blight": """
• Remove lower infected leaves.
• Spray Chlorothalonil or Mancozeb every 7–10 days.
• Avoid overhead watering.
""",

    "Potato___Late_blight": """
• Immediately uproot and destroy affected plants.
• Apply Metalaxyl-based fungicides preventively.
• Improve soil drainage.
""",

    "Potato___healthy": "✅ Healthy. Use compost & avoid wet soil.",

    # SQUASH
    "Squash___Powdery_mildew": """
• Spray Neem oil every 3 days until mildew disappears.
• Ensure plants receive full sunlight.
""",

    # STRAWBERRY
    "Strawberry___Leaf_scorch": """
• Remove infected leaves.
• Apply copper fungicide & improve spacing for airflow.
""",

    "Strawberry___healthy": "✅ Healthy plant.",

    # TOMATO
    "Tomato___Bacterial_spot": """
• Spray copper fungicide.
• Sterilize tools to prevent spread.
""",

    "Tomato___Early_blight": """
• Remove & dispose of lower affected leaves.
• Spray Chlorothalonil / Mancozeb weekly.
""",

    "Tomato___Late_blight": """
• Destroy infected plants immediately.
• Avoid watering leaves directly.
""",

    "Tomato___Leaf_Mold": """
• Increase ventilation in the field/greenhouse.
• Avoid high humidity.
""",

    "Tomato___Septoria_leaf_spot": """
• Remove infected leaves.
• Apply protectant fungicides like Mancozeb.
""",

    "Tomato___Spider_mites Two-spotted_spider_mite": """
• Spray neem oil or miticides such as abamectin.
• Increase humidity around plants.
""",

    "Tomato___Target_Spot": """
• Apply Mancozeb and rotate crops each season.
""",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
• Control whiteflies using yellow sticky traps.
• Remove and destroy infected plants.
""",

    "Tomato___Tomato_mosaic_virus": """
• Disinfect hands and tools.
• Remove infected leaves immediately.
""",

    "Tomato___healthy": "✅ Tomato plant is healthy! Maintain soil moisture and apply organic fertilizer."
}


# ===== Fertilizer Recommendations =====
fertilizers = {
    "Apple": "• Apply NPK 10-10-10.\n• Use compost every 60 days.",
    "Blueberry": "• Use ammonium sulfate.\n• Add acidic mulch.",
    "Cherry_(including_sour)": "• Apply NPK 12-16-12.\n• Compost tea monthly.",
    "Corn_(maize)": "• Apply Urea early.\n• Add DAP.\n• Add Zinc.",
    "Grape": "• Use NPK 13-13-13.\n• Add potash in fruiting stage.",
    "Orange": "• Apply NPK 6-6-6.\n• Use Epsom salt monthly.",
    "Peach": "• Apply NPK 16-4-8.\n• Add vermicompost.",
    "Pepper,_bell": "• Apply NPK 12-24-12.\n• Spray seaweed extract.",
    "Potato": "• Apply NPK 14-14-21.\n• Add gypsum.",
    "Strawberry": "• Apply NPK 12-12-12.\n• Add bone meal.",
    "Tomato": "• Use NPK 19-19-19 early.\n• Switch to NPK 8-16-32 during fruiting.\n• Add calcium nitrate."
}

# ===== Processing =====
def preprocess(image):
    image = image.resize((224, 224))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def predict(image):
    pred = model.predict(preprocess(image))[0]
    idx = np.argmax(pred)
    return inv_class_indices[idx], float(pred[idx] * 100)

# ===== CSS (Light Theme Only) =====
st.markdown("""
<style>
body { background: #F4FCF7; }

.result-card {
    background: #E8F7ED;
    border-left: 6px solid #28a745;
    padding: 22px; border-radius: 14px;
    margin-top: 20px;
    box-shadow: 0 0 18px rgba(52,199,89,0.45);
}

.remedy-card {
    background: #E9F6FF;
    border-left: 6px solid #2b8de0;
    padding: 22px; border-radius: 14px;
    margin-top: 18px;
    box-shadow: 0 0 18px rgba(43,141,224,0.35);
}

.fertilizer-card {
    background: #FFF8E6;
    border-left: 6px solid #c49a13;
    padding: 22px; border-radius: 14px;
    margin-top: 18px;
    box-shadow: 0 0 18px rgba(196,154,19,0.45);
}

.center-img img {
    display: block;
    margin: auto;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ===== UI =====
st.markdown("<h1 style='text-align:center; color:#166534;'>🍃 CropGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:17px;'>AI-powered plant disease diagnosis and crop-care assistance.</p>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload Leaf Image", type=["png", "jpg", "jpeg", "webp"])
analyze_clicked = st.button("🌱 Analyze Leaf", use_container_width=True)

# ===== Result =====
if uploaded_image and analyze_clicked:
    img = Image.open(uploaded_image)

    st.markdown("<div class='center-img'>", unsafe_allow_html=True)
    st.image(img, width=260)
    st.markdown("</div>", unsafe_allow_html=True)

    label, confidence = predict(img)

    st.markdown(f"<div class='result-card'><h3>✅ Diagnosis: <b>{label}</b></h3><p><b>Confidence:</b> {confidence:.2f}%</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='remedy-card'><h3>🌱 Suggested Remedy:</h3><p>{remedies.get(label, 'No remedy available.')}</p></div>", unsafe_allow_html=True)

    plant_name = label.split('___')[0]
    st.markdown(f"<div class='fertilizer-card'><h3>🌾 Fertilizer Recommendations for <b>{plant_name}</b></h3><p>{fertilizers.get(plant_name, 'No guidance available.')}</p></div>", unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align:center; margin-top:40px; font-size:14px; opacity:0.6;'>© 2025 CropGuard AI • Developed by BJT </p>", unsafe_allow_html=True)
