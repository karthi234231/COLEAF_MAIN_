import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import pickle
import base64

# Function to encode image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get Base64 string of the image
image_base64 = get_base64_of_image("C:/Users/KARTHIK/Documents/CoLeaf_Final/bg.jpg")

# --- Custom CSS for High-End UI ---
st.markdown(
    f"""
    <style>
        * {{
             background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                        url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
           
           
            
        }}

        /* Custom Progress Bar Color */
        div[data-testid="stStatusWidget"] div[role="progressbar"] {{
            background-color: #4b382a !important;
        }}

        /* Heading Styling */
        h1 {{
            animation: fadeIn 1s ease-in-out;
            text-align: center;
            font-weight: 600;
            font-size: 28px;
            color: #4b382a;
            text-transform: uppercase;
            letter-spacing: 1px;
            line-height: 1.2;
            margin: auto;
            padding: 15px 0;
        }}

        /* Container */
        .container {{
            max-width: 90%;
            margin: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}

        /* Upload Section */
        .upload-section {{
            background: rgba(76, 72, 72, 0.6);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: all 0.3s ease-in-out;
        }}

        .upload-section:hover {{
            transform: scale(1.02);
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.3);
        }}

        /* Output Box */
        .output-box {{
            background: rgba(96, 94, 94, 0.5);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            max-width: 600px;
            margin: 20px auto;
            text-align: center;
        }}

        .output-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.3);
        }}

        /* Prediction Text */
        .prediction {{
            font-size: 22px;
            font-weight: 700;
            background:white;
            color: #2f6637;
            text-align: center;
            text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.15);
            padding: 10px;
            border-bottom: 2px solid #4b382a;
            display: inline-block;
        }}

        /* Info Section */
        .info-section {{
            font-size: 18px;
            font-weight: 500;
            color: #333;
            padding: 15px;
            background: rgba(245, 245, 245, 0.9);
            border-radius: 10px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1);
            margin-top: 15px;
        }}

        /* Button Styling */
        .button:hover {{
            background: linear-gradient(45deg, rgb(2, 2, 2), rgb(4, 4, 4));
            transform: translateY(-3px);
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
        }}

        /* Image Preview */
        .image-preview {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            margin-top: 15px;
        }}

        /* Fade In Animation */
        .fadeIn {{
            animation: fadeIn 1s ease-in-out;
        }}
        .symptoms-box {{
            background-color: rgba(255, 230, 204, 0.9); /* Light Orange */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }}
        .info-section strong {{
            background-color: #ffcc80; /* Light Orange for Symptoms */
            padding: 5px 10px;
            border-radius: 5px;
            color: #4b382a; /* Dark Text for Contrast */
        }}

        .info-section:nth-of-type(2) strong {{
            background-color: white; /* Light Green for Remedies */
        }}
        .remedies-box {{
            background-color: white; /* Light Green */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            h1 {{
                font-size: 26px;
                letter-spacing: 0.5px;
                max-width: 90%;
            }}

            .output-box {{
                padding: 20px;
            }}

            .prediction {{
                font-size: 20px;
            }}

            .info-section {{
                font-size: 16px;
            }}

            .button {{
                font-size: 16px;
                padding: 10px 20px;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("Coffee leaf nutrient deficiencies detection and classification")

# --- Deficiency Information ---
deficiency_info = {
    "nitrogen-N": {
        "Symptoms": [
            "Older leaves turn yellow (chlorosis), starting from the tips and edges.",
            "Stunted plant growth and smaller leaves."
        ],
        "Natural Remedies": [
            "Add well-rotted manure or compost to the soil.",
            "Grow nitrogen-fixing plants (e.g., clover or cowpeas).",
            "Use fish emulsion or seaweed extract as foliar sprays."
        ]
    },
    "phosphorus-P": {
        "Symptoms": [
            "Dark green or purplish tint on older leaves.",
            "Slow growth and delayed flowering."
        ],
        "Natural Remedies": [
            "Apply bone meal for slow phosphorus release.",
            "Use rock phosphate for long-term replenishment.",
            "Chop and bury banana peels around the plant."
        ]
    },
    "potassium-K": {
        "Symptoms": [
            "Leaf edges and tips turn brown and scorched.",
            "Weak branches and poor crop development."
        ],
        "Natural Remedies": [
            "Spread wood ash lightly around plants.",
            "Compost banana peels as they are rich in potassium.",
            "Apply poultry manure in small amounts."
        ]
    },
    "calcium-Ca": {
        "Symptoms": [
            "New leaves appear deformed or distorted.",
            "Roots are weak and underdeveloped."
        ],
        "Natural Remedies": [
            "Crush and scatter eggshells around the plant for slow calcium release.",
            "Use agricultural lime to neutralize acidity and add calcium.",
            "Add gypsum if calcium is needed without altering soil pH."
        ]
    },
    "magnesium-Mg": {
        "Symptoms": [
            "Yellowing between veins of older leaves, leaving green veins intact."
        ],
        "Natural Remedies": [
            "Dissolve 1 tablespoon of Epsom salt in 1 gallon of water and spray on leaves.",
            "Add dolomite lime to correct magnesium levels and neutralize soil acidity.",
            "Use compost made from green leafy vegetables."
        ]
    },
    "iron-Fe": {
        "Symptoms": [
            "Yellowing between veins of young leaves (interveinal chlorosis)."
        ],
        "Natural Remedies": [
            "Apply foliar sprays of chelated iron for quick absorption.",
            "Use spent coffee grounds to slightly acidify the soil and improve iron uptake.",
            "Enrich compost with iron-rich materials, like green leafy waste."
        ]
    },
    "boron-B": {
        "Symptoms": [
            "Poor flowering and fruit set.",
            "Deformed or brittle leaves."
        ],
        "Natural Remedies": [
            "Add a small amount of borax to the soil (not exceeding 1 teaspoon per gallon).",
            "Use well-aged manure for trace boron content."
        ]
    },
    "manganese-Mn": {
        "Symptoms": [
            "Interveinal chlorosiAAs on younger leaves with brown spots."
        ],
        "Natural Remedies": [
            "Apply liquid seaweed extract for manganese and other trace elements.",
            "Use manganese sulfate as a foliar spray.",
            "Mulch with decomposed leaves to add manganese over time."
        ]
    }
}

# --- Load the trained MobileNet model ---
@st.cache_resource(show_spinner=False)
def load_trained_model():
    try:
        model = load_model("best_mobilenet_model.keras")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# --- Load label encoder ---
@st.cache_resource()
def load_label_encoder():
    try:
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        st.error(f"❌ Error loading label encoder: {e}")
        return None

# --- Preprocess image ---
def preprocess_image(image):
    try:
        image_resized = cv2.resize(image, (128, 128))
        image_resized = image_resized.astype("float32") / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)
        return image_resized
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# --- Classify image ---
def classify_image(model, label_encoder, image):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return None
    try:
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_class
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
        return None

# --- Streamlit UI ---

uploaded_file = st.file_uploader("📸 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- Layout with Columns ---
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(original_image, caption="📷 Uploaded Image", use_column_width=False, channels="BGR", output_format="JPEG", width=250)

    with col2:
        st.markdown("<div class='output-box'>", unsafe_allow_html=True)
        
        model = load_trained_model()
        label_encoder = load_label_encoder()

        if model and label_encoder:
            predicted_class = classify_image(model, label_encoder, original_image)
            if predicted_class:
                st.markdown(f"<p class='prediction'>✅ Prediction: {predicted_class}</p>", unsafe_allow_html=True)
                
                if predicted_class in deficiency_info:
                    st.markdown(
    f"<p class='info-section'><strong style='background:white; padding:5px 10px; border-radius:5px; color:green;'>🩺 Symptoms:</strong> {', '.join(deficiency_info[predicted_class]['Symptoms'])}</p>",
    unsafe_allow_html=True
)
                    st.markdown(
    f"<p class='info-section'><strong style='background:white; padding:5px 10px; border-radius:5px; color:green'>🌱 Natural Remedies:</strong> {', '.join(deficiency_info[predicted_class]['Natural Remedies'])}</p>",
    unsafe_allow_html=True
)
                else:
                    st.info("No additional information available for this deficiency.")
        
        st.markdown("</div>", unsafe_allow_html=True)






