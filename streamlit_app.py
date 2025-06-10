import streamlit as st
from PIL import Image
import openai
import requests
from io import BytesIO

# OpenAI key
openai.api_key = st.secrets["OPEN_AI_KEY_SRI"]

# Lesion class labels
lesion_types = {
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma",
    "bcc": "Basal Cell Carcinoma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "bkl": "Benign Keratosis",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma"
}

# Map user-friendly body location labels to HAM10000 format
location_map = {
    "Abdomen": "abdomen",
    "Back": "back",
    "Chest": "chest",
    "Legs or Feet": "lower extremity",
    "Arms or Hands": "upper extremity",
    "Scalp": "scalp",
    "Face": "face"
}

# Predict class from FastAPI
def predict_class(image, age, sex, body_location):
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    try:
        response = requests.post(
            "https://skinscan-700139180257.europe-west1.run.app/predict",
            files={"file": ("image.png", image_bytes, "image/png")},
            data={
                "age": str(age),
                "sex": sex,
                "body_location": body_location
            }
        )
        if response.status_code == 200:
            result = response.json()
            top_class = max(result, key=result.get)
            return top_class, result
        else:
            st.error(f"Prediction API Error: {response.status_code}")
            return "unknown", {}
    except Exception as e:
        st.error(f"API Error: {e}")
        return "unknown", {}

# OpenAI GPT-3.5 explanation
def patient_report(predicted_class, age, sex, body_location, lifestyle_work, max_tokens=150):
    system_prompt = (
        "You are a friendly and helpful dermatology assistant. "
        "You explain skin cancer results in a concise, calm, simple language that’s easy for anyone to understand. "
        "You avoid medical jargon, speak with warmth, and gently guide the patient on what to do next. "
        "Limit to 150 words."
    )

    prompt = f"""
    A dermatology AI model predicted: {lesion_types.get(predicted_class.lower(), 'Unknown Lesion')}

    Patient details:
    - Age: {age}
    - Sex: {sex}
    - Lesion Location: {body_location}
    - Lifestyle/Work Type: {lifestyle_work}

    Please explain in a friendly and concise manner, why it might appear at this age, body location, and sex,
    how the person's lifestyle or work may affect it, and what they should do next.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

# Streamlit UI
st.title("Skin Scan WEBSITE")

age = st.number_input("Patient Age TEST", min_value=1, max_value=120, value=30)
sex = st.selectbox("Patient Sex", ["Male", "Female", "Other"])
user_friendly_location = st.selectbox("Lesion Location", [
    "Abdomen", "Back", "Chest", "Legs or Feet", "Arms or Hands", "Scalp", "Face"
])
lifestyle_work = st.text_input("Lifestyle/Work Description", "Outdoors")
uploaded_image = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼️ Uploaded Skin Lesion", use_container_width=True)

    if st.button("Get Prediction"):
        with st.spinner("Analyzing image..."):
            mapped_location = location_map.get(user_friendly_location, "unknown")

            if mapped_location == "unknown":
                st.warning("Selected lesion location is not mapped to a known body region in HAM10000.")

            predicted_class, all_probs = predict_class(image, age, sex, mapped_location)

        if predicted_class != "unknown":
            st.subheader("AI Model Prediction")
            st.write(f"**Top Prediction:** {lesion_types.get(predicted_class, predicted_class)}")

            # Sort and format probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            formatted_probs = {}
            for code, prob in sorted_probs:
                try:
                    percent = round(float(prob) * 100, 1)
                    formatted_probs[lesion_types.get(code, code)] = f"{percent}%"
                except (ValueError, TypeError):
                    formatted_probs[lesion_types.get(code, code)] = "N/A"

            st.subheader("All Class Probabilities")
            for label, percent in formatted_probs.items():
                st.write(f"{label} : {percent}")

            report = patient_report(predicted_class, age, sex, mapped_location, lifestyle_work)
            st.subheader("AI Explanation Report")
            st.write(report)
