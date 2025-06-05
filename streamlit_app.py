import streamlit as st
from PIL import Image
import openai

# Your lesion types
lesion_types = {
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma",
    "bcc": "Basal Cell Carcinoma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "bkl": "Benign Keratosis",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma"
}

# OpenAI API setup (reads securely from .streamlit/secrets.toml)
openai.api_key = st.secrets["OPEN_AI_KEY_SRI"]

def patient_report(predicted_class, age, sex, body_location, lifestyle_work, max_tokens=150):
    system_prompt = (
        "You are a friendly and helpful dermatology assistant. "
        "You explain skin cancer results in a concise, calm, simple language that’s easy for anyone to understand. "
        "You avoid medical jargon, speak with warmth, and gently guide the patient on what to do next. "
        "Limit to 150 words."
    )

    prompt = f"""
    A dermatology AI model predicted: {lesion_types[predicted_class.lower()]}

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

# Dummy model prediction (replace this with your actual model call)
def predict_class(image):
    return "nv"  # fake prediction for testing

# Streamlit UI
st.title("Skin Scan")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female", "Other"])
body_location = st.selectbox("Lesion Location", [
    "Abdomen", "Back", "Chest", "Legs or feet", "Arms or hands", "Scalp", "Face"
])
lifestyle_work = st.text_input("Lifestyle/Work", "outdoors")
uploaded_image = st.file_uploader("Upload skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Lesion", use_container_width=True)

    # Predict
    predicted_class = predict_class(image)

    # Generate report
    report = patient_report(predicted_class, age, sex, body_location, lifestyle_work)

    # Output
    st.subheader("AI Model Prediction")
    st.write(lesion_types[predicted_class])

    st.subheader("AI Report")
    st.write(report)
