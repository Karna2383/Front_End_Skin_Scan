import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import openai
import requests
from io import BytesIO

from streamlit_option_menu import option_menu
import streamlit as st

with st.sidebar:
    page = option_menu(
        menu_title="Go to",
        options=["Introduction", "Skin Scan Diagnosis", "About Us"],
        icons=["house", "camera", "info-circle"],
        styles={
            "container": {"padding":"10", "width":"stretch"},
            "nav-link": {"font-size":"18px","padding":"8px 0"},
            "nav-link-selected": {"font-size":"20px"}
        }
    )


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

# Body location mapping
location_map = {
    "Abdomen": "abdomen",
    "Back": "back",
    "Chest": "chest",
    "Legs or Feet": "lower extremity",
    "Arms or Hands": "upper extremity",
    "Scalp": "scalp",
    "Face": "face"
}

# FastAPI prediction function
def predict_class(image, age, sex, body_location):
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    try:
        response = requests.post(
            "https://skinscan-700139180257.europe-west1.run.app/predict",
            files={"file": ("image.png", image_bytes, "image/png")},
            data={"age": str(age), "sex": sex, "body_location": body_location}
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

# GPT-based report generation
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

# Page 1: Introduction
if page == "Introduction":
    st.image("skinscan.jpg", width=1444)
    st.title("🌿 Welcome to Skin Scan")
    st.subheader("Your Smart Dermatology Assistant")
    st.markdown("""
        <style>
    body {
        background: linear-gradient(to right, #ece9e6, #ffffff);
    }
    </style>
    **Skin Scan** is your friendly AI-powered dermatology assistant.
    Just upload a skin lesion(🐭) photo, and our model gives you a possible match with a simple explanation.

    ⬆️ Upload your photo \n
    👾 Get an AI prediction \n
    💭 Understand what it might mean \n

    ---
    **Note:** This tool does not replace medical professionals. It is for educational purposes only.
    """,
    unsafe_allow_html = True)

# Page 2: Diagnosis Tool
elif page == "Skin Scan Diagnosis":
    st.title("🔬 Skin Scan Diagnosis")

    age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Patient Sex", ["Male", "Female"])
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
                predicted_class, all_probs = predict_class(image, age, sex, mapped_location)

            if predicted_class != "unknown":
                st.subheader("AI Model Prediction")
                st.write(f"**Top Prediction:** {lesion_types.get(predicted_class, predicted_class)}")

                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                formatted_probs = {
                    lesion_types.get(code, code): f"{round(float(prob) * 100, 1)}%"
                    for code, prob in sorted_probs
                }

                st.subheader("All Class Probabilities")
                for label, percent in formatted_probs.items():
                    st.write(f"{label} : {percent}")

                report = patient_report(predicted_class, age, sex, mapped_location, lifestyle_work)
                st.subheader("AI Explanation Report")
                st.write(report)

# Page 3: About Us
elif page == "About Us":
    st.title("👥 About Us")

    st.markdown("""
    **About Skin Scan**

    Skin Scan is an AI-powered dermatology assistant developed as part of the **Le Wagon Data Science Bootcamp**.

    We are a team of three with diverse backgrounds in engineering, analysis, and consulting, who are interested in using technology to create new solutions. Our current focus is on making early skin lesion insights more accessible and understandable.

    **Our Goals**

    - 🧠 Empower users with early, AI-generated insights
    - 📚 Provide calm, clear explanations without medical jargon
    - 🤝 Support clinical awareness through educational tools

    ---

    **Who We Are**

    **🔹 Charlie Saunders**
    Software Engineer at SeaMap and former Royal Navy Weapon Engineering Technician.
    *Hobbies:* Rock climbing and cycling.
    GitHub: [@Chapungu](https://github.com/Chapungu)

    **🔹 Marcin Mochnacki**
    Technology, Media & Telecommunications Consultant with a BSc in Mathematics and Economics from LSE.
    *Hobbies:* Polish politics and chess.
    GitHub: [@mohnatz](https://github.com/mohnatz)

    **🔹 Srikant Vedutla**
    A specialist recruiter in Azure Cloud, Data, and Insurance roles within the financial sector.
    *Hobbies:* Football and poker.
    GitHub: [@Karna2383](https://github.com/Karna2383)

    ---

    **Contact Us**
    For feedback, collaboration, or questions, feel free to reach out via GitHub.

    ---
    ⚠️ **Disclaimer:**
    This tool is powered by AI and is intended for educational and informational purposes only.
    It does **not** provide medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for any skin concerns or conditions.
    """)
