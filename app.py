import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Load model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("model_weights.pth", 
                          map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    classes = ["NORMAL", "PNEUMONIA"]
    return classes[predicted.item()], confidence.item() * 100

# ---- Streamlit UI ----
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")

st.title("🫁 Chest X-Ray Pneumonia Detector")
st.write("Upload a chest X-ray image to check for Pneumonia")

uploaded_file = st.file_uploader("Choose an X-ray image", 
                                  type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with st.spinner("Analyzing X-Ray..."):
        model = load_model()
        processed = preprocess(image)
        result, confidence = predict(model, processed)

    st.markdown("---")

    if result == "PNEUMONIA":
        st.error(f"⚠️ Result: **{result}**")
    else:
        st.success(f"✅ Result: **{result}**")

    st.metric(label="Confidence", value=f"{confidence:.2f}%")
    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only. "
               "Always consult a doctor for medical advice.")