import streamlit as st
import requests
import torch
from torchvision import models, transforms
from PIL import Image

# --- Azure Search config ---
endpoint = "https://stockaissearchbasic.search.windows.net"
index_name = "images-index"
api_key = "De209qaEogESGPGOVtkS8jdsT2VeUJXfw0h8p4YEE5AzSeAixLgZ"  # Replace with your actual key
api_version = "2023-11-01"

headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

# --- Load ResNet18 model for embeddings ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model.eval()
    return model

model = load_model()

# --- Transform for image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Generate vector embedding from image ---
def get_image_vector(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(image_tensor).squeeze().numpy()  # Shape (512,)
    return embedding.tolist()

# --- Search in Azure Cognitive Search using vector ---
def search_azure(embedding):
    search_url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"
    payload = {
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": embedding,
                "k": 1,
                "fields": "vector"
            }
        ],
        "select": "*"
    }

    response = requests.post(search_url, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"Search failed: {response.status_code}")
        st.json(response.json())
        return []

    data = response.json()
    return data.get("value", [])

# --- Streamlit UI ---
st.title("🔍 Image Search using Azure AI Search")

uploaded_file = st.file_uploader("Upload an image to search", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔎 Generating vector and searching..."):
        embedding = get_image_vector(image)
        results = search_azure(embedding)

    if results:
        st.subheader("🔗 Top Matches Found:")
        for i, result in enumerate(results):
            if isinstance(result, dict):
                st.markdown(f"**{i+1}. Description:** {result.get('description', 'No description')}**")
            else:
                st.warning(f"Unexpected result format: {result}")
    else:
        st.info("❌ No similar images found.")
