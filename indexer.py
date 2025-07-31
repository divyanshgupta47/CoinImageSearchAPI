import os
import uuid
import numpy as np
from PIL import Image
import clip
import torch
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

endpoint = "https://stockaissearchbasic.search.windows.net"
admin_key = "De209qaEogESGPGOVtkS8jdsT2VeUJXfw0h8p4YEE5AzSeAixLgZ"
index_name = "images-index"

search_client = SearchClient(endpoint=endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(admin_key))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def search_by_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()[0].tolist()

    results = search_client.search(
        search_text=None,
        vector=image_features,
        top_k=5,
        vector_fields="imageVector"
    )
    return [{"id": r["id"], "label": r.get("label", ""), "score": r["@search.score"]} for r in results]