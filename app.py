from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from indexer import search_by_image
from searchAzure import get_image_vector, search_azure
from PIL import Image
import tempfile

app = FastAPI()

@app.post("/search")
async def search(image: UploadFile = File(...)):
    if not image:
        return JSONResponse(content={"error": "No image uploaded"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        tmp.write(await image.read())

    img = Image.open(image_path).convert("RGB")
    embedding = get_image_vector(img)
    value = search_azure(embedding)
    return value

@app.get("/test")
async def test():
    return "ok"
