from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
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
    result = search_azure(embedding)
    value = "Not found" if result is None or len(result) == 0 else result
    return value[0].get("description", "No description")

@app.get("/test")
async def test():
    return "ok"
