from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from searchAzure import get_image_vector, search_azure
from PIL import Image
import io
import base64

app = FastAPI()

class ImageBase64Request(BaseModel):
    image_base64: str

@app.post("/search")
async def search(request: ImageBase64Request):
    try:
        # Decode base64 string to bytes
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate embedding and search
        embedding = get_image_vector(image)
        result = search_azure(embedding)
        value = "Not found" if result is None or len(result) == 0 else result
        return {"result": value[0].get("description", "No description")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image input. Error: {str(e)}")

@app.get("/test")
async def test():
    return "API is running successfully"
