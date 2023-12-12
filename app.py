from fastapi import FastAPI,HTTPException

from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import uvicorn
from fastapi.openapi.utils import get_openapi
from Generate_best_crop import best_crop,correct_orientation
import numpy as np
import logging
import zipfile
import asyncio
from PIL import Image
from io import BytesIO
app = FastAPI()
logger = logging.getLogger("uvicorn.error")
executor = ThreadPoolExecutor()

@app.get("/health")
def read_root():
    return {"message": "Hello, FastAPI!"}


async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                image_bytes = await response.read()
                image_bytes = bytearray(image_bytes)
                image = Image.open(BytesIO(image_bytes))
                image = correct_orientation(image)
                return image
            else:
                return None
def pillow_image_to_bytes(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()
    return image_bytes

@app.get("/default_crops/{url:path}")
async def remove_bg(url: str):
    try:
        image = await fetch_image(url)
        crops = best_crop(image)
        crops[0].save("crops.png")
        crop_bytes = pillow_image_to_bytes(crops[0])
        return StreamingResponse(BytesIO(crop_bytes), media_type="image/PNG")   
    except AssertionError as e:
        logger.error(f"Assertion error: {e}")
        raise HTTPException(status_code=400, detail="Image cannot be processed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=400, detail="Image cannot be processed")


# @app.get("/crops/class0")
# async def get_class0_crops(url: str):
#     try:
#         image = await fetch_image(url)
#         crops = get_crops(image, class_index=0, confidence_threshold=0.6, desired_ratio=1.25)
#         return crops
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

# @app.get("/crops/class1")
# async def get_class1_crops(url: str):
#     try:
#         image = await fetch_image(url)
#         crops = get_crops(image, class_index=1, confidence_threshold=0.6, desired_ratio=0.8)
#         return crops
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


    #     image_byte_array = pillow_image_to_bytes()
    #     return StreamingResponse(BytesIO(image_byte_array), media_type="image/PNG")
    # except AssertionError as e:
    #     logger.error(f"Assertion error: {e}")
    #     raise HTTPException(status_code=400, detail="Image cannot be processed")
    # except Exception as e:
    #     logger.error(f"Unexpected error: {e}")
    #     raise HTTPException(status_code=400, detail="Image cannot be processed")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Best CROP",
        version="0.1.0",
        description="API for getting best default crop",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app)