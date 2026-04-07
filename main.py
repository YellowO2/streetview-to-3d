from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from streetlevel import streetview
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/panorama/{lat},{lon}")
async def panorama(lat: float, lon: float):

    print(f"Received request for panorama at lat: {lat}, lon: {lon}")
    # check if image already exists
    image_path = f"images/panorama_{lat}_{lon}.jpg"
    try:        
        with open(image_path, "rb") as f:
            return {"image_path": "/" + image_path}
    except FileNotFoundError:
        pass

    pano = streetview.find_panorama(lat, lon)
    if not pano:
        return {"error": "Panorama not found"}
    print(f"Found panorama: {pano}")
    
    # save image to disk and return the path
    streetview.download_panorama(pano, image_path)
    return {"image_path": "/" + image_path}


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")