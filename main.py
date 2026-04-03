from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from streetview import search_panoramas
from streetview import get_panorama
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

    # check if image already exists
    image_path = f"images/panorama_{lat}_{lon}.jpg"
    try:        
        with open(image_path, "rb") as f:
            return {"image_path": "/" + image_path}
    except FileNotFoundError:
        pass

    panos = search_panoramas(lat=lat, lon=lon)
    first = panos[0]
    print(f"Found panorama: {first}")
    image = get_panorama(pano_id=first.pano_id)
    # save image to disk and return the path
    image.save(image_path)
    return {"image_path": "/" + image_path}


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")