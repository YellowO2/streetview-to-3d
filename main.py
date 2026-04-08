from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from streetlevel import streetview
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientSession

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

session: ClientSession = None

@app.on_event("startup")
async def startup_event():
    app.state.session = ClientSession()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.session.close()

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

    pano = await streetview.find_panorama_async(lat, lon, session=app.state.session)
    if not pano:
        return {"error": "Panorama not found"}
    print(f"Found panorama: {pano}")
    
    # save image to disk and return the path
    await streetview.download_panorama_async(pano, image_path, session=app.state.session)
    return {"image_path": "/" + image_path}


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")