import os

from aiohttp import ClientSession
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.street_view import patch_tile_url
from routes.panorama import router as panorama_router
from routes.splatting import router as splatting_router

patch_tile_url()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    app.state.session = ClientSession()
    os.makedirs("images", exist_ok=True)
    os.makedirs("splats", exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.session.close()


app.include_router(panorama_router)
app.include_router(splatting_router)

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/splats", StaticFiles(directory="splats"), name="splats")
