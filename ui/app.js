import { SceneManager } from "./SceneManager.js"; //spark's scene manager
import { fetchPanorama } from "./api.js";

let sceneManager = null;

function setStatus(message, isError = false) {
  const status = document.getElementById("status");
  status.textContent = message;
  status.style.color = isError ? "#b42318" : "#2f3a4a";
}

function InitScene() {
  if (sceneManager) return sceneManager;
  sceneManager = new SceneManager();
  sceneManager.startRenderLoop();
  return sceneManager;
}

function AddPanoramaToScene(imageUrl) {
  const scene = InitScene();
  if (scene.sphere) {
    scene.scene.remove(scene.sphere);
    scene.sphere = null;
  }
  scene.addPanorama(imageUrl);
}

function extractLatLon(input) {
  const raw = input.trim();
  const atMatch = raw.match(/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/);
  if (atMatch) return { lat: Number(atMatch[1]), lon: Number(atMatch[2]) };

  const directMatch = raw.match(/^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$/);
  if (directMatch)
    return { lat: Number(directMatch[1]), lon: Number(directMatch[2]) };

  throw new Error(
    "Use a Google Maps URL with /@lat,lon or input lat,lon directly.",
  );
}

async function handleSearchSubmit(event) {
  event.preventDefault();

  try {
    const input = document.getElementById("mapsUrlInput").value;
    const { lat, lon } = extractLatLon(input);
    setStatus("Fetching panorama...");
    const data = await fetchPanorama(lat, lon);

    if (data.error) throw new Error(data.error);
    if (!data.image_path) throw new Error("Backend did not return image_path.");

    AddPanoramaToScene(data.image_path);
    setStatus("Panorama loaded.");
  } catch (error) {
    console.error("Error loading map view:", error);
    setStatus(error.message || "Failed to load panorama.", true);
  }
}

function handleExampleClick() {
  AddPanoramaToScene("/images/panorama_example.jpg");
  setStatus("Loaded example panorama.");
}

function init() {
  document
    .getElementById("searchForm")
    .addEventListener("submit", handleSearchSubmit);
  document
    .getElementById("exampleButton")
    .addEventListener("click", handleExampleClick);
}

init();
