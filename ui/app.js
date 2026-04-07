import { SceneManager } from "./SceneManager.js"; //spark's scene manager
import { fetchPanorama } from "./api.js";

async function init() {
  const scene = new SceneManager();
  scene.startRenderLoop();

  // document.getElementById("startButton").addEventListener("click", (e) => {
  //   scene.enableDeviceOrientation();
  //   e.target.style.display = "none";
  // });

  try {
    // Example coordinates
    // const data = await fetchPanorama(41.8982208, 12.4764804);
    // console.log("Fetched panorama data:", data);
    // Use the returned image URL
    scene.addPanorama("/images/test_better.jpg");
  } catch (error) {
    console.error("Error loading map view:", error);
  }
}

init();
