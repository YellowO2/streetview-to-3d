import { SceneManager } from "./SceneManager.js"; //spark's scene manager
import { fetchPanorama, fetchMetadata, requestBatchDownload } from "./api.js";

let sceneManager = null;
const discoveredPanos = new Set(); // Keep track of fetched panos to avoid downloading dupes broadly

let leafletMap = null;
let mapMarkers = {}; // Keep track of markers by pano_id
let polylines = [];

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

function updatePanoCount() {
  document.getElementById("panoCount").innerText = `(${discoveredPanos.size})`;
}

function initOrResetMap(lat, lon) {
  const mapElement = document.getElementById("map");
  mapElement.style.display = "block";
  mapMarkers = {};
  polylines = [];
  window.drawnLines = new Set(); // Reset drawn lines tracker

  if (!leafletMap) {
    leafletMap = L.map("map").setView([lat, lon], 18);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap",
    }).addTo(leafletMap);
  } else {
    leafletMap.eachLayer((layer) => {
      // remove old markers and lines
      if (layer instanceof L.Marker || layer instanceof L.Polyline) {
        leafletMap.removeLayer(layer);
      }
    });
    leafletMap.setView([lat, lon], 18);
  }
}

function drawNodeOnMap(metadata, isRoot = false) {
  if (!leafletMap || mapMarkers[metadata.id]) return;

  const marker = L.marker([metadata.lat, metadata.lon]).addTo(leafletMap);

  const container = document.createElement("div");
  container.innerHTML = `
    <div style="margin-bottom: 8px;"><b>${isRoot ? "Root " : ""}Pano:</b> <span class="pano-id">${metadata.id}</span></div>
    <div style="margin-bottom: 8px; font-size: 12px; color: #555;">Lat: ${metadata.lat.toFixed(6)}, Lon: ${metadata.lon.toFixed(6)}</div>
    <div style="display: flex; gap: 8px;">
      <button class="view-btn secondary" style="font-size: 12px; padding: 4px 8px; margin: 0;">View 3D</button>
      <button class="expand-btn" style="font-size: 12px; padding: 4px 8px; margin: 0;">Expand Neighbors (+)</button>
    </div>
  `;

  // Attach view 3D event
  container.querySelector(".view-btn").onclick = async () => {
    setStatus(`Loading panorama ${metadata.id}...`);
    try {
      const data = await fetchPanorama(metadata.lat, metadata.lon);
      if (data.error) throw new Error(data.error);
      AddPanoramaToScene(data.image_path);
      setStatus(`Panorama ${metadata.id} loaded.`);
    } catch (err) {
      setStatus(err.message, true);
    }
  };

  // Attach expand neighbors event
  const expandBtn = container.querySelector(".expand-btn");
  expandBtn.onclick = async () => {
    expandBtn.disabled = true;
    expandBtn.textContent = "Expanding...";
    try {
      const fullMetadata = await fetchMetadata(
        metadata.lat,
        metadata.lon,
        metadata.id,
      );
      if (fullMetadata.links) {
        for (const link of fullMetadata.links) {
          if (!discoveredPanos.has(link.id)) {
            discoveredPanos.add(link.id);
            drawNodeOnMap(link); // map markers wait for interaction
          }
          const lineId = [metadata.id, link.id].sort().join("-");
          if (!window.drawnLines.has(lineId)) {
            window.drawnLines.add(lineId);
            drawMapLinks(metadata.lat, metadata.lon, link);
          }
        }
        updatePanoCount();
      }
      expandBtn.textContent = "Expanded";
    } catch (err) {
      console.error(err);
      setStatus("Failed to expand node.", true);
      expandBtn.disabled = false;
      expandBtn.textContent = "Retry Expand (+)";
    }
  };

  marker.bindPopup(container);
  mapMarkers[metadata.id] = marker;

  if (isRoot) {
    marker.openPopup();
  }
}

function drawMapLinks(parentLat, parentLon, link) {
  if (!leafletMap) return;
  // draw a line connecting parent to child
  const polyline = L.polyline(
    [
      [parentLat, parentLon],
      [link.lat, link.lon],
    ],
    { color: "blue", weight: 2, opacity: 0.6 },
  ).addTo(leafletMap);
  polylines.push(polyline);
}

async function handleSearchSubmit(event) {
  event.preventDefault();

  try {
    const input = document.getElementById("mapsUrlInput").value;
    const { lat, lon } = extractLatLon(input);
    setStatus("Fetching panorama metadata...");

    // Reset state
    discoveredPanos.clear();
    const treeContainer = document.getElementById("treeContainer");
    treeContainer.style.display = "block";

    // 1. Get Root Metadata
    const data = await fetchMetadata(lat, lon);
    if (data.error) throw new Error(data.error);

    initOrResetMap(data.lat, data.lon);

    discoveredPanos.add(data.id);
    drawNodeOnMap(data, true);
    updatePanoCount();

    setStatus("Root panorama fetched. Click map markers to expand the path.");

    // Automatically load in 3D viewer
    const panoRes = await fetchPanorama(data.lat, data.lon);
    if (!panoRes.error && panoRes.image_path) {
      AddPanoramaToScene(panoRes.image_path);
    }
  } catch (error) {
    console.error("Error fetching map view:", error);
    setStatus(error.message || "Failed to fetch panorama.", true);
  }
}

async function handleDownloadAllClick() {
  if (discoveredPanos.size === 0) return;
  const btn = document.getElementById("downloadAllBtn");
  btn.disabled = true;

  const depthCheckbox = document.getElementById("depthCheckbox");
  const downloadDepth = depthCheckbox ? depthCheckbox.checked : false;

  const metadataList = [];
  let current = 0;

  for (const panoId of discoveredPanos) {
    current++;
    btn.textContent = `Downloading ${current}/${discoveredPanos.size} Panoramas...`;
    try {
      const resp = await fetch("/download_pano", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pano_id: panoId,
          download_depth: downloadDepth,
        }),
      });
      if (resp.ok) {
        const meta = await resp.json();
        if (!meta.error) {
          metadataList.push(meta);
        }
      }
    } catch (e) {
      console.error(e);
    }
  }

  btn.textContent = `Zipping files...`;
  try {
    const blob = await requestBatchDownload(
      Array.from(discoveredPanos),
      downloadDepth,
      metadataList,
    );
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `panoramas_${Date.now()}.zip`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
  } catch (e) {
    console.error(e);
    alert("Batch download failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Download All Discovered";
  }
}

function handleExampleClick() {
  const input = document.getElementById("mapsUrlInput");
  input.value = "1.323673, 103.7555071";
  document.getElementById("searchForm").dispatchEvent(new Event("submit"));
}

function init() {
  document
    .getElementById("searchForm")
    .addEventListener("submit", handleSearchSubmit);
  document
    .getElementById("exampleButton")
    .addEventListener("click", handleExampleClick);
  document
    .getElementById("downloadAllBtn")
    .addEventListener("click", handleDownloadAllClick);
}

init();
