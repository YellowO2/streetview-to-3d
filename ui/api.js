export async function fetchPanorama(lat, lon) {
  const response = await fetch(`/panorama/${lat},${lon}`);
  if (!response.ok) throw new Error("Failed to fetch panorama");
  return await response.json();
}

export async function fetchMetadata(lat, lon, panoId = null) {
  const url = panoId
    ? `/metadata?pano_id=${panoId}`
    : `/metadata?lat=${lat}&lon=${lon}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error("Failed to fetch metadata");
  return await response.json();
}

export async function requestBatchDownload(panoIds, downloadDepth, metadataList) {
  const response = await fetch("/batch_download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pano_ids: panoIds,
      download_depth: downloadDepth,
      metadata: metadataList,
    }),
  });
  if (!response.ok) throw new Error("Failed to batch download");
  return await response.blob();
}

export async function generateSplat(panoIds, metadata) {
  const response = await fetch("/generate_3dgs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pano_ids: panoIds, metadata }),
  });
  if (!response.ok) throw new Error("Failed to start 3DGS generation");
  return await response.json(); // { job_id }
}

export async function getJobStatus(jobId) {
  const response = await fetch(`/job/${jobId}`);
  if (!response.ok) throw new Error("Failed to get job status");
  return await response.json(); // { status, ply_files, error }
}
