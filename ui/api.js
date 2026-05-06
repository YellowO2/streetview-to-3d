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

export async function requestBatchDownload(
  panoIds,
  downloadDepth,
  metadataList,
) {
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
  // Assuming it returns a ZIP blob
  return await response.blob();
}
