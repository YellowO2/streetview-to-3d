export async function fetchPanorama(lat, lon) {
  const response = await fetch(`/panorama/${lat},${lon}`);
  if (!response.ok) throw new Error("Failed to fetch panorama");
  return await response.json();
}
