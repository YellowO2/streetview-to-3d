"""Street View / Look Around panoramas along a street, reconstructed into one
placed 3D point cloud. See ARCHITECTURE.md for the stages."""
# First, so `spaces` is imported before anything initialises CUDA.
from streetview_to_3d import gpu  # noqa: F401
