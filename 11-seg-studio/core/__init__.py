"""Classical computer vision from scratch: k-means, region growing, components."""

from .kmeans import kmeans, segment_image
from .region_grow import region_grow, overlay_mask
from .components import connected_components, count_objects

__all__ = [
    "kmeans", "segment_image", "region_grow", "overlay_mask",
    "connected_components", "count_objects",
]
