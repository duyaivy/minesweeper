"""Asset loader with caching — load images once, scale, reuse."""

import os
import pygame
from src.config import IMAGES_DIR, CELL_SIZE


class AssetLoader:
    """Loads and caches pygame image surfaces."""

    def __init__(self):
        self._cache: dict[str, pygame.Surface] = {}

    def load_image(self, filename: str, size: tuple | None = None) -> pygame.Surface | None:
        """Load an image from assets/images/. Returns None if file missing."""
        key = f"{filename}_{size}"
        if key in self._cache:
            return self._cache[key]

        path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(path):
            return None

        img = pygame.image.load(path).convert_alpha()
        if size:
            img = pygame.transform.smoothscale(img, size)
        self._cache[key] = img
        return img

    def get_cell_image(self, filename: str) -> pygame.Surface | None:
        """Load image scaled to cell size."""
        return self.load_image(filename, (CELL_SIZE, CELL_SIZE))

    def get_icon(self, filename: str, size: int = 32) -> pygame.Surface | None:
        """Load image scaled to a square icon size (for HUD)."""
        return self.load_image(filename, (size, size))


# Singleton
assets = AssetLoader()
