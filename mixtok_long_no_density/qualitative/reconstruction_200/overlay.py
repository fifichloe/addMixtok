"""Draw the location mask for reconstruction comparison figures."""

import numpy as np
from PIL import Image, ImageDraw


WHITE_OPACITY = 0.55
CELL_SIZE = 16


def paint_selected_grid(target: Image.Image, mask) -> Image.Image:
    """Keep the original crop visible beneath selected white 16x16 cells."""
    if target.size != (256, 256):
        raise ValueError(f"Expected a 256x256 target crop, got {target.size}")
    selected = np.asarray(mask, dtype=bool).reshape(16, 16)
    image = target.convert('RGB')
    white = Image.blend(image, Image.new('RGB', image.size, 'white'), WHITE_OPACITY)
    mask_image = Image.fromarray((selected.astype('uint8') * 255), 'L')
    mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    image = Image.composite(white, image, mask_image)

    draw = ImageDraw.Draw(image)
    for position in range(0, 257, CELL_SIZE):
        draw.line((position, 0, position, 255), fill=(225, 225, 225), width=1)
        draw.line((0, position, 255, position), fill=(225, 225, 225), width=1)
    for y, x in np.argwhere(selected):
        left, top = int(x) * CELL_SIZE, int(y) * CELL_SIZE
        draw.rectangle((left, top, left + CELL_SIZE - 1, top + CELL_SIZE - 1),
                       outline=(75, 75, 75), width=1)
    return image
