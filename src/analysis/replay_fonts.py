"""Portable Pillow fonts for generated simulation images."""
from PIL import ImageFont


def replay_font(size):
    """Prefer system TrueType fonts, then Pillow's bundled default."""
    for name in (
        'C:/Windows/Fonts/segoeui.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        'DejaVuSans.ttf',
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
