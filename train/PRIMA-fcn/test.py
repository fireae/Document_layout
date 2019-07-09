import os, cv2
import numpy as np
from PIL import Image

def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image
    palette.load()
    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    # the 0 above means turn OFF dithering

    # Later versions of Pillow (4.x) rename _makeself to _new
    try:
        return silf._new(im)
    except AttributeError:
        return silf._makeself(im)


palette = [0,0,0, 64,128,64, 128,0,192, 192,128,0, 64,128,0,
            0,0,128, 128,0,64, 192,0,64, 64,128,192, 128,192,192,
            128,64,64]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

palimage = Image.new('P', (16, 16))
palimage.putpalette(palette)
oldimage = Image.open("00000195.png")
newimage = quantizetopalette(oldimage, palimage, dither=False)