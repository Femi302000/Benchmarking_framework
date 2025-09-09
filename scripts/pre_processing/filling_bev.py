import numpy as np
from skimage import io, color, filters, morphology, measure, util
from skimage.segmentation import flood_fill

def fill_bev_image_sk(
    in_path: str,
    out_mask_path: str = "bev_filled_mask.png",
    out_overlay_path: str = "bev_overlay.png",
    morph_radius: int = 3,
    keep_largest: bool = True,
    invert: bool = False,  # set True if aircraft is dark on light background
):
    img = io.imread(in_path)
    gray = color.rgb2gray(img) if img.ndim == 3 else util.img_as_float(img)

    # Threshold (Otsu)
    t = filters.threshold_otsu(gray)
    bw = gray > t
    if invert:
        bw = ~bw

    # Morph closing
    se = morphology.disk(morph_radius)
    closed = morphology.closing(bw, se)

    # Fill holes
    filled = morphology.remove_small_holes(closed, area_threshold=10_000)  # adjust if needed

    # Keep largest component
    if keep_largest:
        labeled = measure.label(filled, connectivity=2)
        props = measure.regionprops(labeled)
        if props:
            largest_label = max(props, key=lambda p: p.area).label
            filled = labeled == largest_label

    # Save binary mask
    io.imsave(out_mask_path, (filled * 255).astype(np.uint8))

    # Overlay
    if img.ndim == 2:
        base = np.dstack([img, img, img])
    else:
        base = img.copy()
    overlay = base.copy()
    overlay[filled] = [255, 0, 0]
    alpha = 0.5
    blended = (base * (1 - alpha) + overlay * alpha).astype(base.dtype)
    io.imsave(out_overlay_path, blended)

# ---- run it ----
fill_bev_image_sk("/home/femi/Benchmarking_framework/scripts/keypoints/birdview.png", morph_radius=5, invert=False,keep_largest=False)
