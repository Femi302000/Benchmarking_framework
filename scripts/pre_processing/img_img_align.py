#!/usr/bin/env python3
# Align a partial image to a full aircraft reference using ORB+RANSAC, then overlay.
import cv2, numpy as np, os

# ====== CONFIG ======
PARTIAL_IMG = "/home/femi/Benchmarking_framework/scripts/yolo/birdview_seg.png"       # the cut/partial image
FULL_IMG    = "/home/femi/Downloads/download.png"      # full aircraft image
OUT_WARPED  = "partial_aligned_to_full.png"
OUT_BLEND   = "overlay_blend.png"
ALPHA       = 0.6   # partial weight in the final blend
# If you also have a mask for the partial (white=keep), set it here; else leave None
PARTIAL_MASK = None  # e.g. "/path/to/partial_mask.png"
# ====================

def load_gray_color(path):
    assert os.path.isfile(path), f"Not found: {path}"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def main():
    full_bgr, full_gray = load_gray_color(FULL_IMG)
    part_bgr, part_gray = load_gray_color(PARTIAL_IMG)

    # 1) Detect + match features
    orb = cv2.ORB_create(nfeatures=5000)
    k1, d1 = orb.detectAndCompute(part_gray, None)
    k2, d2 = orb.detectAndCompute(full_gray, None)

    if d1 is None or d2 is None:
        raise RuntimeError("Not enough features. Try using SIFT (requires non-free build) or provide better images.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(d1, d2, k=2)

    # Lowe’s ratio test
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError(f"Too few good matches ({len(good)}). Try lowering ratio threshold or using more texture.")

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # 2) Homography (RANSAC)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("Homography failed. Images may be too different or lack overlap.")

    # 3) Warp partial into full-image coordinates
    Hh, Hw = full_bgr.shape[:2]
    warped = cv2.warpPerspective(part_bgr, H, (Hw, Hh))

    # Optional: warp mask if you have one
    if PARTIAL_MASK and os.path.isfile(PARTIAL_MASK):
        m = cv2.imread(PARTIAL_MASK, cv2.IMREAD_GRAYSCALE)
        warped_mask = cv2.warpPerspective(m, H, (Hw, Hh))
        warped_mask = (warped_mask > 127).astype(np.uint8) * 255
    else:
        # infer mask from non-black pixels in warped partial
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_mask = cv2.threshold(gray_w, 1, 255, cv2.THRESH_BINARY)

    cv2.imwrite(OUT_WARPED, warped)

    # 4) Seamless-ish composite (alpha blend where warped exists)
    mask3 = cv2.merge([warped_mask]*3)
    overlay = full_bgr.copy()
    # alpha blend only on mask region
    blended = np.where(mask3>0, (ALPHA*warped + (1-ALPHA)*full_bgr).astype(np.uint8), full_bgr)
    cv2.imwrite(OUT_BLEND, blended)

    print(f"Saved:\n  warped partial -> {OUT_WARPED}\n  blended overlay -> {OUT_BLEND}")

if __name__ == "__main__":
    main()
