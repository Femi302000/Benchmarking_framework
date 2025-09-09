import cv2
import numpy as np
from itertools import combinations

# ---------- helpers ----------
def region_proposals(gray, min_area=600, max_area_ratio=0.4):
    # edges -> dilate -> contours -> bboxes
    v1, v2 = np.percentile(gray, [10, 90])
    low = max(0, 0.66*v1); high = min(255, 1.33*v2)
    edges = cv2.Canny(gray, int(low), int(high))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(edges, k, iterations=1)

    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    max_area = (h*w) * max_area_ratio
    boxes = []
    for c in cnts:
        x,y,bw,bh = cv2.boundingRect(c)
        area = bw*bh
        if area < min_area or area > max_area:
            continue
        # discard very skinny/tiny proposals
        if bw < 15 or bh < 15 or bw/bh > 6 or bh/bw > 6:
            continue
        boxes.append((x,y,bw,bh))
    return boxes

def extract_orb_desc(img, boxes, n_features=500):
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=5, scaleFactor=1.2, WTA_K=2)
    kps_list, des_list = [], []
    for (x,y,w,h) in boxes:
        roi = img[y:y+h, x:x+w]
        kps, des = orb.detectAndCompute(roi, None)
        kps_list.append(kps)
        des_list.append(des)
    return kps_list, des_list

def region_similarity(desA, desB, ratio=0.75):
    if desA is None or desB is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desA, desB, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) < 2: 
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good += 1
    denom = min(len(desA), len(desB)) if min(len(desA), len(desB))>0 else 1
    return good / denom

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a,b):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1

# ---------- main ----------
img = cv2.imread("/home/femi/Benchmarking_framework/scripts/yolo/yolo_outputs/scene_000_crop.png")
if img is None:
    raise FileNotFoundError("Put your file next to this script as image.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
boxes = region_proposals(gray)

# fallback: if too few, relax thresholds a bit
if len(boxes) < 4:
    boxes = region_proposals(gray, min_area=200, max_area_ratio=0.6)

kps_list, des_list = extract_orb_desc(gray, boxes)

# pairwise similarity -> cluster with Union-Find
N = len(boxes)
uf = UnionFind(N)

SIM_THRESHOLD = 0.30   # ~30% of keypoints agree → tune 0.25–0.45
for i, j in combinations(range(N), 2):
    s = region_similarity(des_list[i], des_list[j], ratio=0.75)
    if s >= SIM_THRESHOLD:
        uf.union(i, j)

# gather clusters
clusters = {}
for i in range(N):
    r = uf.find(i)
    clusters.setdefault(r, []).append(i)

# draw clusters with distinct colors
out = img.copy()
palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
           (128,0,255),(255,128,0),(0,128,255),(128,255,0)]
for k, idxs in clusters.items():
    if len(idxs) < 2:
        # skip singletons; they aren’t “similar group”
        continue
    color = palette[k % len(palette)]
    for idx in idxs:
        x,y,w,h = boxes[idx]
        cv2.rectangle(out, (x,y), (x+w, y+h), color, 2)
        cv2.putText(out, f"grp{k}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

cv2.imwrite("similar_clusters.png", out)
print(f"Found {sum(1 for v in clusters.values() if len(v)>1)} similar groups across {N} candidates.")
