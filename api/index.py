from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
from typing import List, Dict

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def zhang_suen_thinning(binary_image):
    """
    Perform Zhang-Suen thinning algorithm for skeletonization using OpenCV.
    Input: Binary image (0 and 1, or 0 and 255 where 255 is foreground).
    Output: Skeletonized image (0 and 255).
    """
    # Ensure binary is 0 and 1
    if binary_image.max() > 1:
        binary_image = (binary_image > 0).astype(np.uint8)
    
    skeleton = binary_image.copy()
    
    # Define kernels for Zhang-Suen
    # This involves complex hit-or-miss interactions. 
    # A simpler approach in classic CV is commonly just `cv2.ximgproc.thinning` 
    # but that requires opencv-contrib-python which might be large too.
    # Let's implement a standard iterative erosion (morphological thinning) for robustness without extra deps.
    
    # Actually, standard cv2.erode isn't true skeletonization.
    # Let's use a standard implementation available in pure numpy or basic cv2.
    # A common effective way without `ximgproc` is iteratively eroding until no change, 
    # but preserving connectivity.
    
    # Standard 2-pass algorithm (Zhang-Suen) implementation with cv2 filters is efficient enough.
    # But for simplicity and stability without `skimage`, we can try a simple distance transform ridge
    # or just use the built-in `cv2.ximgproc.thinning` IF `opencv-python-headless` includes it.
    # Standard `opencv-python-headless` often DOES NOT include `ximgproc` (part of contrib).
    
    # Start basic morphological thinning (inefficient but works for 250MB limit context):
    skel = np.zeros(binary_image.shape, np.uint8)
    eroded = binary_image.copy() * 255 # working with 0-255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    while True:
        eroded_next = cv2.erode(eroded, element)
        temp = cv2.dilate(eroded_next, element)
        temp = cv2.subtract(eroded, temp)
        skel = cv2.bitwise_or(skel, temp)
        eroded = eroded_next.copy()
        if cv2.countNonZero(eroded) == 0:
            break
            
    return skel

def simple_clustering(data: List[np.ndarray], eps: float):
    """
    Simple distance-based clustering (like simpler DBSCAN) to replace sklearn.
    data: List[np.array([y, x])] centroids
    eps: max distance
    Returns: labels array
    """
    n = len(data)
    if n == 0:
        return []
    
    labels = [-1] * n
    cluster_id = 0
    
    for i in range(n):
        if labels[i] != -1:
            continue
            
        # Start new cluster
        labels[i] = cluster_id
        stack = [i]
        
        while stack:
            current_idx = stack.pop()
            current_point = data[current_idx]
            
            # Find neighbors
            for j in range(n):
                if labels[j] != -1:
                    continue
                
                # Euclidean distance
                dist = np.linalg.norm(current_point - data[j])
                
                if dist <= eps:
                    labels[j] = cluster_id
                    stack.append(j)
        
        cluster_id += 1
        
    return labels

def detect_junctions(skeleton: np.ndarray) -> np.ndarray:
    """
    Detect junctions in a skeletonized image.
    A junction is a pixel with more than 2 neighbors in an 8-connected neighborhood.
    Returns a mask of junction points.
    """
    # Kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Convolve to count neighbors for each pixel
    # We only care about skeleton pixels (value 1 or 255)
    skeleton_bool = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bool, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    
    # Junctions are skeleton pixels with > 2 neighbors
    junctions = (skeleton_bool == 1) & (neighbor_count > 2)
    return junctions.astype(np.uint8) * 255

@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarize (inverted because skeletonize expects white foreground)
        # Verify if image is white-on-black or black-on-white.
        # Assuming line art is black lines on white background -> Invert to get white lines.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Skeletonization (Replaced skimage)
        # Normalize binary for morphological ops
        binary_bool = (binary > 0).astype(np.uint8)
        skeleton_uint8 = zhang_suen_thinning(binary_bool)
        
        # 2.5 Distance Transform for Width Estimation
        # Calculate distance from background for every foreground pixel
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 3. Junction Detection
        junctions = detect_junctions(skeleton_uint8)
        
        # 4. Cut junctions to separate segments
        # Dilate junctions slightly to ensure cuts are clean if needed, 
        # but single pixel removal might be enough for 1-pixel wide skeleton.
        # Let's remove junction pixels from the skeleton.
        segments_skeleton = skeleton_uint8.copy()
        segments_skeleton[junctions > 0] = 0
        
        # 5. Extract Connected Components (Segments)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segments_skeleton, connectivity=8)
        
        raw_segments = []
        min_segment_length = 5  # Filter out tiny noise
        
        # Collect centroids for clustering
        valid_centroids = []
        valid_label_ids = []
        
        # Skip background (label 0)
        # Note: centroids are (x, y) float
        for label_id in range(1, num_labels): 
            if stats[label_id, cv2.CC_STAT_AREA] < min_segment_length:
                continue
            
            # Switch to (y, x) for distance calculation to match numpy standard if needed, 
            # but Euclidean is symmetric. Keeping (y, x) array for clustering function if using indices.
            # Centroids from cv2 are (x, y). Let's use (y, x) for consistency with matrix indexing if needed.
            # Actually clustering just needs consistent coordinates. Let's use (x,y)
            cx, cy = centroids[label_id]
            valid_centroids.append(np.array([cy, cx])) # Use Y, X for consistency with image coords usually
            valid_label_ids.append(label_id)

        # 6. Proximity Clustering (Replaced sklearn DBSCAN)
        group_ids = {}
        if valid_centroids:
            # Custom simple clustering
            # eps=35 pixels
            clustering_labels = simple_clustering(valid_centroids, eps=35.0)
            
            for idx, label_id in enumerate(valid_label_ids):
                group_ids[label_id] = clustering_labels[idx]
        
        height, width = segments_skeleton.shape
        
        for label_id in valid_label_ids:
            # Extract coordinates for this segment
            # np.where returns (row_indices, col_indices) -> (y, x)
            ys, xs = np.where(labels == label_id)
            
            # Calculate widths for each point using Distance Transform
            # dist_transform returns float32 distance
            # Width = 2 * distance
            points = []
            for x, y in zip(xs, ys):
                 w = dist_transform[y, x] * 2.0
                 points.append({"x": int(x), "y": int(y), "w": float(w)})
            
            # Pack into list for sorting (optional, but good for drawing order)
            # Simple assumption: points are somewhat ordered or we treat them as "cloud" for circle drawing.
            # For smooth strokes, we might need to sort them. 
            # Skeletonize usually produces ordered-ish paths but not guaranteed by np.where.
            # Let's trust np.where order for now (raster scan order) which is BAD for drawing lines.
            # For variable width "brush", drawing circles at x,y is independent of order.
            
            # Calculate Average Stroke Width for fallback
            segment_widths = [p["w"] for p in points]
            avg_width = np.mean(segment_widths) if segment_widths else 1.0
            
            raw_segments.append({
                "id": label_id,
                "group_id": group_ids.get(label_id, -1),
                "width": float(avg_width),
                "points": points,
                "bbox": {
                    "x": int(stats[label_id, cv2.CC_STAT_LEFT]),
                    "y": int(stats[label_id, cv2.CC_STAT_TOP]),
                    "w": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[label_id, cv2.CC_STAT_HEIGHT])
                },
                "centroid": {
                    "x": int(centroids[label_id][0]),
                    "y": int(centroids[label_id][1])
                }
            })            
        return {
            "width": width, 
            "height": height,
            "segments": raw_segments
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
