from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from skimage.morphology import skeletonize
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
        
        # 2. Skeletonization
        # skimage skeletonize expects boolean or 0/1 array.
        binary_bool = binary > 0
        skeleton = skeletonize(binary_bool)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        
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

        for label_id in range(1, num_labels): # Skip background (0)
            if stats[label_id, cv2.CC_STAT_AREA] < min_segment_length:
                continue
            
            valid_centroids.append(centroids[label_id])
            valid_label_ids.append(label_id)

        # 6. Proximity Clustering (DBSCAN)
        # Epsilon is max distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples=1 ensures every point is part of a cluster (no noise points thrown away if possible, or just own cluster)
        group_ids = {}
        if valid_centroids:
            from sklearn.cluster import DBSCAN
            # eps=50 pixels seems reasonable for "nearby" strokes? Adjustable.
            # Convert centroids to np array
            X = np.array(valid_centroids)
            # eps 30-50 depends on image resolution. Let's pick 35.
            clustering = DBSCAN(eps=35, min_samples=1).fit(X)
            
            for idx, label_id in enumerate(valid_label_ids):
                group_ids[label_id] = int(clustering.labels_[idx])
        
        height, width = segments_skeleton.shape
        
        for label_id in valid_label_ids:
            # Extract coordinates for this segment
            # np.where returns (row_indices, col_indices) -> (y, x)
            ys, xs = np.where(labels == label_id)
            
            # Sort points to form a continuous path
            # 1. Build Adjacency Graph within the segment
            pixel_set = set(zip(xs, ys))
            adj = { (x,y): [] for x,y in zip(xs, ys) }
            
            for px, py in zip(xs, ys):
                # Check 8-neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = px + dx, py + dy
                        if (nx, ny) in pixel_set:
                            adj[(px, py)].append((nx, ny))
                            
            # 2. Find Endpoints (Degree 1) or Start anywhere if Loop
            start_node = None
            for node, neighbors in adj.items():
                if len(neighbors) == 1:
                    start_node = node
                    break
            
            if start_node is None and len(pixel_set) > 0:
                # Loop or isolated point or internal weirdness, pick random
                 start_node = list(pixel_set)[0]
                 
            # 3. Traverse (DFS/BFS)
            visited = set()
            ordered_points = []
            stack = [start_node]
            
            # Simple traversal for simple paths
            while stack:
                curr = stack.pop()
                if curr in visited: continue
                visited.add(curr)
                
                # Get width for this point
                cx, cy = curr
                w = dist_transform[cy, cx] * 2.0
                ordered_points.append({"x": int(cx), "y": int(cy), "w": float(w)})
                
                # Add neighbors
                # Sort neighbors to prefer closest/direction?
                # For skeleton, simple neighbor follow is usually okay
                for n in adj[curr]:
                    if n not in visited:
                        stack.append(n)
                        
            # If we have disjoint components in one label (shouldn't happen with connectedComponents),
            # we only grabbed one. But connectedComponents guarantees connectivity.
            # However, logic above might miss branches if it's not a simple line.
            # But we cut junctions, so segments should be simple lines.
            
            points = ordered_points
            
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
