import cv2
import numpy as np
from collections import Counter
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import tempfile
import os

def extract_colors(image, num_colors=10):
    # Resize image to speed up processing
    max_size = 200
    orig_width, orig_height = image.size
    scale = min(max_size/orig_width, max_size/orig_height)
    
    if scale < 1:
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert PIL image to NumPy array
    image_array = np.array(image)
    
    # Reshape and sample pixels
    pixels = image_array.reshape(-1, 3)
    pixel_positions = np.array([(i % image_array.shape[1], i // image_array.shape[1]) 
                               for i in range(len(pixels))])
    
    # Sample pixels if too many
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
        pixel_positions = pixel_positions[indices]

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=1)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Get sample positions for each cluster
    color_regions = []
    for i in range(num_colors):
        cluster_pixels = pixel_positions[labels == i]
        if len(cluster_pixels) > 0:
            # Get center point of cluster
            center_point = cluster_pixels.mean(axis=0)
            color_regions.append({
                'x': float(center_point[0] / image_array.shape[1]),  # Normalize to 0-1
                'y': float(center_point[1] / image_array.shape[0])   # Normalize to 0-1
            })
        else:
            color_regions.append({'x': 0.5, 'y': 0.5})  # Fallback center position

    return centers.astype(int).tolist(), color_regions

def extract_colors_from_video(video_path, num_colors=10, total_colors=30, max_frames=15, start_time=0, end_time=5):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video file")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            raise Exception("Invalid video FPS")

        # Convert time to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame
        
        # Sample frames evenly throughout the selected segment
        frames_to_sample = min(max_frames, total_frames)
        frame_indices = np.linspace(start_frame, end_frame-1, frames_to_sample, dtype=int)
        
        all_pixels = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (200, int(200 * frame.shape[0] / frame.shape[1])))
                pixels = frame.reshape(-1, 3)
                all_pixels.extend(pixels)
        
        cap.release()
        
        if not all_pixels:
            raise Exception("No frames could be extracted from video")

        all_pixels = np.array(all_pixels)
        
        if len(all_pixels) > 10000:
            indices = np.random.choice(len(all_pixels), 10000, replace=False)
            all_pixels = all_pixels[indices]
        
        kmeans = KMeans(n_clusters=total_colors, n_init=1)
        kmeans.fit(all_pixels)
        all_colors = kmeans.cluster_centers_.astype(int).tolist()
        
        return all_colors[:num_colors], all_colors[num_colors:], end_time - start_time

    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

class UploadMediaView(APIView):
    def post(self, request):
        if "file" not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)

        try:
            file = request.FILES["file"]
            num_colors = int(request.POST.get('numColors', 4))
            total_colors = 30  # Total colors to extract for shuffling
            start_time = float(request.POST.get('startTime', 0))
            end_time = float(request.POST.get('endTime', 5))
            
            is_video = file.content_type.startswith('video/')
            
            if is_video:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    for chunk in file.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                
                try:
                    colors, reserve_colors, duration = extract_colors_from_video(
                        tmp_path, 
                        num_colors=num_colors,
                        total_colors=total_colors,
                        start_time=start_time,
                        end_time=end_time
                    )
                    return JsonResponse({
                        "colors": colors,
                        "reserveColors": reserve_colors,
                        "duration": duration,
                        "isVideo": True
                    })
                except Exception as e:
                    return JsonResponse({"error": str(e)}, status=400)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                # Handle image as before
                image = Image.open(file)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                all_colors, all_regions = extract_colors(image, num_colors=total_colors)
                
                return JsonResponse({
                    "colors": all_colors[:num_colors],
                    "regions": all_regions[:num_colors],
                    "reserveColors": all_colors[num_colors:],
                    "reserveRegions": all_regions[num_colors:],
                    "isVideo": False
                })
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
