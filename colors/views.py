import cv2
import numpy as np
from collections import Counter
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import tempfile
import os
import io
from rest_framework.response import Response
from django.core.exceptions import ValidationError
import logging

logger = logging.getLogger(__name__)

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

def optimize_image_for_upload(image, max_dimension=1200):
    """Resize and optimize image for upload if it's too large"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check if resize is needed
        w, h = image.size
        if max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save optimized image to bytes
        img_io = io.BytesIO()
        image.save(img_io, format='PNG', optimize=True, quality=85)
        img_io.seek(0)
        
        return img_io
    except Exception as e:
        print(f"Error optimizing image: {str(e)}")
        return None

class UploadMediaView(APIView):
    def post(self, request):
        if "file" not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)

        try:
            file = request.FILES["file"]
            
            # Check file size
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                return JsonResponse({"error": "File too large. Maximum size is 10MB"}, status=400)
            
            num_colors = int(request.POST.get('numColors', 4))
            total_colors = 30  # Total colors to extract for shuffling
            
            if file.content_type.startswith('video/'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    for chunk in file.chunks():
                        tmp_file.write(chunk)
                    tmp_path = tmp_file.name
                
                try:
                    colors, reserve_colors, duration = extract_colors_from_video(
                        tmp_path, 
                        num_colors=num_colors,
                        total_colors=total_colors
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
                # Open and optimize image
                image = Image.open(file)
                optimized_io = optimize_image_for_upload(image)
                if optimized_io is None:
                    return JsonResponse({"error": "Failed to process image"}, status=400)
                
                # Store optimized image in session or cache for later use
                request.session['optimized_image'] = optimized_io.getvalue()
                
                # Extract colors from optimized image
                image = Image.open(optimized_io)
                all_colors, all_regions = extract_colors(image, num_colors=total_colors)
                
                return JsonResponse({
                    "colors": all_colors[:num_colors],
                    "regions": all_regions[:num_colors],
                    "reserveColors": all_colors[num_colors:],
                    "reserveRegions": all_regions[num_colors:],
                    "isVideo": False
                })
                
        except Exception as e:
            print(f"Error in upload: {str(e)}")
            return JsonResponse({"error": str(e)}, status=400)

def resize_image_for_processing(img, max_dimension=1200):
    """Resize image while maintaining aspect ratio if it's too large"""
    w, h = img.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def validate_image(file):
    """Validate image file"""
    if not file.content_type.startswith('image/'):
        raise ValidationError('Invalid file type. Please upload an image.')
    
    # Check file size (10MB limit)
    if file.size > 10 * 1024 * 1024:
        raise ValidationError('File too large. Maximum size is 10MB')
    
    try:
        img = Image.open(file)
        img.verify()  # Verify it's a valid image
        
        # Reset file pointer
        file.seek(0)
        
        # Check dimensions
        img = Image.open(file)
        if img.size[0] * img.size[1] > 5000 * 5000:
            raise ValidationError('Image dimensions too large. Maximum size is 25MP')
            
        return img
    except Exception as e:
        raise ValidationError(f'Invalid image file: {str(e)}')

@api_view(['POST'])
def posterize_image(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        pixel_size = int(request.POST.get('pixelSize', 8))
        num_colors = int(request.POST.get('numColors', 8))
        
        # Use optimized image if available in session
        if 'optimized_image' in request.session:
            img_data = request.session['optimized_image']
            original_img = Image.open(io.BytesIO(img_data))
        else:
            # Fallback to original file
            original_img = Image.open(file)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
        
        # Store original dimensions
        original_w, original_h = original_img.size
        
        # Resize for processing if image is too large
        img = resize_image_for_processing(original_img)
        w, h = img.size
        
        # Calculate resize ratios
        w_ratio = original_w / w
        h_ratio = original_h / h
        
        # Adjust pixel size based on ratio
        adjusted_pixel_size = max(2, int(pixel_size / min(w_ratio, h_ratio)))
        
        # Create posterized version
        small_w = max(1, w // adjusted_pixel_size)
        small_h = max(1, h // adjusted_pixel_size)
        img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Convert to numpy array for color quantization
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
        
        # Convert back to uint8
        palette = np.uint8(palette)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(img_array.shape)
        
        # Convert back to PIL Image
        posterized_img = Image.fromarray(quantized)
        
        # Resize back to original dimensions
        posterized_img = posterized_img.resize((original_w, original_h), Image.Resampling.NEAREST)
        
        # Save to bytes
        img_io = io.BytesIO()
        posterized_img.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        
        return HttpResponse(img_io, content_type='image/png')
        
    except Exception as e:
        print(f"Error in posterize_image: {str(e)}")
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def posterize_svg(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        pixel_size = int(request.POST.get('pixelSize', 8))
        num_colors = int(request.POST.get('numColors', 8))
        
        # Open and process image
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for processing if image is too large
        img = resize_image_for_processing(img, max_dimension=400)  # Further reduced max dimension
        w, h = img.size
        
        # Calculate pixelated dimensions
        small_w = max(1, w // pixel_size)
        small_h = max(1, h // pixel_size)
        
        # Resize for pixelation
        small_img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Convert to numpy array and process colors
        img_array = np.array(small_img)
        pixels = img_array.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Perform k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
        
        # Convert back to uint8
        palette = np.uint8(palette)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(img_array.shape)
        
        # Initialize SVG with minimal content
        svg_parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
        
        # Process image row by row and combine identical adjacent pixels
        for y in range(quantized.shape[0]):
            run_start = 0
            current_color = tuple(quantized[y, 0])
            
            for x in range(1, quantized.shape[1]):
                pixel_color = tuple(quantized[y, x])
                if pixel_color != current_color:
                    # End of a run of identical colors
                    if run_start < x - 1:  # If run is longer than 1 pixel
                        svg_parts.append(
                            f'<rect x="{run_start*pixel_size}" y="{y*pixel_size}" '
                            f'width="{(x-run_start)*pixel_size}" height="{pixel_size}" '
                            f'fill="rgb{current_color}"/>'
                        )
                    run_start = x
                    current_color = pixel_color
            
            # Handle the last run in the row
            if run_start < quantized.shape[1]:
                svg_parts.append(
                    f'<rect x="{run_start*pixel_size}" y="{y*pixel_size}" '
                    f'width="{(quantized.shape[1]-run_start)*pixel_size}" '
                    f'height="{pixel_size}" fill="rgb{current_color}"/>'
                )
        
        svg_parts.append('</svg>')
        svg_string = ''.join(svg_parts)
        
        # Create response with proper headers
        response = HttpResponse(
            content=svg_string,
            content_type='image/svg+xml; charset=utf-8'
        )
        response['Content-Disposition'] = 'attachment; filename="posterized-image.svg"'
        return response
        
    except Exception as e:
        print(f"Error in posterize_svg: {str(e)}")
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def convert_to_line_art(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        threshold = int(request.POST.get('threshold', 127))
        blur_radius = int(request.POST.get('blurRadius', 0))
        
        # Open and process image
        original_img = Image.open(file)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Store original dimensions
        original_w, original_h = original_img.size
        
        # Resize for processing if image is too large
        img = resize_image_for_processing(original_img)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        if blur_radius > 0:
            gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Apply edge detection
        edges = cv2.Canny(gray, threshold/2, threshold)
        
        # Invert the image
        edges = 255 - edges
        
        # Convert back to PIL Image
        line_art = Image.fromarray(edges)
        
        # Resize back to original dimensions
        line_art = line_art.resize((original_w, original_h), Image.Resampling.LANCZOS)
        
        # Save to bytes with optimization
        img_io = io.BytesIO()
        line_art.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        
        return HttpResponse(img_io, content_type='image/png')
        
    except Exception as e:
        print(f"Error in convert_to_line_art: {str(e)}")
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def convert_to_line_art_svg(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        threshold = int(request.POST.get('threshold', 127))
        blur_radius = int(request.POST.get('blurRadius', 0))
        
        # Open and process image
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get dimensions
        w, h = img.size
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Optional: Apply Gaussian blur to reduce noise
        if blur_radius > 0:
            gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Apply edge detection
        edges = cv2.Canny(gray, threshold/2, threshold)
        
        # Create SVG content
        svg_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        svg_content.append(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">')
        svg_content.append(f'<rect width="{w}" height="{h}" fill="white"/>')
        
        # Convert edges to SVG paths
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 1:
                path = "M"
                for point in contour:
                    x, y = point[0]
                    path += f" {x},{y}"
                path += "Z"
                svg_content.append(f'<path d="{path}" stroke="black" fill="none" stroke-width="1"/>')
        
        svg_content.append('</svg>')
        
        # Join all lines
        svg_string = '\n'.join(svg_content)
        
        # Return SVG response
        response = HttpResponse(svg_string, content_type='image/svg+xml')
        response['Content-Disposition'] = 'attachment; filename="line-art.svg"'
        return response
        
    except Exception as e:
        print(f"Error in convert_to_line_art_svg: {str(e)}")
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def blend_art(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        pixel_size = int(request.POST.get('pixelSize', 8))
        num_colors = int(request.POST.get('numColors', 8))
        threshold = int(request.POST.get('threshold', 127))
        blur_radius = int(request.POST.get('blurRadius', 0))
        
        # Open and process image
        original_img = Image.open(file)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Store original dimensions
        original_w, original_h = original_img.size
        
        # Resize for processing
        img = resize_image_for_processing(original_img)
        w, h = img.size
        
        # Calculate resize ratios
        w_ratio = original_w / w
        h_ratio = original_h / h
        
        # Adjust pixel size based on ratio
        adjusted_pixel_size = max(2, int(pixel_size / min(w_ratio, h_ratio)))
        
        # Process at reduced size
        small_w = max(1, w // adjusted_pixel_size)
        small_h = max(1, h // adjusted_pixel_size)
        
        # Create posterized version
        posterized = img.resize((small_w, small_h), Image.Resampling.NEAREST)
        posterized = posterized.resize((w, h), Image.Resampling.NEAREST)
        
        # Process posterized image
        img_array = np.array(posterized)
        pixels = img_array.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
        
        palette = np.uint8(palette)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(img_array.shape)
        
        # Create line art
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        if blur_radius > 0:
            gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        edges = cv2.Canny(gray, threshold/2, threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend
        mask = edges_rgb == 255
        result = quantized.copy()
        result[mask] = 0
        
        # Convert to PIL and resize to original dimensions
        blended_img = Image.fromarray(result)
        blended_img = blended_img.resize((original_w, original_h), Image.Resampling.NEAREST)
        
        # Save with optimization
        img_io = io.BytesIO()
        blended_img.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        
        return HttpResponse(img_io, content_type='image/png')
        
    except Exception as e:
        print(f"Error in blend_art: {str(e)}")
        return Response({'error': str(e)}, status=500)
