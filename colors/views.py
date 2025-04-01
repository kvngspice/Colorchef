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
import math

logger = logging.getLogger(__name__)

def extract_colors(image, num_colors=10):
    try:
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
        
        # Ensure image is in RGB format
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be in RGB format")
        
        # Reshape and sample pixels
        pixels = image_array.reshape(-1, 3)
        pixel_positions = np.array([(i % image_array.shape[1], i // image_array.shape[1]) 
                                  for i in range(len(pixels))])
        
        # Sample pixels if too many
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
            pixel_positions = pixel_positions[indices]

        # Use KMeans clustering with error handling
        try:
            kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), n_init=1)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
        except Exception as e:
            print(f"KMeans error: {str(e)}")
            # Fallback to simple color averaging if KMeans fails
            centers = np.mean(pixels.reshape(-1, 3), axis=0).reshape(1, 3)
            labels = np.zeros(len(pixels))
            num_colors = 1

        # Get sample positions for each cluster
        color_regions = []
        for i in range(min(num_colors, len(centers))):
            cluster_pixels = pixel_positions[labels == i]
            if len(cluster_pixels) > 0:
                # Get center point of cluster
                center_point = cluster_pixels.mean(axis=0)
                color_regions.append({
                    'x': float(center_point[0] / image_array.shape[1]),
                    'y': float(center_point[1] / image_array.shape[0])
                })
            else:
                color_regions.append({'x': 0.5, 'y': 0.5})

        return centers.astype(int).tolist(), color_regions

    except Exception as e:
        print(f"Error in extract_colors: {str(e)}")
        # Return fallback colors if everything fails
        fallback_colors = [[128, 128, 128]]  # Gray
        fallback_regions = [{'x': 0.5, 'y': 0.5}]
        return fallback_colors * num_colors, fallback_regions * num_colors

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
                # Open and process image
                image = Image.open(file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize for processing if needed
                if image.size[0] * image.size[1] > 1000000:  # If larger than 1MP
                    process_w, process_h = get_optimal_dimensions(image.size[0], image.size[1])
                    image = image.resize((process_w, process_h), Image.Resampling.LANCZOS)
                
                # Extract colors directly from the processed image
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

def get_optimal_dimensions(width, height, max_pixels=1000000):  # 1MP default limit
    """Calculate optimal dimensions while maintaining aspect ratio"""
    pixels = width * height
    if pixels <= max_pixels:
        return width, height
        
    ratio = math.sqrt(max_pixels / pixels)
    return int(width * ratio), int(height * ratio)

@api_view(['POST'])
def posterize_image(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        pixel_size = int(request.POST.get('pixelSize', 8))
        num_colors = int(request.POST.get('numColors', 8))
        style = request.POST.get('style', 'classic')
        
        # Open and process image
        original_img = Image.open(file)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')
        
        # Process based on style
        if style == 'cartoon':
            result_img = create_cartoon_effect(original_img, pixel_size, num_colors)
        else:  # classic
            result_img = create_classic_posterize(original_img, pixel_size, num_colors)
        
        # Save output
        img_io = io.BytesIO()
        result_img.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        
        return HttpResponse(img_io, content_type='image/png')
        
    except Exception as e:
        print(f"Error in posterize_image: {str(e)}")
        return Response({'error': str(e)}, status=500)

def create_cartoon_effect(img, pixel_size, num_colors):
    """Creates a flat, cartoon-like effect with smooth outlines and vibrant colors"""
    # Resize for processing
    img = resize_image_for_processing(img)
    img_array = np.array(img)
    
    # Apply bilateral filter to smooth while preserving edges
    bilateral = cv2.bilateralFilter(img_array, 15, 80, 80)
    
    # Convert to LAB color space for better color separation
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    
    # Enhance color channels
    a = cv2.add(a, 2)
    b = cv2.add(b, 2)
    lab_enhanced = cv2.merge([l, a, b])
    
    # Color quantization in LAB space
    pixels = lab_enhanced.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # Color clustering with enhanced saturation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .95)
    _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags=cv2.KMEANS_PP_CENTERS)
    
    # Convert back to image space
    palette = np.uint8(palette)
    quantized = palette[labels.flatten()].reshape(lab_enhanced.shape)
    quantized = cv2.cvtColor(quantized, cv2.COLOR_LAB2RGB)
    
    # Enhance saturation
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)
    v = cv2.add(v, 10)
    quantized = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
    
    # Multi-stage edge detection for smoother lines
    gray = cv2.cvtColor(bilateral, cv2.COLOR_RGB2GRAY)
    
    # Get dark regions
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Detect edges using multiple methods
    edges_canny = cv2.Canny(gray, 50, 150)
    
    # Detect significant edges using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobelx**2 + sobely**2)
    edges_sobel = np.uint8(np.clip(edges_sobel * 255 / edges_sobel.max(), 0, 255))
    
    # Combine edges
    edges = cv2.addWeighted(edges_canny, 0.7, edges_sobel, 0.3, 0)
    edges = cv2.bitwise_and(edges, dark_mask)
    
    # Smooth and clean edges
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.GaussianBlur(edges, (3, 3), 0.5)
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    # Process contours for smoother lines
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    edges = np.zeros_like(edges)
    
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            # Smooth the contour
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw smooth lines
            cv2.drawContours(edges, [approx], -1, 255, 1, cv2.LINE_AA)
    
    # Final edge smoothing
    edges = cv2.GaussianBlur(edges, (3, 3), 0.5)
    
    # Create edge mask
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_rgb = cv2.bitwise_not(edges_rgb)
    
    # Combine quantized image with edges
    result = cv2.bitwise_and(quantized, edges_rgb)
    
    # Final color enhancement
    lab_result = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_result)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(result)

def create_classic_posterize(img, pixel_size, num_colors):
    """The original posterization effect"""
    # Store original dimensions
    original_w, original_h = img.size
    
    # Calculate optimal processing dimensions based on image size
    if original_w * original_h > 1000000:  # If larger than 1MP
        process_w, process_h = get_optimal_dimensions(original_w, original_h)
        img = img.resize((process_w, process_h), Image.Resampling.LANCZOS)
    else:
        process_w, process_h = original_w, original_h
    
    # Adjust pixel size based on resize ratio
    scale_factor = min(original_w / process_w, original_h / process_h)
    adjusted_pixel_size = max(2, int(pixel_size / scale_factor))
    
    # Create posterized version
    small_w = max(1, process_w // adjusted_pixel_size)
    small_h = max(1, process_h // adjusted_pixel_size)
    img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
    
    # Process colors
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
    
    # Convert back to image
    palette = np.uint8(palette)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img_array.shape)
    posterized_img = Image.fromarray(quantized)
    
    # Resize back to original dimensions
    posterized_img = posterized_img.resize((original_w, original_h), Image.Resampling.NEAREST)
    
    return posterized_img

@api_view(['POST'])
def posterize_svg(request):
    if 'file' not in request.FILES:
        return Response({'error': 'No file uploaded'}, status=400)
    
    try:
        file = request.FILES['file']
        pixel_size = int(request.POST.get('pixelSize', 8))
        num_colors = int(request.POST.get('numColors', 8))
        style = request.POST.get('style', 'classic')
        
        # Open and process image
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if style == 'cartoon':
            img_array = np.array(img)
            bilateral = cv2.bilateralFilter(img_array, 15, 80, 80)
            
            # Convert to LAB color space for better color separation
            lab_image = cv2.cvtColor(bilateral, cv2.COLOR_RGB2LAB)
            pixels = lab_image.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # Color clustering with better parameters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .95)
            _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags=cv2.KMEANS_PP_CENTERS)
            
            # Start SVG
            w, h = img.size
            svg_parts = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
                f'<rect width="{w}" height="{h}" fill="white"/>'
            ]
            
            # Convert colors to RGB and prepare masks
            rgb_palette = cv2.cvtColor(np.uint8([palette]), cv2.COLOR_LAB2RGB)[0]
            labels_2d = labels.reshape(img.size[::-1])
            
            # Process colors from darkest to lightest for better layering
            color_brightnesses = [np.mean(color) for color in rgb_palette]
            color_indices = np.argsort(color_brightnesses)
            
            # Add color regions with smart grouping
            for color_idx in color_indices:
                color = tuple(map(int, rgb_palette[color_idx]))
                mask = (labels_2d == color_idx).astype(np.uint8) * 255
                
                # Dilate mask slightly to prevent gaps
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                # Find contours with hierarchy for better nesting
                contours, hierarchy = cv2.findContours(
                    mask, 
                    cv2.RETR_CCOMP,  # Use connected components
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Group contours by their hierarchy
                    outer_paths = []
                    inner_paths = []
                    
                    for i, cnt in enumerate(contours):
                        if cv2.contourArea(cnt) < 10:  # Skip tiny areas
                            continue
                            
                        # Smooth contour with adaptive epsilon
                        area = cv2.contourArea(cnt)
                        epsilon = 0.001 * cv2.arcLength(cnt, True) * math.sqrt(1000 / (area + 1))
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        points = approx.reshape(-1, 2)
                        
                        if len(points) > 2:
                            # Create path with optimized coordinates
                            path = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
                            for x, y in points[1:]:
                                path += f"L {x:.1f},{y:.1f}"
                            path += "Z"
                            
                            # Add to appropriate group based on hierarchy
                            if hierarchy[0][i][3] == -1:  # Outer contour
                                outer_paths.append(path)
                            else:  # Inner contour (hole)
                                inner_paths.append(path)
                    
                    if outer_paths or inner_paths:
                        # Combine paths with fill-rule for proper rendering
                        svg_parts.append(
                            f'<path fill="rgb({color[0]},{color[1]},{color[2]})" '
                            f'fill-rule="evenodd" '
                            f'd="{" ".join(outer_paths + inner_paths)}"/>'
                        )
        
            # Add edges with better processing
            gray = cv2.cvtColor(bilateral, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Process edges with better grouping
            svg_parts.append('<g fill="none" stroke="black" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">')
            
            # Find and process edge contours
            edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            
            for cnt in edge_contours:
                if cv2.contourArea(cnt) > 20:
                    # Adaptive smoothing based on contour size
                    area = cv2.contourArea(cnt)
                    epsilon = 0.002 * cv2.arcLength(cnt, True) * math.sqrt(1000 / (area + 1))
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    points = approx.reshape(-1, 2)
                    
                    if len(points) > 2:
                        # Create optimized path
                        path = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
                        path += "".join([f"L {x:.1f},{y:.1f}" for x, y in points[1:]])
                        if cv2.arcLength(cnt, True) > 0:
                            path += "Z"
                        svg_parts.append(f'<path d="{path}"/>')
            
            svg_parts.append('</g>')
            svg_parts.append('</svg>')
            
            # Create optimized SVG string
            svg_string = '\n'.join(svg_parts)
            
            response = HttpResponse(svg_string, content_type='image/svg+xml')
            response['Content-Disposition'] = 'attachment; filename="cartoon-art.svg"'
            return response
        else:
            # classic posterize style
            w, h = img.size
            small_w = max(1, w // pixel_size)
            small_h = max(1, h // pixel_size)
            small_img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
            
            # Process colors
            img_array = np.array(small_img)
            pixels = img_array.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # Color clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags=cv2.KMEANS_RANDOM_CENTERS)
            
            # Start SVG
            svg_parts = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
                f'<rect width="{w}" height="{h}" fill="white"/>'
            ]
            
            # Convert labels back to 2D
            labels_2d = labels.reshape(small_h, small_w)
            
            # Process each color region more efficiently
            for color_idx in range(num_colors):
                color = tuple(map(int, palette[color_idx]))
                color_mask = (labels_2d == color_idx)
                
                if not np.any(color_mask):
                    continue
                
                # Find runs of same-colored pixels
                runs = []
                for y in range(small_h):
                    run_start = None
                    for x in range(small_w + 1):
                        if x < small_w and color_mask[y, x]:
                            if run_start is None:
                                run_start = x
                        elif run_start is not None:
                            runs.append((y, run_start, x))
                            run_start = None
                
                # Merge vertical runs
                merged_rects = []
                if runs:
                    current_rect = [runs[0][0], runs[0][1], runs[0][2], runs[0][0]]
                    
                    for y, start_x, end_x in runs[1:]:
                        if (y == current_rect[3] + 1 and 
                            start_x == current_rect[1] and 
                            end_x == current_rect[2]):
                            # Extend current rectangle
                            current_rect[3] = y
                        else:
                            # Add current rectangle and start new one
                            merged_rects.append(current_rect)
                            current_rect = [y, start_x, end_x, y]
                    
                    merged_rects.append(current_rect)
                
                # Create SVG elements for this color
                if merged_rects:
                    svg_parts.append(f'<g fill="rgb({color[0]},{color[1]},{color[2]}">')
                    for y_start, x_start, x_end, y_end in merged_rects:
                        x = x_start * pixel_size
                        y = y_start * pixel_size
                        width = (x_end - x_start) * pixel_size
                        height = (y_end - y_start + 1) * pixel_size
                        svg_parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}"/>')
                    svg_parts.append('</g>')
            
            svg_parts.append('</svg>')
            svg_string = '\n'.join(svg_parts)
            
            response = HttpResponse(svg_string, content_type='image/svg+xml')
            response['Content-Disposition'] = 'attachment; filename="posterized-classic.svg"'
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
        
        # Calculate optimal processing dimensions
        if original_img.size[0] * original_img.size[1] > 1000000:  # If larger than 1MP
            process_w, process_h = get_optimal_dimensions(original_img.size[0], original_img.size[1])
            img = original_img.resize((process_w, process_h), Image.Resampling.LANCZOS)
        else:
            img = original_img
            process_w, process_h = original_img.size[0], original_img.size[1]
        
        # Process at reduced size for faster computation
        img_array = np.array(img)
        
        # Create line art at reduced size
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if blur_radius > 0:
            gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        edges = cv2.Canny(gray, threshold/2, threshold)
        
        # Create posterized version at reduced size
        small_w = max(1, process_w // pixel_size)
        small_h = max(1, process_h // pixel_size)
        small_img = cv2.resize(img_array, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # Flatten and convert to float32 for k-means
        pixels = small_img.reshape(-1, 3).astype(np.float32)
        
        # Perform k-means with optimized parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 3, flags)
        
        # Convert back to uint8 and reshape
        palette = np.uint8(palette)
        quantized = palette[labels.flatten()].reshape(small_img.shape)
        
        # Resize posterized image to match original
        quantized = cv2.resize(quantized, (process_w, process_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert edges to 3 channels and resize
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Optimize blending operation
        mask = edges_rgb == 255
        result = quantized.copy()
        result[mask] = 0
        
        # Convert to PIL and resize if needed
        blended_img = Image.fromarray(result)
        if process_w != original_img.size[0] or process_h != original_img.size[1]:
            blended_img = blended_img.resize(original_img.size, Image.Resampling.NEAREST)
        
        # Save with optimization
        img_io = io.BytesIO()
        blended_img.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        
        return HttpResponse(img_io, content_type='image/png')
        
    except Exception as e:
        print(f"Error in blend_art: {str(e)}")
        return Response({'error': str(e)}, status=500)
