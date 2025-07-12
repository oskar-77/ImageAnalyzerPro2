import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class ImageExploration:
    """Module for interactive pixel exploration and analysis"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def get_pixel_value(self, x: int, y: int) -> Dict[str, Any]:
        """Get pixel value at specified coordinates"""
        try:
            # Validate coordinates
            if not (0 <= x < self.width and 0 <= y < self.height):
                return {
                    'error': f'Coordinates ({x}, {y}) are out of bounds. Image size: {self.width}x{self.height}'
                }
            
            # Get pixel values
            bgr_pixel = self.image[y, x]
            rgb_pixel = self.image_rgb[y, x]
            
            # Convert to different color spaces
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_pixel = hsv_image[y, x]
            
            lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            lab_pixel = lab_image[y, x]
            
            # Calculate brightness
            brightness = np.mean(rgb_pixel)
            
            return {
                'coordinates': {'x': int(x), 'y': int(y)},
                'rgb': {
                    'r': int(rgb_pixel[0]),
                    'g': int(rgb_pixel[1]),
                    'b': int(rgb_pixel[2])
                },
                'bgr': {
                    'b': int(bgr_pixel[0]),
                    'g': int(bgr_pixel[1]),
                    'r': int(bgr_pixel[2])
                },
                'hsv': {
                    'h': int(hsv_pixel[0]),
                    's': int(hsv_pixel[1]),
                    'v': int(hsv_pixel[2])
                },
                'lab': {
                    'l': int(lab_pixel[0]),
                    'a': int(lab_pixel[1]),
                    'b': int(lab_pixel[2])
                },
                'brightness': float(brightness),
                'hex': f"#{rgb_pixel[0]:02x}{rgb_pixel[1]:02x}{rgb_pixel[2]:02x}"
            }
        
        except Exception as e:
            return {'error': f'Failed to get pixel value: {str(e)}'}
    
    def get_neighborhood_values(self, x: int, y: int, radius: int = 1) -> Dict[str, Any]:
        """Get pixel values in a neighborhood around specified coordinates"""
        try:
            # Validate coordinates
            if not (0 <= x < self.width and 0 <= y < self.height):
                return {'error': f'Coordinates ({x}, {y}) are out of bounds'}
            
            # Define neighborhood bounds
            x_min = max(0, x - radius)
            x_max = min(self.width, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(self.height, y + radius + 1)
            
            # Extract neighborhood
            neighborhood = self.image_rgb[y_min:y_max, x_min:x_max]
            
            # Calculate statistics
            stats = {
                'mean': np.mean(neighborhood, axis=(0, 1)).tolist(),
                'std': np.std(neighborhood, axis=(0, 1)).tolist(),
                'min': np.min(neighborhood, axis=(0, 1)).tolist(),
                'max': np.max(neighborhood, axis=(0, 1)).tolist()
            }
            
            return {
                'center': {'x': int(x), 'y': int(y)},
                'radius': radius,
                'bounds': {
                    'x_min': x_min, 'x_max': x_max - 1,
                    'y_min': y_min, 'y_max': y_max - 1
                },
                'statistics': stats,
                'size': neighborhood.shape
            }
        
        except Exception as e:
            return {'error': f'Failed to get neighborhood values: {str(e)}'}
    
    def get_line_profile(self, x1: int, y1: int, x2: int, y2: int) -> Dict[str, Any]:
        """Get pixel intensity profile along a line"""
        try:
            # Validate coordinates
            if not (0 <= x1 < self.width and 0 <= y1 < self.height and
                    0 <= x2 < self.width and 0 <= y2 < self.height):
                return {'error': 'Line coordinates are out of bounds'}
            
            # Generate line points using Bresenham's algorithm
            points = []
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            x, y = x1, y1
            while True:
                points.append((x, y))
                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
            
            # Extract pixel values along the line
            rgb_values = []
            brightness_values = []
            
            for x, y in points:
                pixel = self.image_rgb[y, x]
                rgb_values.append(pixel.tolist())
                brightness_values.append(float(np.mean(pixel)))
            
            return {
                'start': {'x': x1, 'y': y1},
                'end': {'x': x2, 'y': y2},
                'points': [{'x': x, 'y': y} for x, y in points],
                'rgb_values': rgb_values,
                'brightness_values': brightness_values,
                'length': len(points)
            }
        
        except Exception as e:
            return {'error': f'Failed to get line profile: {str(e)}'}
    
    def get_image_info(self) -> Dict[str, Any]:
        """Get basic image information"""
        try:
            return {
                'width': self.width,
                'height': self.height,
                'channels': self.channels,
                'total_pixels': self.width * self.height,
                'color_space': 'BGR' if self.channels == 3 else 'Grayscale',
                'data_type': str(self.image.dtype)
            }
        
        except Exception as e:
            return {'error': f'Failed to get image info: {str(e)}'}
