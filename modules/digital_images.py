import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class DigitalImages:
    """Module for comprehensive digital image analysis and pixel exploration"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Calculate bit depth and color depth
        self.bit_depth = self._calculate_bit_depth()
        self.color_depth = self._calculate_color_depth()
        
    def _calculate_bit_depth(self) -> int:
        """Calculate bit depth of the image"""
        if self.image.dtype == np.uint8:
            return 8
        elif self.image.dtype == np.uint16:
            return 16
        elif self.image.dtype == np.float32:
            return 32
        else:
            return 8
    
    def _calculate_color_depth(self) -> int:
        """Calculate total color depth"""
        return self.bit_depth * self.channels
    
    def get_digital_properties(self) -> Dict[str, Any]:
        """Get comprehensive digital properties of the image"""
        try:
            properties = {
                'basic_info': {
                    'width': self.width,
                    'height': self.height,
                    'channels': self.channels,
                    'total_pixels': self.width * self.height,
                    'aspect_ratio': round(self.width / self.height, 3)
                },
                'color_properties': {
                    'bit_depth': self.bit_depth,
                    'color_depth': self.color_depth,
                    'max_possible_values': 2 ** self.bit_depth,
                    'total_color_combinations': (2 ** self.bit_depth) ** self.channels,
                    'data_type': str(self.image.dtype)
                },
                'image_types': {
                    'is_binary': self._is_binary(),
                    'is_grayscale': self._is_grayscale(),
                    'is_color': self._is_color(),
                    'has_transparency': self._has_transparency()
                },
                'memory_usage': {
                    'size_bytes': self.image.nbytes,
                    'size_kb': round(self.image.nbytes / 1024, 2),
                    'size_mb': round(self.image.nbytes / (1024 * 1024), 2),
                    'bytes_per_pixel': self.image.nbytes / (self.width * self.height)
                }
            }
            
            # Add unique color count
            properties['color_analysis'] = self._analyze_colors()
            
            return properties
        
        except Exception as e:
            return {'error': f'Failed to get digital properties: {str(e)}'}
    
    def _is_binary(self) -> bool:
        """Check if image is binary (only 0 and 255 values)"""
        if self.channels > 1:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.squeeze()
        
        unique_values = np.unique(gray)
        return len(unique_values) == 2 and 0 in unique_values and 255 in unique_values
    
    def _is_grayscale(self) -> bool:
        """Check if image is grayscale"""
        if self.channels == 1:
            return True
        
        # Check if all channels are identical
        b, g, r = cv2.split(self.image)
        return np.array_equal(b, g) and np.array_equal(g, r)
    
    def _is_color(self) -> bool:
        """Check if image is color"""
        return self.channels >= 3 and not self._is_grayscale()
    
    def _has_transparency(self) -> bool:
        """Check if image has transparency channel"""
        return self.channels == 4
    
    def _analyze_colors(self) -> Dict[str, Any]:
        """Analyze color distribution and unique colors"""
        if self.channels == 1:
            unique_colors = len(np.unique(self.image))
            dominant_color = None
        else:
            # Reshape image to list of pixels
            pixels = self.image_rgb.reshape(-1, self.channels)
            unique_pixels = np.unique(pixels, axis=0)
            unique_colors = len(unique_pixels)
            
            # Find dominant color
            pixel_counts = {}
            for pixel in pixels:
                key = tuple(pixel)
                pixel_counts[key] = pixel_counts.get(key, 0) + 1
            
            dominant_color = max(pixel_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'unique_colors': unique_colors,
            'color_diversity': round(unique_colors / (self.width * self.height), 4),
            'dominant_color': dominant_color
        }
    
    def get_pixel_matrix(self, x: int, y: int, size: int = 5) -> Dict[str, Any]:
        """Get pixel matrix around specified coordinates"""
        try:
            # Validate coordinates
            if not (0 <= x < self.width and 0 <= y < self.height):
                return {'error': f'Coordinates ({x}, {y}) are out of bounds'}
            
            # Calculate bounds
            half_size = size // 2
            x_min = max(0, x - half_size)
            x_max = min(self.width, x + half_size + 1)
            y_min = max(0, y - half_size)
            y_max = min(self.height, y + half_size + 1)
            
            # Extract matrix from different color spaces
            rgb_matrix = self.image_rgb[y_min:y_max, x_min:x_max]
            bgr_matrix = self.image[y_min:y_max, x_min:x_max]
            
            # Convert to other color spaces
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_matrix = hsv_image[y_min:y_max, x_min:x_max]
            
            lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            lab_matrix = lab_image[y_min:y_max, x_min:x_max]
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_matrix = gray_image[y_min:y_max, x_min:x_max]
            
            return {
                'center': {'x': x, 'y': y},
                'matrix_size': f"{rgb_matrix.shape[1]}x{rgb_matrix.shape[0]}",
                'bounds': {
                    'x_min': x_min, 'x_max': x_max - 1,
                    'y_min': y_min, 'y_max': y_max - 1
                },
                'matrices': {
                    'rgb': rgb_matrix.tolist(),
                    'bgr': bgr_matrix.tolist(),
                    'hsv': hsv_matrix.tolist(),
                    'lab': lab_matrix.tolist(),
                    'grayscale': gray_matrix.tolist()
                },
                'center_pixel': {
                    'rgb': rgb_matrix[half_size, half_size].tolist() if rgb_matrix.shape[0] > half_size and rgb_matrix.shape[1] > half_size else None,
                    'hsv': hsv_matrix[half_size, half_size].tolist() if hsv_matrix.shape[0] > half_size and hsv_matrix.shape[1] > half_size else None,
                    'lab': lab_matrix[half_size, half_size].tolist() if lab_matrix.shape[0] > half_size and lab_matrix.shape[1] > half_size else None,
                    'grayscale': int(gray_matrix[half_size, half_size]) if gray_matrix.shape[0] > half_size and gray_matrix.shape[1] > half_size else None
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to get pixel matrix: {str(e)}'}
    
    def analyze_bit_planes(self) -> Dict[str, Any]:
        """Analyze bit planes of the image"""
        try:
            if self.channels > 1:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image.squeeze()
            
            bit_planes = []
            bit_plane_data = {}
            
            for i in range(8):  # For 8-bit images
                # Extract bit plane
                bit_plane = (gray >> i) & 1
                bit_plane = bit_plane * 255  # Convert to 0-255 range
                
                bit_planes.append(bit_plane)
                
                # Calculate statistics for this bit plane
                bit_plane_data[f'bit_plane_{i}'] = {
                    'significance': f'2^{i} = {2**i}',
                    'contribution_percentage': round(100 * (2**i) / 255, 2),
                    'ones_count': int(np.sum(bit_plane == 255)),
                    'zeros_count': int(np.sum(bit_plane == 0)),
                    'ones_percentage': round(100 * np.sum(bit_plane == 255) / bit_plane.size, 2)
                }
            
            return {
                'bit_planes_data': bit_plane_data,
                'total_bit_planes': 8,
                'most_significant_plane': 'bit_plane_7',
                'least_significant_plane': 'bit_plane_0'
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze bit planes: {str(e)}'}
    
    def get_channel_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of each color channel"""
        try:
            if self.channels == 1:
                return {'error': 'Single channel image - no separate channels to analyze'}
            
            channels = cv2.split(self.image)
            channel_names = ['Blue', 'Green', 'Red'] if self.channels >= 3 else []
            
            if self.channels == 4:
                channel_names.append('Alpha')
            
            channel_analysis = {}
            
            for i, (channel, name) in enumerate(zip(channels, channel_names)):
                stats = {
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel)),
                    'min': int(np.min(channel)),
                    'max': int(np.max(channel)),
                    'median': float(np.median(channel)),
                    'unique_values': int(len(np.unique(channel))),
                    'contribution_to_brightness': float(np.mean(channel) / 255 * 100)
                }
                
                # Calculate histogram
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                stats['histogram'] = hist.flatten().tolist()
                
                # Find dominant values
                hist_counts = [(i, int(count)) for i, count in enumerate(hist.flatten())]
                hist_counts.sort(key=lambda x: x[1], reverse=True)
                stats['dominant_values'] = hist_counts[:5]  # Top 5 most frequent values
                
                channel_analysis[name.lower()] = stats
            
            return {
                'channel_count': self.channels,
                'channels': channel_analysis,
                'color_space': 'BGR',
                'analysis_summary': {
                    'most_dominant_channel': max(channel_analysis.keys(), 
                                               key=lambda k: channel_analysis[k]['mean']),
                    'least_dominant_channel': min(channel_analysis.keys(), 
                                                key=lambda k: channel_analysis[k]['mean']),
                    'highest_contrast_channel': max(channel_analysis.keys(), 
                                                  key=lambda k: channel_analysis[k]['std']),
                    'most_uniform_channel': min(channel_analysis.keys(), 
                                              key=lambda k: channel_analysis[k]['std'])
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze channels: {str(e)}'}
    
    def create_pixel_value_visualization(self, x: int, y: int, radius: int = 10) -> Dict[str, Any]:
        """Create visualization of pixel values in a region"""
        try:
            # Get neighborhood
            x_min = max(0, x - radius)
            x_max = min(self.width, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(self.height, y + radius + 1)
            
            region_rgb = self.image_rgb[y_min:y_max, x_min:x_max]
            region_gray = cv2.cvtColor(self.image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
            
            # Create 3D surface plot data for visualization
            h, w = region_gray.shape
            x_coords, y_coords = np.meshgrid(range(w), range(h))
            
            return {
                'region_bounds': {
                    'x_min': x_min, 'x_max': x_max - 1,
                    'y_min': y_min, 'y_max': y_max - 1
                },
                'visualization_data': {
                    'x_coords': x_coords.tolist(),
                    'y_coords': y_coords.tolist(),
                    'z_values': region_gray.tolist(),
                    'rgb_values': region_rgb.tolist()
                },
                'statistics': {
                    'mean_intensity': float(np.mean(region_gray)),
                    'max_intensity': int(np.max(region_gray)),
                    'min_intensity': int(np.min(region_gray)),
                    'intensity_range': int(np.max(region_gray) - np.min(region_gray)),
                    'gradient_magnitude': float(np.mean(np.gradient(region_gray.astype(float))))
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to create pixel visualization: {str(e)}'}