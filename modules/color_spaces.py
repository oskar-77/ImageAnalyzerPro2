import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ColorSpaces:
    """Advanced color space conversion and analysis module"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def convert_all_color_spaces(self) -> Dict[str, Any]:
        """Convert image to all major color spaces"""
        try:
            conversions = {}
            
            # RGB (already available)
            conversions['rgb'] = {
                'data': self.image_rgb,
                'description': 'Red, Green, Blue - Standard color space for displays',
                'channels': ['Red', 'Green', 'Blue'],
                'range': [0, 255]
            }
            
            # BGR (original OpenCV format)
            conversions['bgr'] = {
                'data': self.image,
                'description': 'Blue, Green, Red - OpenCV default format',
                'channels': ['Blue', 'Green', 'Red'],
                'range': [0, 255]
            }
            
            # HSV Color Space
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            conversions['hsv'] = {
                'data': hsv,
                'description': 'Hue, Saturation, Value - Intuitive color representation',
                'channels': ['Hue (0-179)', 'Saturation (0-255)', 'Value (0-255)'],
                'range': [[0, 179], [0, 255], [0, 255]]
            }
            
            # LAB Color Space
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            conversions['lab'] = {
                'data': lab,
                'description': 'Lightness, A (green-red), B (blue-yellow) - Perceptually uniform',
                'channels': ['L (0-255)', 'A (0-255)', 'B (0-255)'],
                'range': [[0, 255], [0, 255], [0, 255]]
            }
            
            # YUV Color Space
            yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            conversions['yuv'] = {
                'data': yuv,
                'description': 'Luminance, U, V - Used in video compression',
                'channels': ['Y (Luminance)', 'U (Chroma)', 'V (Chroma)'],
                'range': [0, 255]
            }
            
            # XYZ Color Space
            xyz = cv2.cvtColor(self.image, cv2.COLOR_BGR2XYZ)
            conversions['xyz'] = {
                'data': xyz,
                'description': 'CIE XYZ - Device-independent color space',
                'channels': ['X', 'Y', 'Z'],
                'range': [0, 255]
            }
            
            # HLS Color Space
            hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
            conversions['hls'] = {
                'data': hls,
                'description': 'Hue, Lightness, Saturation - Alternative to HSV',
                'channels': ['Hue (0-179)', 'Lightness (0-255)', 'Saturation (0-255)'],
                'range': [[0, 179], [0, 255], [0, 255]]
            }
            
            # Grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            conversions['grayscale'] = {
                'data': gray,
                'description': 'Single channel intensity values',
                'channels': ['Intensity'],
                'range': [0, 255]
            }
            
            return {
                'conversions': conversions,
                'original_space': 'BGR',
                'total_spaces': len(conversions)
            }
        
        except Exception as e:
            return {'error': f'Failed to convert color spaces: {str(e)}'}
    
    def compare_color_spaces(self, pixel_x: int, pixel_y: int) -> Dict[str, Any]:
        """Compare pixel values across different color spaces"""
        try:
            if not (0 <= pixel_x < self.width and 0 <= pixel_y < self.height):
                return {'error': f'Coordinates ({pixel_x}, {pixel_y}) are out of bounds'}
            
            # Get pixel from all color spaces
            rgb_pixel = self.image_rgb[pixel_y, pixel_x]
            bgr_pixel = self.image[pixel_y, pixel_x]
            
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_pixel = hsv_image[pixel_y, pixel_x]
            
            lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            lab_pixel = lab_image[pixel_y, pixel_x]
            
            yuv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            yuv_pixel = yuv_image[pixel_y, pixel_x]
            
            xyz_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2XYZ)
            xyz_pixel = xyz_image[pixel_y, pixel_x]
            
            hls_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)
            hls_pixel = hls_image[pixel_y, pixel_x]
            
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray_pixel = gray_image[pixel_y, pixel_x]
            
            return {
                'coordinates': {'x': pixel_x, 'y': pixel_y},
                'color_values': {
                    'rgb': {
                        'r': int(rgb_pixel[0]),
                        'g': int(rgb_pixel[1]),
                        'b': int(rgb_pixel[2]),
                        'hex': f"#{rgb_pixel[0]:02x}{rgb_pixel[1]:02x}{rgb_pixel[2]:02x}"
                    },
                    'bgr': {
                        'b': int(bgr_pixel[0]),
                        'g': int(bgr_pixel[1]),
                        'r': int(bgr_pixel[2])
                    },
                    'hsv': {
                        'h': int(hsv_pixel[0]),
                        's': int(hsv_pixel[1]),
                        'v': int(hsv_pixel[2]),
                        'h_degrees': int(hsv_pixel[0]) * 2,  # Convert to 0-360 range
                        's_percent': round(int(hsv_pixel[1]) / 255 * 100, 1),
                        'v_percent': round(int(hsv_pixel[2]) / 255 * 100, 1)
                    },
                    'lab': {
                        'l': int(lab_pixel[0]),
                        'a': int(lab_pixel[1]),
                        'b': int(lab_pixel[2])
                    },
                    'yuv': {
                        'y': int(yuv_pixel[0]),
                        'u': int(yuv_pixel[1]),
                        'v': int(yuv_pixel[2])
                    },
                    'xyz': {
                        'x': int(xyz_pixel[0]),
                        'y': int(xyz_pixel[1]),
                        'z': int(xyz_pixel[2])
                    },
                    'hls': {
                        'h': int(hls_pixel[0]),
                        'l': int(hls_pixel[1]),
                        's': int(hls_pixel[2])
                    },
                    'grayscale': {
                        'intensity': int(gray_pixel)
                    }
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to compare color spaces: {str(e)}'}
    
    def analyze_color_distribution(self, color_space: str = 'hsv') -> Dict[str, Any]:
        """Analyze color distribution in specified color space"""
        try:
            if color_space.lower() == 'rgb':
                image_data = self.image_rgb
                channels = ['Red', 'Green', 'Blue']
            elif color_space.lower() == 'hsv':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                channels = ['Hue', 'Saturation', 'Value']
            elif color_space.lower() == 'lab':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
                channels = ['L', 'A', 'B']
            elif color_space.lower() == 'yuv':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
                channels = ['Y', 'U', 'V']
            else:
                return {'error': f'Unsupported color space: {color_space}'}
            
            # Split channels
            if len(image_data.shape) == 3:
                channel_data = cv2.split(image_data)
            else:
                channel_data = [image_data]
                channels = ['Intensity']
            
            distribution_analysis = {}
            
            for i, (channel, name) in enumerate(zip(channel_data, channels)):
                # Calculate histogram
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                hist_normalized = hist / hist.sum()
                
                # Calculate statistics
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                skewness = self._calculate_skewness(channel)
                kurtosis = self._calculate_kurtosis(channel)
                
                # Find peaks in histogram
                peaks = self._find_histogram_peaks(hist.flatten())
                
                distribution_analysis[name.lower()] = {
                    'histogram': hist.flatten().tolist(),
                    'histogram_normalized': hist_normalized.flatten().tolist(),
                    'statistics': {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'min': int(np.min(channel)),
                        'max': int(np.max(channel)),
                        'median': float(np.median(channel)),
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis)
                    },
                    'peaks': peaks,
                    'dominant_values': self._get_dominant_values(hist.flatten())
                }
            
            return {
                'color_space': color_space.upper(),
                'channels': distribution_analysis,
                'overall_analysis': self._analyze_overall_distribution(distribution_analysis)
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze color distribution: {str(e)}'}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        skew = np.mean(((data - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        kurt = np.mean(((data - mean) / std) ** 4) - 3
        return kurt
    
    def _find_histogram_peaks(self, hist: np.ndarray, prominence: float = 0.01) -> List[Dict[str, Any]]:
        """Find peaks in histogram"""
        peaks = []
        max_val = np.max(hist)
        threshold = max_val * prominence
        
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > threshold:
                peaks.append({
                    'position': i,
                    'value': int(hist[i]),
                    'prominence': float(hist[i] / max_val)
                })
        
        # Sort by prominence
        peaks.sort(key=lambda x: x['prominence'], reverse=True)
        return peaks[:5]  # Return top 5 peaks
    
    def _get_dominant_values(self, hist: np.ndarray) -> List[Dict[str, Any]]:
        """Get dominant values from histogram"""
        # Find indices sorted by histogram values
        sorted_indices = np.argsort(hist)[::-1]
        
        dominant = []
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            if hist[idx] > 0:
                dominant.append({
                    'value': int(idx),
                    'count': int(hist[idx]),
                    'percentage': float(hist[idx] / hist.sum() * 100)
                })
        
        return dominant
    
    def _analyze_overall_distribution(self, channel_analysis: Dict) -> Dict[str, Any]:
        """Analyze overall distribution characteristics"""
        try:
            all_means = [ch['statistics']['mean'] for ch in channel_analysis.values()]
            all_stds = [ch['statistics']['std'] for ch in channel_analysis.values()]
            
            return {
                'overall_brightness': float(np.mean(all_means)),
                'overall_contrast': float(np.mean(all_stds)),
                'color_balance': {
                    'most_prominent_channel': max(channel_analysis.keys(), 
                                                key=lambda k: channel_analysis[k]['statistics']['mean']),
                    'least_prominent_channel': min(channel_analysis.keys(), 
                                                 key=lambda k: channel_analysis[k]['statistics']['mean']),
                    'highest_contrast_channel': max(channel_analysis.keys(), 
                                                  key=lambda k: channel_analysis[k]['statistics']['std']),
                    'most_uniform_channel': min(channel_analysis.keys(), 
                                              key=lambda k: channel_analysis[k]['statistics']['std'])
                }
            }
        except:
            return {}
    
    def create_color_space_comparison_plot(self) -> Dict[str, Any]:
        """Create data for color space comparison visualization"""
        try:
            # Convert to different color spaces
            rgb = self.image_rgb
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Create visualization data
            return {
                'images': {
                    'rgb': rgb.tolist(),
                    'hsv': hsv.tolist(),
                    'lab': lab.tolist(),
                    'grayscale': gray.tolist()
                },
                'descriptions': {
                    'rgb': 'RGB - Red, Green, Blue channels',
                    'hsv': 'HSV - Hue, Saturation, Value channels',
                    'lab': 'LAB - Lightness, A, B channels',
                    'grayscale': 'Grayscale - Single intensity channel'
                },
                'dimensions': {
                    'width': self.width,
                    'height': self.height
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to create comparison plot: {str(e)}'}
    
    def extract_color_palette(self, n_colors: int = 8, color_space: str = 'rgb') -> Dict[str, Any]:
        """Extract dominant color palette from image"""
        try:
            from sklearn.cluster import KMeans
            
            if color_space.lower() == 'rgb':
                image_data = self.image_rgb
            elif color_space.lower() == 'hsv':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            elif color_space.lower() == 'lab':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            else:
                image_data = self.image_rgb
            
            # Reshape image to list of pixels
            pixels = image_data.reshape(-1, 3)
            
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get color centers
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate percentages
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            
            # Sort by percentage
            sorted_indices = np.argsort(percentages)[::-1]
            
            palette = []
            for i in sorted_indices:
                color = colors[i].astype(int)
                
                # Convert back to RGB if needed
                if color_space.lower() == 'hsv':
                    # Create single pixel image and convert
                    hsv_color = np.uint8([[color]])
                    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
                elif color_space.lower() == 'lab':
                    # Create single pixel image and convert
                    lab_color = np.uint8([[color]])
                    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]
                    rgb_color = bgr_color[::-1]  # BGR to RGB
                else:
                    rgb_color = color
                
                palette.append({
                    'index': int(i),
                    'rgb': rgb_color.tolist(),
                    'hex': f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}",
                    'percentage': float(percentages[i]),
                    'original_space_values': color.tolist()
                })
            
            return {
                'palette': palette,
                'total_colors': n_colors,
                'extraction_space': color_space.upper(),
                'coverage': sum(p['percentage'] for p in palette)
            }
        
        except Exception as e:
            return {'error': f'Failed to extract color palette: {str(e)}'}