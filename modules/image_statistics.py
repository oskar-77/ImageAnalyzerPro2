import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt

class ImageStatistics:
    """Module for comprehensive image statistics calculation"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def get_brightness_statistics(self) -> Dict[str, Any]:
        """Calculate brightness statistics"""
        try:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            mean_brightness = float(np.mean(gray))
            std_brightness = float(np.std(gray))
            min_brightness = int(np.min(gray))
            max_brightness = int(np.max(gray))
            median_brightness = float(np.median(gray))
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = [float(np.percentile(gray, p)) for p in percentiles]
            
            # Entropy (measure of information content)
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
            
            return {
                'mean': mean_brightness,
                'std': std_brightness,
                'min': min_brightness,
                'max': max_brightness,
                'median': median_brightness,
                'percentiles': dict(zip(percentiles, percentile_values)),
                'entropy': float(entropy),
                'dynamic_range': max_brightness - min_brightness
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate brightness statistics: {str(e)}'}
    
    def get_channel_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for each color channel"""
        try:
            if self.channels == 1:
                return {'message': 'Single channel image'}
            
            channel_names = ['Blue', 'Green', 'Red'] if self.channels == 3 else ['Channel_0']
            channel_stats = {}
            
            for i, name in enumerate(channel_names):
                channel = self.image[:, :, i]
                
                channel_stats[name] = {
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel)),
                    'min': int(np.min(channel)),
                    'max': int(np.max(channel)),
                    'median': float(np.median(channel)),
                    'skewness': float(stats.skew(channel.flatten())),
                    'kurtosis': float(stats.kurtosis(channel.flatten())),
                    'unique_values': int(len(np.unique(channel)))
                }
            
            return {'channel_stats': channel_stats}
        
        except Exception as e:
            return {'error': f'Failed to calculate channel statistics: {str(e)}'}
    
    def get_histogram_data(self) -> Dict[str, Any]:
        """Calculate histogram data for plotting"""
        try:
            histogram_data = {}
            
            if self.channels == 1:
                # Grayscale histogram
                hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
                histogram_data['grayscale'] = {
                    'bins': list(range(256)),
                    'values': hist.flatten().tolist()
                }
            else:
                # Color histograms
                colors = ['blue', 'green', 'red']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    histogram_data[color] = {
                        'bins': list(range(256)),
                        'values': hist.flatten().tolist()
                    }
                
                # Combined histogram
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                histogram_data['combined'] = {
                    'bins': list(range(256)),
                    'values': hist.flatten().tolist()
                }
            
            return histogram_data
        
        except Exception as e:
            return {'error': f'Failed to calculate histogram data: {str(e)}'}
    
    def get_texture_features(self) -> Dict[str, Any]:
        """Calculate texture features using various methods"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Laplacian edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Gabor filter response
            gabor_real, gabor_imag = cv2.getGaborKernel((15, 15), 2, 0, 2*np.pi/4, 0.5, 0, ktype=cv2.CV_32F), None
            gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_real)
            
            return {
                'edge_density': float(np.mean(sobel_magnitude > 10)),
                'edge_strength': float(np.mean(sobel_magnitude)),
                'laplacian_variance': float(np.var(laplacian)),
                'gabor_response': float(np.mean(gabor_response)),
                'local_binary_pattern_uniformity': self._calculate_lbp_uniformity(gray)
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate texture features: {str(e)}'}
    
    def _calculate_lbp_uniformity(self, gray_image: np.ndarray) -> float:
        """Calculate Local Binary Pattern uniformity measure"""
        try:
            # Simple LBP implementation
            rows, cols = gray_image.shape
            lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray_image[i, j]
                    code = 0
                    code |= (gray_image[i-1, j-1] > center) << 7
                    code |= (gray_image[i-1, j] > center) << 6
                    code |= (gray_image[i-1, j+1] > center) << 5
                    code |= (gray_image[i, j+1] > center) << 4
                    code |= (gray_image[i+1, j+1] > center) << 3
                    code |= (gray_image[i+1, j] > center) << 2
                    code |= (gray_image[i+1, j-1] > center) << 1
                    code |= (gray_image[i, j-1] > center) << 0
                    lbp_image[i-1, j-1] = code
            
            # Calculate uniformity
            hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
            uniformity = np.sum(hist**2) / (lbp_image.size**2)
            
            return float(uniformity)
        
        except Exception as e:
            return 0.0
    
    def get_color_analysis(self) -> Dict[str, Any]:
        """Analyze color distribution and properties"""
        try:
            if self.channels == 1:
                return {'message': 'Color analysis not applicable to grayscale images'}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            
            # Color dominance
            image_reshaped = self.image.reshape(-1, 3)
            unique_colors = len(np.unique(image_reshaped.view(np.dtype((np.void, image_reshaped.dtype.itemsize * image_reshaped.shape[1])))))
            
            # HSV analysis
            hue_mean = float(np.mean(hsv[:, :, 0]))
            saturation_mean = float(np.mean(hsv[:, :, 1]))
            value_mean = float(np.mean(hsv[:, :, 2]))
            
            # Color temperature estimation (simplified)
            r_mean = float(np.mean(self.image_rgb[:, :, 0]))
            b_mean = float(np.mean(self.image_rgb[:, :, 2]))
            color_temperature = 'warm' if r_mean > b_mean else 'cool'
            
            return {
                'unique_colors': unique_colors,
                'color_diversity': unique_colors / (self.width * self.height),
                'hsv_analysis': {
                    'hue_mean': hue_mean,
                    'saturation_mean': saturation_mean,
                    'value_mean': value_mean
                },
                'color_temperature': color_temperature,
                'temperature_ratio': float(r_mean / (b_mean + 1e-10))
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze colors: {str(e)}'}
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get all statistics in one call"""
        try:
            return {
                'brightness': self.get_brightness_statistics(),
                'channels': self.get_channel_statistics(),
                'texture': self.get_texture_features(),
                'color': self.get_color_analysis(),
                'image_info': {
                    'width': self.width,
                    'height': self.height,
                    'channels': self.channels,
                    'total_pixels': self.width * self.height
                }
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate comprehensive statistics: {str(e)}'}
