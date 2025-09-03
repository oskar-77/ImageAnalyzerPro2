import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage
from skimage import filters

class Thresholding:
    """Advanced thresholding techniques module"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for thresholding
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def apply_global_threshold(self, threshold: int = 127, max_value: int = 255, 
                             threshold_type: str = 'binary') -> Dict[str, Any]:
        """Apply global thresholding"""
        try:
            # Define threshold types
            thresh_types = {
                'binary': cv2.THRESH_BINARY,
                'binary_inv': cv2.THRESH_BINARY_INV,
                'trunc': cv2.THRESH_TRUNC,
                'tozero': cv2.THRESH_TOZERO,
                'tozero_inv': cv2.THRESH_TOZERO_INV
            }
            
            if threshold_type not in thresh_types:
                return {'error': f'Unsupported threshold type: {threshold_type}'}
            
            # Apply thresholding
            ret_val, thresholded = cv2.threshold(self.gray, threshold, max_value, thresh_types[threshold_type])
            
            # Calculate statistics
            foreground_pixels = np.sum(thresholded == max_value)
            background_pixels = np.sum(thresholded == 0)
            total_pixels = self.width * self.height
            
            foreground_percentage = (foreground_pixels / total_pixels) * 100
            background_percentage = (background_pixels / total_pixels) * 100
            
            return {
                'thresholded_image': thresholded,
                'threshold_value': threshold,
                'actual_threshold': float(ret_val),
                'threshold_type': threshold_type,
                'statistics': {
                    'foreground_pixels': int(foreground_pixels),
                    'background_pixels': int(background_pixels),
                    'total_pixels': total_pixels,
                    'foreground_percentage': float(foreground_percentage),
                    'background_percentage': float(background_percentage)
                },
                'parameters': {
                    'threshold': threshold,
                    'max_value': max_value,
                    'type': threshold_type
                },
                'description': f'Global {threshold_type} thresholding at {threshold}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply global threshold: {str(e)}'}
    
    def apply_otsu_threshold(self) -> Dict[str, Any]:
        """Apply Otsu's automatic thresholding"""
        try:
            # Apply Otsu thresholding
            otsu_threshold, thresholded = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate histogram for analysis
            hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
            
            # Calculate between-class variance manually for verification
            def calculate_otsu_threshold(histogram):
                total_pixels = histogram.sum()
                sum_total = np.sum(np.arange(256) * histogram.flatten())
                
                sum_bg = 0
                weight_bg = 0
                max_variance = 0
                optimal_threshold = 0
                
                for t in range(256):
                    weight_bg += histogram[t]
                    if weight_bg == 0:
                        continue
                    
                    weight_fg = total_pixels - weight_bg
                    if weight_fg == 0:
                        break
                    
                    sum_bg += t * histogram[t]
                    
                    mean_bg = sum_bg / weight_bg
                    mean_fg = (sum_total - sum_bg) / weight_fg
                    
                    # Between-class variance
                    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                    
                    if variance > max_variance:
                        max_variance = variance
                        optimal_threshold = t
                
                return optimal_threshold, max_variance
            
            calculated_threshold, max_variance = calculate_otsu_threshold(hist)
            
            # Calculate statistics
            foreground_pixels = np.sum(thresholded == 255)
            background_pixels = np.sum(thresholded == 0)
            total_pixels = self.width * self.height
            
            # Calculate class separability
            fg_mean = np.mean(self.gray[thresholded == 255]) if foreground_pixels > 0 else 0
            bg_mean = np.mean(self.gray[thresholded == 0]) if background_pixels > 0 else 0
            separability = abs(fg_mean - bg_mean)
            
            return {
                'thresholded_image': thresholded,
                'otsu_threshold': float(otsu_threshold),
                'calculated_threshold': int(calculated_threshold),
                'max_variance': float(max_variance),
                'histogram': hist.flatten().tolist(),
                'statistics': {
                    'foreground_pixels': int(foreground_pixels),
                    'background_pixels': int(background_pixels),
                    'foreground_percentage': float((foreground_pixels / total_pixels) * 100),
                    'background_percentage': float((background_pixels / total_pixels) * 100),
                    'foreground_mean': float(fg_mean),
                    'background_mean': float(bg_mean),
                    'class_separability': float(separability)
                },
                'description': f"Otsu's automatic thresholding at {otsu_threshold:.1f}",
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Otsu threshold: {str(e)}'}
    
    def apply_adaptive_threshold(self, max_value: int = 255, method: str = 'gaussian',
                               threshold_type: str = 'binary', block_size: int = 11,
                               c_constant: float = 2) -> Dict[str, Any]:
        """Apply adaptive thresholding"""
        try:
            # Ensure odd block size
            if block_size % 2 == 0:
                block_size += 1
            
            # Define methods
            adaptive_methods = {
                'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
                'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            }
            
            thresh_types = {
                'binary': cv2.THRESH_BINARY,
                'binary_inv': cv2.THRESH_BINARY_INV
            }
            
            if method not in adaptive_methods:
                return {'error': f'Unsupported adaptive method: {method}'}
            
            if threshold_type not in thresh_types:
                return {'error': f'Unsupported threshold type: {threshold_type}'}
            
            # Apply adaptive thresholding
            thresholded = cv2.adaptiveThreshold(
                self.gray, max_value, adaptive_methods[method],
                thresh_types[threshold_type], block_size, c_constant
            )
            
            # Calculate local threshold variation
            # Create a sliding window to analyze local thresholds
            kernel = np.ones((block_size, block_size), np.float32) / (block_size ** 2)
            local_mean = cv2.filter2D(self.gray.astype(np.float32), -1, kernel)
            local_threshold = local_mean - c_constant
            
            threshold_variation = np.std(local_threshold)
            mean_local_threshold = np.mean(local_threshold)
            
            # Calculate statistics
            foreground_pixels = np.sum(thresholded == max_value)
            background_pixels = np.sum(thresholded == 0)
            total_pixels = self.width * self.height
            
            return {
                'thresholded_image': thresholded,
                'parameters': {
                    'method': method,
                    'threshold_type': threshold_type,
                    'block_size': block_size,
                    'c_constant': c_constant,
                    'max_value': max_value
                },
                'statistics': {
                    'foreground_pixels': int(foreground_pixels),
                    'background_pixels': int(background_pixels),
                    'foreground_percentage': float((foreground_pixels / total_pixels) * 100),
                    'background_percentage': float((background_pixels / total_pixels) * 100),
                    'threshold_variation': float(threshold_variation),
                    'mean_local_threshold': float(mean_local_threshold),
                    'min_local_threshold': float(np.min(local_threshold)),
                    'max_local_threshold': float(np.max(local_threshold)),
                    'local_threshold_stats_only': True
                },
                'description': f'Adaptive {method} thresholding with {block_size}x{block_size} blocks',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply adaptive threshold: {str(e)}'}
    
    def apply_triangle_threshold(self) -> Dict[str, Any]:
        """Apply Triangle thresholding algorithm"""
        try:
            # Calculate histogram
            hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256]).flatten()
            
            # Find Triangle threshold
            # This is based on the geometric interpretation of the histogram
            def triangle_threshold(histogram):
                # Find the peak of the histogram
                peak_idx = np.argmax(histogram)
                
                # Find the tail (rightmost non-zero bin)
                tail_idx = len(histogram) - 1
                while tail_idx > 0 and histogram[tail_idx] == 0:
                    tail_idx -= 1
                
                if peak_idx == tail_idx:
                    return peak_idx
                
                # Calculate distances from line connecting peak to tail
                max_distance = 0
                threshold_val = peak_idx
                
                for i in range(peak_idx, tail_idx + 1):
                    # Distance from point to line
                    x1, y1 = peak_idx, histogram[peak_idx]
                    x2, y2 = tail_idx, histogram[tail_idx]
                    x0, y0 = i, histogram[i]
                    
                    # Line equation: ax + by + c = 0
                    a = y2 - y1
                    b = x1 - x2
                    c = x2 * y1 - x1 * y2
                    
                    distance = abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
                    
                    if distance > max_distance:
                        max_distance = distance
                        threshold_val = i
                
                return threshold_val
            
            triangle_thresh = triangle_threshold(hist)
            
            # Apply threshold
            _, thresholded = cv2.threshold(self.gray, float(triangle_thresh), 255, cv2.THRESH_BINARY)
            
            # Calculate statistics
            foreground_pixels = np.sum(thresholded == 255)
            total_pixels = self.width * self.height
            
            return {
                'thresholded_image': thresholded,
                'triangle_threshold': int(triangle_thresh),
                'histogram': hist.tolist(),
                'statistics': {
                    'foreground_pixels': int(foreground_pixels),
                    'foreground_percentage': float((foreground_pixels / total_pixels) * 100),
                    'histogram_peak': int(np.argmax(hist)),
                    'histogram_peak_value': int(np.max(hist))
                },
                'description': f'Triangle thresholding at {triangle_thresh}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Triangle threshold: {str(e)}'}
    
    def apply_li_threshold(self) -> Dict[str, Any]:
        """Apply Li's minimum cross entropy thresholding"""
        try:
            # Use scikit-image for Li threshold
            li_thresh = filters.threshold_li(self.gray)
            
            # Apply threshold
            _, thresholded = cv2.threshold(self.gray, li_thresh, 255, cv2.THRESH_BINARY)
            
            # Calculate cross entropy
            def cross_entropy(image, threshold):
                fg_pixels = image[image > threshold]
                bg_pixels = image[image <= threshold]
                
                if len(fg_pixels) == 0 or len(bg_pixels) == 0:
                    return float('inf')
                
                fg_mean = np.mean(fg_pixels)
                bg_mean = np.mean(bg_pixels)
                
                # Avoid log(0)
                if fg_mean == 0:
                    fg_mean = 1
                if bg_mean == 0:
                    bg_mean = 1
                
                cross_ent = -np.sum(fg_pixels * np.log(fg_pixels / fg_mean))
                cross_ent += -np.sum(bg_pixels * np.log(bg_pixels / bg_mean))
                
                return cross_ent
            
            entropy_value = cross_entropy(self.gray.astype(np.float64), li_thresh)
            
            # Calculate statistics
            foreground_pixels = np.sum(thresholded == 255)
            total_pixels = self.width * self.height
            
            return {
                'thresholded_image': thresholded,
                'li_threshold': float(li_thresh),
                'cross_entropy': float(entropy_value),
                'statistics': {
                    'foreground_pixels': int(foreground_pixels),
                    'foreground_percentage': float((foreground_pixels / total_pixels) * 100)
                },
                'description': f"Li's minimum cross entropy thresholding at {li_thresh:.1f}",
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Li threshold: {str(e)}'}
    
    def apply_multi_threshold(self, levels: int = 3) -> Dict[str, Any]:
        """Apply multi-level thresholding"""
        try:
            if levels < 2 or levels > 5:
                return {'error': 'Number of levels must be between 2 and 5'}
            
            # Calculate multiple thresholds using Otsu for each level
            thresholds = []
            
            # Use histogram to find multiple thresholds
            hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256]).flatten()
            
            # Divide intensity range into levels
            intensity_range = 256
            step = intensity_range // levels
            
            for i in range(1, levels):
                # Find threshold in each segment
                start_idx = max(0, i * step - step // 2)
                end_idx = min(256, i * step + step // 2)
                
                segment_hist = hist[start_idx:end_idx]
                if np.sum(segment_hist) > 0:
                    # Find local maximum as threshold
                    local_max = np.argmax(segment_hist)
                    threshold = start_idx + local_max
                    thresholds.append(threshold)
                else:
                    thresholds.append(i * step)
            
            # Sort thresholds
            thresholds = sorted(thresholds)
            
            # Create multi-level thresholded image
            thresholded = np.zeros_like(self.gray)
            
            for i, thresh in enumerate(thresholds):
                if i == 0:
                    mask = self.gray <= thresh
                    thresholded[mask] = int(255 / (levels - 1) * i)
                else:
                    mask = (self.gray > thresholds[i-1]) & (self.gray <= thresh)
                    thresholded[mask] = int(255 / (levels - 1) * i)
            
            # Handle highest level
            mask = self.gray > thresholds[-1]
            thresholded[mask] = 255
            
            # Calculate statistics for each level
            level_stats = []
            for i in range(levels):
                level_value = int(255 / (levels - 1) * i)
                level_pixels = np.sum(thresholded == level_value)
                level_percentage = (level_pixels / (self.width * self.height)) * 100
                
                level_stats.append({
                    'level': i,
                    'value': level_value,
                    'pixels': int(level_pixels),
                    'percentage': float(level_percentage)
                })
            
            return {
                'thresholded_image': thresholded,
                'thresholds': thresholds,
                'levels': levels,
                'level_statistics': level_stats,
                'description': f'Multi-level thresholding with {levels} levels',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply multi-threshold: {str(e)}'}
    
    def compare_thresholding_methods(self) -> Dict[str, Any]:
        """Compare different thresholding methods"""
        try:
            methods_results = {}
            
            # Apply different methods
            global_result = self.apply_global_threshold(127)
            otsu_result = self.apply_otsu_threshold()
            adaptive_result = self.apply_adaptive_threshold()
            triangle_result = self.apply_triangle_threshold()
            
            if global_result.get('success'):
                methods_results['global'] = {
                    'image': global_result['thresholded_image'],
                    'threshold': global_result['threshold_value'],
                    'foreground_percentage': global_result['statistics']['foreground_percentage']
                }
            
            if otsu_result.get('success'):
                methods_results['otsu'] = {
                    'image': otsu_result['thresholded_image'],
                    'threshold': otsu_result['otsu_threshold'],
                    'foreground_percentage': otsu_result['statistics']['foreground_percentage']
                }
            
            if adaptive_result.get('success'):
                methods_results['adaptive'] = {
                    'image': adaptive_result['thresholded_image'],
                    'threshold': 'variable',
                    'foreground_percentage': adaptive_result['statistics']['foreground_percentage']
                }
            
            if triangle_result.get('success'):
                methods_results['triangle'] = {
                    'image': triangle_result['thresholded_image'],
                    'threshold': triangle_result['triangle_threshold'],
                    'foreground_percentage': triangle_result['statistics']['foreground_percentage']
                }
            
            # Find best method based on foreground/background balance
            best_method = None
            best_balance = float('inf')
            
            for method, result in methods_results.items():
                fg_percent = result['foreground_percentage']
                # Ideal balance is around 30-70% or 70-30%
                balance_score = min(abs(fg_percent - 30), abs(fg_percent - 70))
                
                if balance_score < best_balance:
                    best_balance = balance_score
                    best_method = method
            
            return {
                'methods': methods_results,
                'best_method': best_method,
                'best_balance_score': float(best_balance),
                'comparison_criteria': 'Foreground/background balance',
                'description': 'Comparison of thresholding methods',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to compare thresholding methods: {str(e)}'}