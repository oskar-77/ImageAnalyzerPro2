import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json

class Masks:
    """Advanced masking operations with geometric and color-based masks"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
    
    def create_geometric_mask(self, shape: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create geometric masks (circle, rectangle, polygon)"""
        try:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            
            if shape.lower() == 'circle':
                center_x = params.get('center_x', self.width // 2)
                center_y = params.get('center_y', self.height // 2)
                radius = params.get('radius', min(self.width, self.height) // 4)
                
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                mask_info = {
                    'shape': 'circle',
                    'center': (center_x, center_y),
                    'radius': radius,
                    'area': np.pi * radius * radius
                }
            
            elif shape.lower() == 'rectangle':
                x1 = params.get('x1', self.width // 4)
                y1 = params.get('y1', self.height // 4)
                x2 = params.get('x2', 3 * self.width // 4)
                y2 = params.get('y2', 3 * self.height // 4)
                
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                mask_info = {
                    'shape': 'rectangle',
                    'top_left': (x1, y1),
                    'bottom_right': (x2, y2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                }
            
            elif shape.lower() == 'polygon':
                points = params.get('points', [])
                if len(points) >= 3:
                    pts = np.array(points, np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    
                    mask_info = {
                        'shape': 'polygon',
                        'points': points,
                        'vertex_count': len(points),
                        'area': cv2.contourArea(pts)
                    }
                else:
                    return {'error': 'Polygon needs at least 3 points'}
            
            elif shape.lower() == 'ellipse':
                center_x = params.get('center_x', self.width // 2)
                center_y = params.get('center_y', self.height // 2)
                axes_a = params.get('axes_a', self.width // 4)
                axes_b = params.get('axes_b', self.height // 4)
                angle = params.get('angle', 0)
                
                cv2.ellipse(mask, (center_x, center_y), (axes_a, axes_b), angle, 0, 360, 255, -1)
                
                mask_info = {
                    'shape': 'ellipse',
                    'center': (center_x, center_y),
                    'axes': (axes_a, axes_b),
                    'angle': angle,
                    'area': np.pi * axes_a * axes_b
                }
            
            else:
                return {'error': f'Unsupported shape: {shape}'}
            
            # Calculate mask statistics
            total_pixels = self.width * self.height
            mask_pixels = np.sum(mask == 255)
            coverage_percentage = (mask_pixels / total_pixels) * 100
            
            return {
                'mask': mask,
                'mask_info': mask_info,
                'statistics': {
                    'total_pixels': total_pixels,
                    'mask_pixels': int(mask_pixels),
                    'coverage_percentage': float(coverage_percentage),
                    'background_pixels': int(total_pixels - mask_pixels)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to create geometric mask: {str(e)}'}
    
    def create_color_mask(self, color_space: str = 'hsv', **kwargs) -> Dict[str, Any]:
        """Create color-based mask using HSV or other color spaces"""
        try:
            if color_space.lower() == 'hsv':
                image_data = self.image_hsv
                
                # Default HSV ranges for common colors if not specified
                if 'lower_bound' not in kwargs or 'upper_bound' not in kwargs:
                    color_name = kwargs.get('color', 'blue')
                    lower_bound, upper_bound = self._get_color_range(color_name)
                else:
                    lower_bound = np.array(kwargs['lower_bound'])
                    upper_bound = np.array(kwargs['upper_bound'])
                
                mask = cv2.inRange(image_data, lower_bound, upper_bound)
                
                mask_info = {
                    'color_space': 'HSV',
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist(),
                    'target_color': kwargs.get('color', 'custom')
                }
            
            elif color_space.lower() == 'rgb':
                image_data = self.image_rgb
                lower_bound = np.array(kwargs.get('lower_bound', [0, 0, 0]))
                upper_bound = np.array(kwargs.get('upper_bound', [255, 255, 255]))
                
                mask = cv2.inRange(image_data, lower_bound, upper_bound)
                
                mask_info = {
                    'color_space': 'RGB',
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist()
                }
            
            elif color_space.lower() == 'lab':
                lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
                lower_bound = np.array(kwargs.get('lower_bound', [0, 0, 0]))
                upper_bound = np.array(kwargs.get('upper_bound', [255, 255, 255]))
                
                mask = cv2.inRange(lab_image, lower_bound, upper_bound)
                
                mask_info = {
                    'color_space': 'LAB',
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist()
                }
            
            else:
                return {'error': f'Unsupported color space: {color_space}'}
            
            # Calculate statistics
            total_pixels = self.width * self.height
            mask_pixels = np.sum(mask == 255)
            coverage_percentage = (mask_pixels / total_pixels) * 100
            
            return {
                'mask': mask,
                'mask_info': mask_info,
                'statistics': {
                    'total_pixels': total_pixels,
                    'mask_pixels': int(mask_pixels),
                    'coverage_percentage': float(coverage_percentage),
                    'background_pixels': int(total_pixels - mask_pixels)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to create color mask: {str(e)}'}
    
    def _get_color_range(self, color_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get HSV range for common colors"""
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 50, 50], [180, 255, 255]),  # Red wraps around in HSV
            'green': ([50, 50, 50], [70, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 50, 50], [30, 255, 255]),
            'orange': ([10, 50, 50], [20, 255, 255]),
            'purple': ([130, 50, 50], [160, 255, 255]),
            'cyan': ([80, 50, 50], [100, 255, 255]),
            'magenta': ([160, 50, 50], [170, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 30])
        }
        
        if color_name.lower() in color_ranges:
            lower, upper = color_ranges[color_name.lower()]
            return np.array(lower), np.array(upper)
        else:
            # Default to blue if color not found
            lower, upper = color_ranges['blue']
            return np.array(lower), np.array(upper)
    
    def create_threshold_mask(self, method: str = 'global', **kwargs) -> Dict[str, Any]:
        """Create threshold-based masks"""
        try:
            # Convert to grayscale if not already
            if len(self.image.shape) == 3:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.image
            
            if method.lower() == 'global':
                threshold_value = kwargs.get('threshold', 127)
                _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                
                mask_info = {
                    'method': 'Global Threshold',
                    'threshold_value': threshold_value,
                    'threshold_type': 'BINARY'
                }
            
            elif method.lower() == 'otsu':
                threshold_value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                mask_info = {
                    'method': 'Otsu Threshold',
                    'calculated_threshold': float(threshold_value),
                    'threshold_type': 'BINARY + OTSU'
                }
            
            elif method.lower() == 'adaptive':
                block_size = kwargs.get('block_size', 11)
                c_constant = kwargs.get('c_constant', 2)
                adaptive_method = kwargs.get('adaptive_method', 'gaussian')
                
                if adaptive_method.lower() == 'gaussian':
                    method_flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                else:
                    method_flag = cv2.ADAPTIVE_THRESH_MEAN_C
                
                mask = cv2.adaptiveThreshold(gray, 255, method_flag, cv2.THRESH_BINARY, block_size, c_constant)
                
                mask_info = {
                    'method': 'Adaptive Threshold',
                    'adaptive_method': adaptive_method,
                    'block_size': block_size,
                    'c_constant': c_constant
                }
            
            else:
                return {'error': f'Unsupported threshold method: {method}'}
            
            # Calculate statistics
            total_pixels = self.width * self.height
            mask_pixels = np.sum(mask == 255)
            coverage_percentage = (mask_pixels / total_pixels) * 100
            
            return {
                'mask': mask,
                'mask_info': mask_info,
                'statistics': {
                    'total_pixels': total_pixels,
                    'mask_pixels': int(mask_pixels),
                    'coverage_percentage': float(coverage_percentage),
                    'background_pixels': int(total_pixels - mask_pixels)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to create threshold mask: {str(e)}'}
    
    def combine_masks(self, mask1: np.ndarray, mask2: np.ndarray, operation: str) -> Dict[str, Any]:
        """Combine two masks using logical operations"""
        try:
            if mask1.shape != mask2.shape:
                return {'error': 'Masks must have the same dimensions'}
            
            if operation.lower() == 'and':
                combined_mask = cv2.bitwise_and(mask1, mask2)
                operation_description = 'AND - Intersection of both masks'
            
            elif operation.lower() == 'or':
                combined_mask = cv2.bitwise_or(mask1, mask2)
                operation_description = 'OR - Union of both masks'
            
            elif operation.lower() == 'xor':
                combined_mask = cv2.bitwise_xor(mask1, mask2)
                operation_description = 'XOR - Exclusive OR of both masks'
            
            elif operation.lower() == 'not':
                combined_mask = cv2.bitwise_not(mask1)
                operation_description = 'NOT - Inverse of first mask'
            
            elif operation.lower() == 'subtract':
                combined_mask = cv2.subtract(mask1, mask2)
                operation_description = 'SUBTRACT - First mask minus second mask'
            
            else:
                return {'error': f'Unsupported operation: {operation}'}
            
            # Calculate statistics
            total_pixels = combined_mask.size
            mask_pixels = np.sum(combined_mask == 255)
            coverage_percentage = (mask_pixels / total_pixels) * 100
            
            return {
                'combined_mask': combined_mask,
                'operation': operation.upper(),
                'operation_description': operation_description,
                'statistics': {
                    'total_pixels': total_pixels,
                    'mask_pixels': int(mask_pixels),
                    'coverage_percentage': float(coverage_percentage),
                    'background_pixels': int(total_pixels - mask_pixels)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to combine masks: {str(e)}'}
    
    def apply_mask_to_image(self, mask: np.ndarray, background_color: Tuple[int, int, int] = (0, 0, 0)) -> Dict[str, Any]:
        """Apply mask to image with specified background color"""
        try:
            if len(mask.shape) != 2:
                return {'error': 'Mask must be a 2D array'}
            
            if mask.shape[:2] != (self.height, self.width):
                return {'error': 'Mask dimensions must match image dimensions'}
            
            # Create masked image
            masked_image = self.image_rgb.copy()
            
            # Apply background color to areas outside mask
            background_mask = mask == 0
            masked_image[background_mask] = background_color
            
            # Calculate statistics
            total_pixels = self.width * self.height
            visible_pixels = np.sum(mask == 255)
            hidden_pixels = total_pixels - visible_pixels
            
            return {
                'masked_image': masked_image,
                'background_color': background_color,
                'statistics': {
                    'total_pixels': total_pixels,
                    'visible_pixels': int(visible_pixels),
                    'hidden_pixels': int(hidden_pixels),
                    'visibility_percentage': float((visible_pixels / total_pixels) * 100)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply mask to image: {str(e)}'}
    
    def refine_mask(self, mask: np.ndarray, operation: str, kernel_size: int = 5) -> Dict[str, Any]:
        """Refine mask using morphological operations"""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            if operation.lower() == 'opening':
                refined_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                description = 'Opening - Removes small noise and separates connected objects'
            
            elif operation.lower() == 'closing':
                refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                description = 'Closing - Fills small holes and connects nearby objects'
            
            elif operation.lower() == 'erosion':
                refined_mask = cv2.erode(mask, kernel, iterations=1)
                description = 'Erosion - Shrinks white regions'
            
            elif operation.lower() == 'dilation':
                refined_mask = cv2.dilate(mask, kernel, iterations=1)
                description = 'Dilation - Expands white regions'
            
            elif operation.lower() == 'gradient':
                refined_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                description = 'Gradient - Shows edges of objects'
            
            else:
                return {'error': f'Unsupported morphological operation: {operation}'}
            
            # Calculate statistics
            original_pixels = np.sum(mask == 255)
            refined_pixels = np.sum(refined_mask == 255)
            change_percentage = ((refined_pixels - original_pixels) / original_pixels) * 100 if original_pixels > 0 else 0
            
            return {
                'refined_mask': refined_mask,
                'operation': operation.upper(),
                'operation_description': description,
                'kernel_size': kernel_size,
                'statistics': {
                    'original_mask_pixels': int(original_pixels),
                    'refined_mask_pixels': int(refined_pixels),
                    'change_percentage': float(change_percentage),
                    'pixels_gained': int(max(0, refined_pixels - original_pixels)),
                    'pixels_lost': int(max(0, original_pixels - refined_pixels))
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to refine mask: {str(e)}'}
    
    def analyze_mask_regions(self, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze connected regions in mask"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            regions = []
            for i in range(1, num_labels):  # Skip background (label 0)
                region_stats = {
                    'label': i,
                    'area': int(stats[i, cv2.CC_STAT_AREA]),
                    'centroid': (float(centroids[i, 0]), float(centroids[i, 1])),
                    'bounding_box': {
                        'x': int(stats[i, cv2.CC_STAT_LEFT]),
                        'y': int(stats[i, cv2.CC_STAT_TOP]),
                        'width': int(stats[i, cv2.CC_STAT_WIDTH]),
                        'height': int(stats[i, cv2.CC_STAT_HEIGHT])
                    }
                }
                regions.append(region_stats)
            
            # Sort regions by area (largest first)
            regions.sort(key=lambda x: x['area'], reverse=True)
            
            return {
                'total_regions': num_labels - 1,  # Exclude background
                'regions': regions,
                'largest_region_area': regions[0]['area'] if regions else 0,
                'smallest_region_area': regions[-1]['area'] if regions else 0,
                'average_region_area': float(np.mean([r['area'] for r in regions])) if regions else 0,
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to analyze mask regions: {str(e)}'}