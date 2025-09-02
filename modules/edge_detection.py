import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import ndimage

class EdgeDetection:
    """Advanced edge detection algorithms module"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for edge detection
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def apply_sobel(self, ksize: int = 3, dx: int = 1, dy: int = 1) -> Dict[str, Any]:
        """Apply Sobel edge detection"""
        try:
            # Apply Sobel in X and Y directions
            sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=ksize)
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            direction = np.arctan2(sobel_y, sobel_x)
            
            # Normalize to 0-255 range
            magnitude_norm = ((magnitude / magnitude.max()) * 255).astype(np.uint8)
            direction_norm = ((direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Individual gradients
            sobel_x_norm = np.abs(sobel_x)
            sobel_x_norm = ((sobel_x_norm / sobel_x_norm.max()) * 255).astype(np.uint8)
            sobel_y_norm = np.abs(sobel_y)
            sobel_y_norm = ((sobel_y_norm / sobel_y_norm.max()) * 255).astype(np.uint8)
            
            # Calculate edge statistics
            edge_pixels = np.sum(magnitude_norm > np.mean(magnitude_norm))
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': magnitude_norm,
                'sobel_x': sobel_x_norm,
                'sobel_y': sobel_y_norm,
                'gradient_magnitude': magnitude_norm,
                'gradient_direction': direction_norm,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'max_gradient': float(magnitude.max()),
                    'mean_gradient': float(magnitude.mean())
                },
                'parameters': {
                    'kernel_size': ksize,
                    'dx': dx,
                    'dy': dy
                },
                'algorithm': 'Sobel',
                'description': f'Sobel edge detection with {ksize}x{ksize} kernel',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Sobel edge detection: {str(e)}'}
    
    def apply_canny(self, low_threshold: int = 50, high_threshold: int = 150, 
                   aperture_size: int = 3, l2_gradient: bool = False) -> Dict[str, Any]:
        """Apply Canny edge detection"""
        try:
            # Apply Gaussian blur first
            blurred = cv2.GaussianBlur(self.gray, (5, 5), 1.4)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, low_threshold, high_threshold, 
                            apertureSize=aperture_size, L2gradient=l2_gradient)
            
            # Calculate statistics
            edge_pixels = np.sum(edges > 0)
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            # Calculate edge connectivity (how well connected edges are)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)
            
            # Average contour length
            if contours:
                contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
                avg_contour_length = np.mean(contour_lengths)
                max_contour_length = np.max(contour_lengths)
            else:
                avg_contour_length = 0
                max_contour_length = 0
            
            return {
                'edges': edges,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'num_contours': num_contours,
                    'avg_contour_length': float(avg_contour_length),
                    'max_contour_length': float(max_contour_length)
                },
                'parameters': {
                    'low_threshold': low_threshold,
                    'high_threshold': high_threshold,
                    'aperture_size': aperture_size,
                    'l2_gradient': l2_gradient
                },
                'algorithm': 'Canny',
                'description': f'Canny edge detection with thresholds [{low_threshold}, {high_threshold}]',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Canny edge detection: {str(e)}'}
    
    def apply_laplacian(self, ksize: int = 3, scale: float = 1, delta: float = 0) -> Dict[str, Any]:
        """Apply Laplacian edge detection"""
        try:
            # Apply Laplacian
            laplacian = cv2.Laplacian(self.gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
            
            # Convert to absolute values and normalize
            laplacian_abs = np.abs(laplacian)
            laplacian_norm = ((laplacian_abs / laplacian_abs.max()) * 255).astype(np.uint8)
            
            # Apply threshold to get binary edges
            threshold_value = np.mean(laplacian_norm) + np.std(laplacian_norm)
            _, edges_binary = cv2.threshold(laplacian_norm, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Calculate zero-crossings (more precise edge detection)
            zero_crossings = self._find_zero_crossings(laplacian)
            
            # Calculate statistics
            edge_pixels = np.sum(edges_binary > 0)
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': laplacian_norm,
                'edges_binary': edges_binary,
                'zero_crossings': zero_crossings,
                'raw_laplacian': laplacian,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'threshold_used': float(threshold_value),
                    'max_response': float(laplacian_abs.max()),
                    'mean_response': float(laplacian_abs.mean())
                },
                'parameters': {
                    'kernel_size': ksize,
                    'scale': scale,
                    'delta': delta
                },
                'algorithm': 'Laplacian',
                'description': f'Laplacian edge detection with {ksize}x{ksize} kernel',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Laplacian edge detection: {str(e)}'}
    
    def _find_zero_crossings(self, laplacian: np.ndarray) -> np.ndarray:
        """Find zero crossings in Laplacian for more precise edge detection"""
        zero_crossings = np.zeros_like(laplacian, dtype=np.uint8)
        
        # Check horizontal neighbors
        for i in range(1, laplacian.shape[0]):
            for j in range(laplacian.shape[1]):
                if laplacian[i-1, j] * laplacian[i, j] < 0:
                    zero_crossings[i, j] = 255
        
        # Check vertical neighbors
        for i in range(laplacian.shape[0]):
            for j in range(1, laplacian.shape[1]):
                if laplacian[i, j-1] * laplacian[i, j] < 0:
                    zero_crossings[i, j] = 255
        
        return zero_crossings
    
    def apply_scharr(self) -> Dict[str, Any]:
        """Apply Scharr edge detection (better rotation invariance than Sobel)"""
        try:
            # Apply Scharr in X and Y directions
            scharr_x = cv2.Scharr(self.gray, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(self.gray, cv2.CV_64F, 0, 1)
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
            direction = np.arctan2(scharr_y, scharr_x)
            
            # Normalize
            magnitude_norm = ((magnitude / magnitude.max()) * 255).astype(np.uint8)
            direction_norm = ((direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Individual gradients
            scharr_x_norm = np.abs(scharr_x)
            scharr_x_norm = ((scharr_x_norm / scharr_x_norm.max()) * 255).astype(np.uint8)
            scharr_y_norm = np.abs(scharr_y)
            scharr_y_norm = ((scharr_y_norm / scharr_y_norm.max()) * 255).astype(np.uint8)
            
            # Calculate statistics
            edge_pixels = np.sum(magnitude_norm > np.mean(magnitude_norm))
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': magnitude_norm,
                'scharr_x': scharr_x_norm,
                'scharr_y': scharr_y_norm,
                'gradient_magnitude': magnitude_norm,
                'gradient_direction': direction_norm,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'max_gradient': float(magnitude.max()),
                    'mean_gradient': float(magnitude.mean())
                },
                'algorithm': 'Scharr',
                'description': 'Scharr edge detection - better rotation invariance',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Scharr edge detection: {str(e)}'}
    
    def apply_prewitt(self) -> Dict[str, Any]:
        """Apply Prewitt edge detection"""
        try:
            # Define Prewitt kernels
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            
            # Apply Prewitt operators
            prewitt_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
            
            # Calculate magnitude
            magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
            magnitude_norm = ((magnitude / magnitude.max()) * 255).astype(np.uint8)
            
            # Calculate direction
            direction = np.arctan2(prewitt_y, prewitt_x)
            direction_norm = ((direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Statistics
            edge_pixels = np.sum(magnitude_norm > np.mean(magnitude_norm))
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': magnitude_norm,
                'prewitt_x': ((np.abs(prewitt_x) / np.abs(prewitt_x).max()) * 255).astype(np.uint8),
                'prewitt_y': ((np.abs(prewitt_y) / np.abs(prewitt_y).max()) * 255).astype(np.uint8),
                'gradient_magnitude': magnitude_norm,
                'gradient_direction': direction_norm,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'max_gradient': float(magnitude.max()),
                    'mean_gradient': float(magnitude.mean())
                },
                'algorithm': 'Prewitt',
                'description': 'Prewitt edge detection - simple gradient operator',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Prewitt edge detection: {str(e)}'}
    
    def apply_roberts(self) -> Dict[str, Any]:
        """Apply Roberts Cross edge detection"""
        try:
            # Define Roberts kernels
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])
            
            # Apply Roberts operators
            roberts_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
            roberts_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
            
            # Calculate magnitude
            magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
            magnitude_norm = ((magnitude / magnitude.max()) * 255).astype(np.uint8)
            
            # Statistics
            edge_pixels = np.sum(magnitude_norm > np.mean(magnitude_norm))
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': magnitude_norm,
                'roberts_x': ((np.abs(roberts_x) / np.abs(roberts_x).max()) * 255).astype(np.uint8),
                'roberts_y': ((np.abs(roberts_y) / np.abs(roberts_y).max()) * 255).astype(np.uint8),
                'gradient_magnitude': magnitude_norm,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'max_gradient': float(magnitude.max()),
                    'mean_gradient': float(magnitude.mean())
                },
                'algorithm': 'Roberts Cross',
                'description': 'Roberts Cross edge detection - diagonal gradient',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Roberts edge detection: {str(e)}'}
    
    def apply_log(self, sigma: float = 1.0) -> Dict[str, Any]:
        """Apply Laplacian of Gaussian (LoG) edge detection"""
        try:
            # Apply Gaussian blur first
            blurred = cv2.GaussianBlur(self.gray.astype(np.float32), (0, 0), sigma)
            
            # Apply Laplacian
            log = cv2.Laplacian(blurred, cv2.CV_64F)
            
            # Find zero crossings
            zero_crossings = self._find_zero_crossings(log)
            
            # Normalize Laplacian response
            log_abs = np.abs(log)
            log_norm = ((log_abs / log_abs.max()) * 255).astype(np.uint8)
            
            # Statistics
            edge_pixels = np.sum(zero_crossings > 0)
            total_pixels = self.width * self.height
            edge_density = (edge_pixels / total_pixels) * 100
            
            return {
                'edges': zero_crossings,
                'log_response': log_norm,
                'raw_log': log,
                'edge_statistics': {
                    'edge_pixels': int(edge_pixels),
                    'total_pixels': total_pixels,
                    'edge_density_percentage': float(edge_density),
                    'max_response': float(log_abs.max()),
                    'mean_response': float(log_abs.mean())
                },
                'parameters': {
                    'sigma': sigma
                },
                'algorithm': 'Laplacian of Gaussian (LoG)',
                'description': f'LoG edge detection with sigma={sigma}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply LoG edge detection: {str(e)}'}
    
    def compare_edge_detectors(self) -> Dict[str, Any]:
        """Compare different edge detection algorithms"""
        try:
            # Apply different edge detectors
            sobel_result = self.apply_sobel()
            canny_result = self.apply_canny()
            laplacian_result = self.apply_laplacian()
            scharr_result = self.apply_scharr()
            
            # Collect results
            comparison = {
                'sobel': {
                    'edges': sobel_result.get('edges'),
                    'edge_density': sobel_result.get('edge_statistics', {}).get('edge_density_percentage', 0)
                },
                'canny': {
                    'edges': canny_result.get('edges'),
                    'edge_density': canny_result.get('edge_statistics', {}).get('edge_density_percentage', 0)
                },
                'laplacian': {
                    'edges': laplacian_result.get('edges'),
                    'edge_density': laplacian_result.get('edge_statistics', {}).get('edge_density_percentage', 0)
                },
                'scharr': {
                    'edges': scharr_result.get('edges'),
                    'edge_density': scharr_result.get('edge_statistics', {}).get('edge_density_percentage', 0)
                }
            }
            
            # Find best detector based on edge density
            best_detector = max(comparison.keys(), 
                              key=lambda k: comparison[k]['edge_density'])
            
            return {
                'comparison': comparison,
                'best_detector': best_detector,
                'rankings': sorted(comparison.items(), 
                                 key=lambda x: x[1]['edge_density'], 
                                 reverse=True),
                'description': 'Comparison of edge detection algorithms',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to compare edge detectors: {str(e)}'}