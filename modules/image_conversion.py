import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os

class ImageConversion:
    """Module for image format conversions and transformations"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def to_grayscale(self, output_path: str, method: str = 'weighted') -> bool:
        """Convert image to grayscale"""
        try:
            if method == 'weighted':
                # Standard weighted conversion
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            elif method == 'average':
                # Simple average of RGB channels
                gray = np.mean(self.image, axis=2).astype(np.uint8)
            elif method == 'luminance':
                # Using luminance formula
                gray = 0.299 * self.image[:, :, 2] + 0.587 * self.image[:, :, 1] + 0.114 * self.image[:, :, 0]
                gray = gray.astype(np.uint8)
            else:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            return cv2.imwrite(output_path, gray)
        
        except Exception as e:
            print(f"Error converting to grayscale: {str(e)}")
            return False
    
    def to_binary(self, output_path: str, threshold: int = 127, method: str = 'global') -> bool:
        """Convert image to binary"""
        try:
            # First convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            if method == 'global':
                # Global thresholding
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            elif method == 'otsu':
                # Otsu's thresholding
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'adaptive':
                # Adaptive thresholding
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            return cv2.imwrite(output_path, binary)
        
        except Exception as e:
            print(f"Error converting to binary: {str(e)}")
            return False
    
    def to_hsv(self, output_path: str) -> bool:
        """Convert image to HSV color space"""
        try:
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            return cv2.imwrite(output_path, hsv)
        
        except Exception as e:
            print(f"Error converting to HSV: {str(e)}")
            return False
    
    def to_lab(self, output_path: str) -> bool:
        """Convert image to LAB color space"""
        try:
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            return cv2.imwrite(output_path, lab)
        
        except Exception as e:
            print(f"Error converting to LAB: {str(e)}")
            return False
    
    def adjust_brightness(self, output_path: str, factor: float = 1.0) -> bool:
        """Adjust image brightness"""
        try:
            if factor < 0:
                factor = 0
            elif factor > 3:
                factor = 3
            
            # Convert to float for calculations
            bright_image = self.image.astype(np.float32)
            bright_image = bright_image * factor
            
            # Clip values to valid range
            bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
            
            return cv2.imwrite(output_path, bright_image)
        
        except Exception as e:
            print(f"Error adjusting brightness: {str(e)}")
            return False
    
    def adjust_contrast(self, output_path: str, factor: float = 1.0) -> bool:
        """Adjust image contrast"""
        try:
            if factor < 0:
                factor = 0
            elif factor > 3:
                factor = 3
            
            # Convert to float for calculations
            contrast_image = self.image.astype(np.float32)
            
            # Apply contrast adjustment
            contrast_image = (contrast_image - 127.5) * factor + 127.5
            
            # Clip values to valid range
            contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
            
            return cv2.imwrite(output_path, contrast_image)
        
        except Exception as e:
            print(f"Error adjusting contrast: {str(e)}")
            return False
    
    def apply_gamma_correction(self, output_path: str, gamma: float = 1.0) -> bool:
        """Apply gamma correction"""
        try:
            if gamma <= 0:
                gamma = 0.1
            elif gamma > 3:
                gamma = 3
            
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            gamma_corrected = cv2.LUT(self.image, table)
            
            return cv2.imwrite(output_path, gamma_corrected)
        
        except Exception as e:
            print(f"Error applying gamma correction: {str(e)}")
            return False
    
    def apply_histogram_equalization(self, output_path: str) -> bool:
        """Apply histogram equalization"""
        try:
            if self.channels == 1:
                # Grayscale image
                equalized = cv2.equalizeHist(self.image)
            else:
                # Color image - convert to YUV, equalize Y channel, convert back
                yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            return cv2.imwrite(output_path, equalized)
        
        except Exception as e:
            print(f"Error applying histogram equalization: {str(e)}")
            return False
    
    def apply_gaussian_blur(self, output_path: str, kernel_size: int = 5, sigma: float = 1.0) -> bool:
        """Apply Gaussian blur"""
        try:
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            
            if kernel_size < 3:
                kernel_size = 3
            elif kernel_size > 31:
                kernel_size = 31
            
            blurred = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma)
            return cv2.imwrite(output_path, blurred)
        
        except Exception as e:
            print(f"Error applying Gaussian blur: {str(e)}")
            return False
    
    def apply_edge_detection(self, output_path: str, method: str = 'canny') -> bool:
        """Apply edge detection"""
        try:
            # Convert to grayscale first
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            if method == 'canny':
                edges = cv2.Canny(gray, 50, 150)
            elif method == 'sobel':
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobel_x**2 + sobel_y**2)
                edges = np.uint8(edges / edges.max() * 255)
            elif method == 'laplacian':
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))
            else:
                edges = cv2.Canny(gray, 50, 150)
            
            return cv2.imwrite(output_path, edges)
        
        except Exception as e:
            print(f"Error applying edge detection: {str(e)}")
            return False
    
    def resize_image(self, output_path: str, width: int, height: int, interpolation: str = 'linear') -> bool:
        """Resize image"""
        try:
            if width <= 0 or height <= 0:
                return False
            
            # Map interpolation methods
            interp_methods = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4
            }
            
            interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)
            resized = cv2.resize(self.image, (width, height), interpolation=interp)
            
            return cv2.imwrite(output_path, resized)
        
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return False
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about available conversions"""
        return {
            'supported_formats': ['grayscale', 'binary', 'hsv', 'lab'],
            'supported_adjustments': ['brightness', 'contrast', 'gamma', 'histogram_equalization'],
            'supported_filters': ['gaussian_blur', 'edge_detection'],
            'supported_methods': {
                'grayscale': ['weighted', 'average', 'luminance'],
                'binary': ['global', 'otsu', 'adaptive'],
                'edge_detection': ['canny', 'sobel', 'laplacian'],
                'interpolation': ['nearest', 'linear', 'cubic', 'lanczos']
            }
        }
