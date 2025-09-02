import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt

class HistogramEnhancement:
    """Advanced histogram analysis and contrast enhancement module"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def calculate_histogram(self, color_space: str = 'rgb') -> Dict[str, Any]:
        """Calculate and analyze image histogram"""
        try:
            if color_space.lower() == 'rgb':
                image_data = self.image_rgb
                channels = ['Red', 'Green', 'Blue']
                channel_data = cv2.split(image_data)
            elif color_space.lower() == 'hsv':
                image_data = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                channels = ['Hue', 'Saturation', 'Value']
                channel_data = cv2.split(image_data)
            elif color_space.lower() == 'gray':
                image_data = self.gray
                channels = ['Gray']
                channel_data = [self.gray]
            else:
                return {'error': f'Unsupported color space: {color_space}'}
            
            histograms = {}
            histogram_stats = {}
            
            for i, (channel, name) in enumerate(zip(channel_data, channels)):
                # Calculate histogram
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                hist_normalized = hist / hist.sum()
                
                # Calculate statistics
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                median_val = np.median(channel)
                mode_val = np.argmax(hist)
                
                # Calculate entropy
                hist_norm = hist_normalized[hist_normalized > 0]
                entropy = -np.sum(hist_norm * np.log2(hist_norm))
                
                # Calculate contrast metrics
                dynamic_range = np.max(channel) - np.min(channel)
                rms_contrast = np.sqrt(np.mean((channel - mean_val) ** 2))
                
                histograms[name.lower()] = {
                    'histogram': hist.flatten().tolist(),
                    'histogram_normalized': hist_normalized.flatten().tolist(),
                    'statistics': {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'median': float(median_val),
                        'mode': int(mode_val),
                        'min': int(np.min(channel)),
                        'max': int(np.max(channel)),
                        'dynamic_range': int(dynamic_range),
                        'rms_contrast': float(rms_contrast),
                        'entropy': float(entropy)
                    }
                }
            
            # Overall image statistics
            overall_brightness = np.mean(self.gray)
            overall_contrast = np.std(self.gray)
            
            return {
                'color_space': color_space.upper(),
                'channels': histograms,
                'overall_statistics': {
                    'brightness': float(overall_brightness),
                    'contrast': float(overall_contrast),
                    'is_low_contrast': overall_contrast < 50,
                    'is_high_contrast': overall_contrast > 150,
                    'is_dark': overall_brightness < 85,
                    'is_bright': overall_brightness > 170
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate histogram: {str(e)}'}
    
    def apply_histogram_equalization(self, method: str = 'global') -> Dict[str, Any]:
        """Apply histogram equalization for contrast enhancement"""
        try:
            if method.lower() == 'global':
                # Global histogram equalization on grayscale
                equalized_gray = cv2.equalizeHist(self.gray)
                
                # For color images, work in YUV space
                if self.channels == 3:
                    yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                    equalized_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    equalized = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)
                else:
                    equalized = equalized_gray
                
                method_description = 'Global histogram equalization'
                
            elif method.lower() == 'clahe':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized_gray = clahe.apply(self.gray)
                
                if self.channels == 3:
                    yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
                    equalized_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    equalized = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)
                else:
                    equalized = equalized_gray
                
                method_description = 'CLAHE - Contrast Limited Adaptive Histogram Equalization'
                
            else:
                return {'error': f'Unsupported method: {method}'}
            
            # Calculate improvement metrics
            original_contrast = np.std(self.gray)
            enhanced_contrast = np.std(equalized_gray)
            contrast_improvement = (enhanced_contrast - original_contrast) / original_contrast * 100
            
            # Calculate histogram before and after
            hist_original = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
            hist_equalized = cv2.calcHist([equalized_gray], [0], None, [256], [0, 256])
            
            return {
                'enhanced_image': equalized,
                'enhanced_gray': equalized_gray,
                'method': method.upper(),
                'method_description': method_description,
                'histograms': {
                    'original': hist_original.flatten().tolist(),
                    'enhanced': hist_equalized.flatten().tolist()
                },
                'metrics': {
                    'original_contrast': float(original_contrast),
                    'enhanced_contrast': float(enhanced_contrast),
                    'contrast_improvement_percentage': float(contrast_improvement),
                    'original_dynamic_range': int(np.max(self.gray) - np.min(self.gray)),
                    'enhanced_dynamic_range': int(np.max(equalized_gray) - np.min(equalized_gray))
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply histogram equalization: {str(e)}'}
    
    def apply_clahe_advanced(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Dict[str, Any]:
        """Apply advanced CLAHE with custom parameters"""
        try:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            # Apply to grayscale
            enhanced_gray = clahe.apply(self.gray)
            
            # Apply to color image
            if self.channels == 3:
                # Convert to LAB for better results
                lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            else:
                enhanced = enhanced_gray
            
            # Calculate metrics
            original_contrast = np.std(self.gray)
            enhanced_contrast = np.std(enhanced_gray)
            
            # Calculate local contrast improvement
            local_std_original = ndimage.generic_filter(self.gray.astype(np.float32), np.std, size=tile_grid_size)
            local_std_enhanced = ndimage.generic_filter(enhanced_gray.astype(np.float32), np.std, size=tile_grid_size)
            local_improvement = np.mean(local_std_enhanced) / np.mean(local_std_original) * 100 - 100
            
            return {
                'enhanced_image': enhanced,
                'enhanced_gray': enhanced_gray,
                'parameters': {
                    'clip_limit': clip_limit,
                    'tile_grid_size': tile_grid_size
                },
                'metrics': {
                    'global_contrast_improvement': float((enhanced_contrast - original_contrast) / original_contrast * 100),
                    'local_contrast_improvement': float(local_improvement),
                    'original_entropy': float(self._calculate_entropy(self.gray)),
                    'enhanced_entropy': float(self._calculate_entropy(enhanced_gray))
                },
                'description': f'Advanced CLAHE with clip limit {clip_limit} and {tile_grid_size[0]}x{tile_grid_size[1]} tiles',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply advanced CLAHE: {str(e)}'}
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return entropy
    
    def apply_gamma_correction(self, gamma: float = 1.0) -> Dict[str, Any]:
        """Apply gamma correction for brightness adjustment"""
        try:
            if gamma <= 0:
                gamma = 0.1
            
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction
            corrected = cv2.LUT(self.image_rgb, table)
            corrected_gray = cv2.LUT(self.gray, table)
            
            # Calculate brightness change
            original_brightness = np.mean(self.gray)
            corrected_brightness = np.mean(corrected_gray)
            brightness_change = (corrected_brightness - original_brightness) / original_brightness * 100
            
            return {
                'corrected_image': corrected,
                'corrected_gray': corrected_gray,
                'gamma': gamma,
                'lookup_table': table.tolist(),
                'metrics': {
                    'original_brightness': float(original_brightness),
                    'corrected_brightness': float(corrected_brightness),
                    'brightness_change_percentage': float(brightness_change)
                },
                'description': f'Gamma correction with γ={gamma}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply gamma correction: {str(e)}'}
    
    def stretch_histogram(self, lower_percentile: float = 2, upper_percentile: float = 98) -> Dict[str, Any]:
        """Apply histogram stretching for improved contrast"""
        try:
            # Calculate percentiles
            lower_bound = np.percentile(self.gray, lower_percentile)
            upper_bound = np.percentile(self.gray, upper_percentile)
            
            # Avoid division by zero
            if upper_bound == lower_bound:
                upper_bound = lower_bound + 1
            
            # Apply stretching
            stretched_gray = np.clip((self.gray - lower_bound) / (upper_bound - lower_bound) * 255, 0, 255).astype(np.uint8)
            
            # Apply to color image
            if self.channels == 3:
                stretched = np.zeros_like(self.image_rgb)
                for i in range(3):
                    channel = self.image_rgb[:, :, i]
                    ch_lower = np.percentile(channel, lower_percentile)
                    ch_upper = np.percentile(channel, upper_percentile)
                    
                    if ch_upper == ch_lower:
                        ch_upper = ch_lower + 1
                    
                    stretched[:, :, i] = np.clip((channel - ch_lower) / (ch_upper - ch_lower) * 255, 0, 255)
                
                stretched = stretched.astype(np.uint8)
            else:
                stretched = stretched_gray
            
            # Calculate improvement
            original_dynamic_range = np.max(self.gray) - np.min(self.gray)
            stretched_dynamic_range = np.max(stretched_gray) - np.min(stretched_gray)
            
            return {
                'stretched_image': stretched,
                'stretched_gray': stretched_gray,
                'parameters': {
                    'lower_percentile': lower_percentile,
                    'upper_percentile': upper_percentile,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'metrics': {
                    'original_dynamic_range': int(original_dynamic_range),
                    'stretched_dynamic_range': int(stretched_dynamic_range),
                    'range_improvement': float((stretched_dynamic_range - original_dynamic_range) / original_dynamic_range * 100)
                },
                'description': f'Histogram stretching using {lower_percentile}-{upper_percentile} percentile range',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply histogram stretching: {str(e)}'}
    
    def apply_histogram_matching(self, reference_image_path: str) -> Dict[str, Any]:
        """Apply histogram matching to match reference image"""
        try:
            # Load reference image
            ref_image = cv2.imread(reference_image_path)
            if ref_image is None:
                return {'error': 'Cannot load reference image'}
            
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate CDFs
            def calculate_cdf(image):
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf.max()
                return cdf_normalized
            
            # Get CDFs
            source_cdf = calculate_cdf(self.gray)
            ref_cdf = calculate_cdf(ref_gray)
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest value in reference CDF
                diff = np.abs(ref_cdf - source_cdf[i])
                mapping[i] = np.argmin(diff)
            
            # Apply mapping
            matched_gray = cv2.LUT(self.gray, mapping)
            
            # Apply to color image
            if self.channels == 3:
                # Convert to LAB and match L channel
                lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.LUT(lab[:, :, 0], mapping)
                matched_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                matched = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)
            else:
                matched = matched_gray
            
            # Calculate similarity
            hist_source = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
            hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            hist_matched = cv2.calcHist([matched_gray], [0], None, [256], [0, 256])
            
            # Normalize histograms
            hist_source = hist_source / hist_source.sum()
            hist_ref = hist_ref / hist_ref.sum()
            hist_matched = hist_matched / hist_matched.sum()
            
            # Calculate correlation
            correlation_before = cv2.compareHist(hist_source, hist_ref, cv2.HISTCMP_CORREL)
            correlation_after = cv2.compareHist(hist_matched, hist_ref, cv2.HISTCMP_CORREL)
            
            return {
                'matched_image': matched,
                'matched_gray': matched_gray,
                'mapping_function': mapping.tolist(),
                'histograms': {
                    'source': hist_source.flatten().tolist(),
                    'reference': hist_ref.flatten().tolist(),
                    'matched': hist_matched.flatten().tolist()
                },
                'metrics': {
                    'correlation_before': float(correlation_before),
                    'correlation_after': float(correlation_after),
                    'improvement': float(correlation_after - correlation_before)
                },
                'description': 'Histogram matching to reference image',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply histogram matching: {str(e)}'}