import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

class ImageComparison:
    """Module for comparing two images with various metrics"""
    
    def __init__(self, image_path1: str, image_path2: str):
        """Initialize with two image paths"""
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        
        self.image1 = cv2.imread(image_path1)
        self.image2 = cv2.imread(image_path2)
        
        if self.image1 is None or self.image2 is None:
            raise ValueError("Cannot load one or both images")
        
        self.image1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        self.image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        self.shape1 = self.image1.shape
        self.shape2 = self.image2.shape
        
        # Resize images to same dimensions for comparison
        self._prepare_images_for_comparison()
    
    def _prepare_images_for_comparison(self):
        """Prepare images for comparison by resizing to same dimensions"""
        try:
            # Find common dimensions (use the smaller of the two)
            min_height = min(self.shape1[0], self.shape2[0])
            min_width = min(self.shape1[1], self.shape2[1])
            
            # Resize both images
            self.image1_resized = cv2.resize(self.image1, (min_width, min_height))
            self.image2_resized = cv2.resize(self.image2, (min_width, min_height))
            
            self.image1_resized_rgb = cv2.cvtColor(self.image1_resized, cv2.COLOR_BGR2RGB)
            self.image2_resized_rgb = cv2.cvtColor(self.image2_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to grayscale for some metrics
            self.gray1 = cv2.cvtColor(self.image1_resized, cv2.COLOR_BGR2GRAY)
            self.gray2 = cv2.cvtColor(self.image2_resized, cv2.COLOR_BGR2GRAY)
            
        except Exception as e:
            raise ValueError(f"Error preparing images for comparison: {str(e)}")
    
    def calculate_ssim(self, multichannel: bool = True) -> Dict[str, float]:
        """Calculate Structural Similarity Index (SSIM)"""
        try:
            if multichannel and len(self.image1_resized.shape) == 3:
                # Multi-channel SSIM
                ssim_value = ssim(
                    self.image1_resized_rgb, 
                    self.image2_resized_rgb, 
                    multichannel=True,
                    channel_axis=2
                )
            else:
                # Single channel SSIM
                ssim_value = ssim(self.gray1, self.gray2)
            
            return {
                'ssim': float(ssim_value),
                'similarity_percentage': float(ssim_value * 100),
                'interpretation': self._interpret_ssim(ssim_value)
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate SSIM: {str(e)}'}
    
    def _interpret_ssim(self, ssim_value: float) -> str:
        """Interpret SSIM value"""
        if ssim_value >= 0.95:
            return "Nearly identical"
        elif ssim_value >= 0.8:
            return "Very similar"
        elif ssim_value >= 0.6:
            return "Moderately similar"
        elif ssim_value >= 0.4:
            return "Somewhat similar"
        elif ssim_value >= 0.2:
            return "Slightly similar"
        else:
            return "Very different"
    
    def calculate_psnr(self) -> Dict[str, float]:
        """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
        try:
            # Calculate PSNR for each channel
            psnr_values = []
            
            if len(self.image1_resized.shape) == 3:
                for i in range(self.image1_resized.shape[2]):
                    psnr_val = psnr(
                        self.image1_resized[:, :, i],
                        self.image2_resized[:, :, i]
                    )
                    psnr_values.append(float(psnr_val))
                
                psnr_avg = np.mean(psnr_values)
            else:
                psnr_avg = psnr(self.gray1, self.gray2)
                psnr_values = [float(psnr_avg)]
            
            return {
                'psnr': float(psnr_avg),
                'psnr_per_channel': psnr_values,
                'interpretation': self._interpret_psnr(psnr_avg)
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate PSNR: {str(e)}'}
    
    def _interpret_psnr(self, psnr_value: float) -> str:
        """Interpret PSNR value"""
        if psnr_value >= 50:
            return "Excellent quality (nearly identical)"
        elif psnr_value >= 40:
            return "Good quality (very similar)"
        elif psnr_value >= 30:
            return "Acceptable quality (moderately similar)"
        elif psnr_value >= 20:
            return "Poor quality (noticeably different)"
        else:
            return "Very poor quality (very different)"
    
    def calculate_mse(self) -> Dict[str, float]:
        """Calculate Mean Squared Error (MSE)"""
        try:
            mse_values = []
            
            if len(self.image1_resized.shape) == 3:
                for i in range(self.image1_resized.shape[2]):
                    mse_val = mse(
                        self.image1_resized[:, :, i],
                        self.image2_resized[:, :, i]
                    )
                    mse_values.append(float(mse_val))
                
                mse_avg = np.mean(mse_values)
            else:
                mse_avg = mse(self.gray1, self.gray2)
                mse_values = [float(mse_avg)]
            
            return {
                'mse': float(mse_avg),
                'mse_per_channel': mse_values,
                'rmse': float(np.sqrt(mse_avg)),
                'interpretation': self._interpret_mse(mse_avg)
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate MSE: {str(e)}'}
    
    def _interpret_mse(self, mse_value: float) -> str:
        """Interpret MSE value"""
        if mse_value <= 10:
            return "Very similar images"
        elif mse_value <= 100:
            return "Moderately similar images"
        elif mse_value <= 1000:
            return "Somewhat different images"
        else:
            return "Very different images"
    
    def calculate_histogram_comparison(self) -> Dict[str, Any]:
        """Compare histograms of the two images"""
        try:
            comparison_results = {}
            
            if len(self.image1_resized.shape) == 3:
                # Color image comparison
                colors = ['blue', 'green', 'red']
                for i, color in enumerate(colors):
                    hist1 = cv2.calcHist([self.image1_resized], [i], None, [256], [0, 256])
                    hist2 = cv2.calcHist([self.image2_resized], [i], None, [256], [0, 256])
                    
                    # Normalize histograms
                    hist1 = hist1 / np.sum(hist1)
                    hist2 = hist2 / np.sum(hist2)
                    
                    # Calculate correlation
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                    bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                    
                    comparison_results[color] = {
                        'correlation': float(correlation),
                        'chi_square': float(chi_square),
                        'bhattacharyya': float(bhattacharyya)
                    }
            
            # Grayscale comparison
            hist1_gray = cv2.calcHist([self.gray1], [0], None, [256], [0, 256])
            hist2_gray = cv2.calcHist([self.gray2], [0], None, [256], [0, 256])
            
            hist1_gray = hist1_gray / np.sum(hist1_gray)
            hist2_gray = hist2_gray / np.sum(hist2_gray)
            
            correlation_gray = cv2.compareHist(hist1_gray, hist2_gray, cv2.HISTCMP_CORREL)
            chi_square_gray = cv2.compareHist(hist1_gray, hist2_gray, cv2.HISTCMP_CHISQR)
            bhattacharyya_gray = cv2.compareHist(hist1_gray, hist2_gray, cv2.HISTCMP_BHATTACHARYYA)
            
            comparison_results['grayscale'] = {
                'correlation': float(correlation_gray),
                'chi_square': float(chi_square_gray),
                'bhattacharyya': float(bhattacharyya_gray)
            }
            
            return comparison_results
        
        except Exception as e:
            return {'error': f'Failed to compare histograms: {str(e)}'}
    
    def generate_difference_map(self, output_path: str) -> bool:
        """Generate a visual difference map"""
        try:
            # Calculate absolute difference
            diff = cv2.absdiff(self.image1_resized, self.image2_resized)
            
            # Create a heatmap of differences
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            # Create a side-by-side comparison
            height, width = diff_gray.shape
            comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)
            
            # Original images
            comparison[:, :width] = self.image1_resized_rgb
            comparison[:, width:2*width] = self.image2_resized_rgb
            comparison[:, 2*width:] = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
            
            # Save the comparison
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(output_path, comparison_bgr)
        
        except Exception as e:
            print(f"Error generating difference map: {str(e)}")
            return False
    
    def compare_images(self) -> Dict[str, Any]:
        """Perform comprehensive image comparison"""
        try:
            # Basic information
            basic_info = {
                'image1_dimensions': self.shape1,
                'image2_dimensions': self.shape2,
                'comparison_dimensions': (self.image1_resized.shape[1], self.image1_resized.shape[0]),
                'size_difference': {
                    'width': abs(self.shape1[1] - self.shape2[1]),
                    'height': abs(self.shape1[0] - self.shape2[0])
                }
            }
            
            # Calculate all metrics
            ssim_results = self.calculate_ssim()
            psnr_results = self.calculate_psnr()
            mse_results = self.calculate_mse()
            histogram_results = self.calculate_histogram_comparison()
            
            # Overall similarity assessment
            overall_similarity = self._calculate_overall_similarity(ssim_results, psnr_results, mse_results)
            
            return {
                'basic_info': basic_info,
                'ssim': ssim_results,
                'psnr': psnr_results,
                'mse': mse_results,
                'histogram_comparison': histogram_results,
                'overall_similarity': overall_similarity
            }
        
        except Exception as e:
            return {'error': f'Failed to perform comprehensive comparison: {str(e)}'}
    
    def _calculate_overall_similarity(self, ssim_results: Dict, psnr_results: Dict, mse_results: Dict) -> Dict[str, Any]:
        """Calculate overall similarity score"""
        try:
            # Extract values, handle errors
            ssim_val = ssim_results.get('ssim', 0) if 'error' not in ssim_results else 0
            psnr_val = psnr_results.get('psnr', 0) if 'error' not in psnr_results else 0
            mse_val = mse_results.get('mse', float('inf')) if 'error' not in mse_results else float('inf')
            
            # Normalize PSNR to 0-1 range (assume max useful PSNR is 50)
            psnr_normalized = min(psnr_val / 50.0, 1.0) if psnr_val > 0 else 0
            
            # Normalize MSE to 0-1 range (lower is better, assume max useful MSE is 10000)
            mse_normalized = max(0, 1.0 - (mse_val / 10000.0)) if mse_val != float('inf') else 0
            
            # Calculate weighted average
            weights = [0.5, 0.3, 0.2]  # SSIM, PSNR, MSE
            overall_score = (
                weights[0] * ssim_val +
                weights[1] * psnr_normalized +
                weights[2] * mse_normalized
            )
            
            return {
                'score': float(overall_score),
                'percentage': float(overall_score * 100),
                'interpretation': self._interpret_overall_similarity(overall_score)
            }
        
        except Exception as e:
            return {'error': f'Failed to calculate overall similarity: {str(e)}'}
    
    def _interpret_overall_similarity(self, score: float) -> str:
        """Interpret overall similarity score"""
        if score >= 0.9:
            return "Images are nearly identical"
        elif score >= 0.7:
            return "Images are very similar"
        elif score >= 0.5:
            return "Images are moderately similar"
        elif score >= 0.3:
            return "Images are somewhat similar"
        elif score >= 0.1:
            return "Images are slightly similar"
        else:
            return "Images are very different"
