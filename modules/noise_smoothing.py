import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import random

class NoiseSmoothing:
    """Advanced noise generation and smoothing filters module"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def add_gaussian_noise(self, mean: float = 0, sigma: float = 25) -> Dict[str, Any]:
        """Add Gaussian noise to image"""
        try:
            # Generate Gaussian noise
            noise = np.random.normal(mean, sigma, self.image_rgb.shape)
            
            # Add noise to image
            noisy_image = self.image_rgb.astype(np.float64) + noise
            
            # Clip values to valid range
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            # Calculate noise statistics
            noise_energy = np.mean(noise ** 2)
            signal_energy = np.mean(self.image_rgb.astype(np.float64) ** 2)
            snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else float('inf')
            
            return {
                'noisy_image': noisy_image,
                'noise_type': 'gaussian',
                'parameters': {
                    'mean': mean,
                    'sigma': sigma
                },
                'statistics': {
                    'noise_energy': float(noise_energy),
                    'signal_energy': float(signal_energy),
                    'snr_db': float(snr)
                },
                'description': f'Gaussian noise with mean={mean}, sigma={sigma}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to add Gaussian noise: {str(e)}'}
    
    def add_salt_pepper_noise(self, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> Dict[str, Any]:
        """Add salt and pepper noise to image"""
        try:
            noisy_image = self.image_rgb.copy()
            
            # Generate random values for each pixel
            random_vals = np.random.random(self.image_rgb.shape[:2])
            
            # Add salt noise (white pixels)
            salt_mask = random_vals < salt_prob
            noisy_image[salt_mask] = 255
            
            # Add pepper noise (black pixels)
            pepper_mask = random_vals > (1 - pepper_prob)
            noisy_image[pepper_mask] = 0
            
            # Calculate statistics
            total_pixels = self.width * self.height
            salt_pixels = np.sum(salt_mask)
            pepper_pixels = np.sum(pepper_mask)
            affected_pixels = salt_pixels + pepper_pixels
            
            return {
                'noisy_image': noisy_image,
                'noise_type': 'salt_and_pepper',
                'parameters': {
                    'salt_probability': salt_prob,
                    'pepper_probability': pepper_prob
                },
                'statistics': {
                    'total_pixels': total_pixels,
                    'salt_pixels': int(salt_pixels),
                    'pepper_pixels': int(pepper_pixels),
                    'affected_pixels': int(affected_pixels),
                    'noise_percentage': float((affected_pixels / total_pixels) * 100)
                },
                'description': f'Salt & pepper noise: {salt_prob*100}% salt, {pepper_prob*100}% pepper',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to add salt and pepper noise: {str(e)}'}
    
    def add_uniform_noise(self, low: float = -30, high: float = 30) -> Dict[str, Any]:
        """Add uniform noise to image"""
        try:
            # Generate uniform noise
            noise = np.random.uniform(low, high, self.image_rgb.shape)
            
            # Add noise to image
            noisy_image = self.image_rgb.astype(np.float64) + noise
            
            # Clip values to valid range
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            # Calculate statistics
            noise_range = high - low
            noise_energy = np.mean(noise ** 2)
            
            return {
                'noisy_image': noisy_image,
                'noise_type': 'uniform',
                'parameters': {
                    'low': low,
                    'high': high,
                    'range': noise_range
                },
                'statistics': {
                    'noise_energy': float(noise_energy),
                    'noise_range': float(noise_range)
                },
                'description': f'Uniform noise in range [{low}, {high}]',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to add uniform noise: {str(e)}'}
    
    def add_speckle_noise(self, intensity: float = 0.1) -> Dict[str, Any]:
        """Add speckle (multiplicative) noise to image"""
        try:
            # Generate speckle noise (multiplicative)
            noise = np.random.randn(*self.image_rgb.shape) * intensity
            
            # Apply speckle noise: I_noisy = I + I * noise
            noisy_image = self.image_rgb.astype(np.float64)
            noisy_image = noisy_image + (noisy_image * noise)
            
            # Clip values to valid range
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            return {
                'noisy_image': noisy_image,
                'noise_type': 'speckle',
                'parameters': {
                    'intensity': intensity
                },
                'description': f'Speckle noise with intensity={intensity}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to add speckle noise: {str(e)}'}
    
    def apply_gaussian_blur(self, kernel_size: int = 5, sigma_x: float = 0, sigma_y: float = 0) -> Dict[str, Any]:
        """Apply Gaussian blur filter"""
        try:
            # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply Gaussian blur
            if sigma_x == 0:
                sigma_x = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            if sigma_y == 0:
                sigma_y = sigma_x
            
            blurred = cv2.GaussianBlur(self.image_rgb, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
            
            # Calculate blur metrics
            laplacian_original = cv2.Laplacian(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            laplacian_blurred = cv2.Laplacian(cv2.cvtColor(cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            
            sharpness_original = laplacian_original.var()
            sharpness_blurred = laplacian_blurred.var()
            
            return {
                'filtered_image': blurred,
                'filter_type': 'gaussian_blur',
                'parameters': {
                    'kernel_size': kernel_size,
                    'sigma_x': sigma_x,
                    'sigma_y': sigma_y
                },
                'metrics': {
                    'original_sharpness': float(sharpness_original),
                    'filtered_sharpness': float(sharpness_blurred),
                    'sharpness_reduction': float((sharpness_original - sharpness_blurred) / sharpness_original * 100)
                },
                'description': f'Gaussian blur with kernel size {kernel_size}x{kernel_size}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Gaussian blur: {str(e)}'}
    
    def apply_median_filter(self, kernel_size: int = 5) -> Dict[str, Any]:
        """Apply median filter (good for salt & pepper noise)"""
        try:
            # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Apply median filter to each channel
            if len(self.image_rgb.shape) == 3:
                filtered = np.zeros_like(self.image_rgb)
                for i in range(3):
                    filtered[:, :, i] = cv2.medianBlur(self.image_rgb[:, :, i], kernel_size)
            else:
                filtered = cv2.medianBlur(self.image_rgb, kernel_size)
            
            # Calculate noise reduction metric
            diff_original = np.abs(self.image_rgb.astype(np.float64) - np.mean(self.image_rgb))
            diff_filtered = np.abs(filtered.astype(np.float64) - np.mean(filtered))
            
            noise_reduction = (np.mean(diff_original) - np.mean(diff_filtered)) / np.mean(diff_original) * 100
            
            return {
                'filtered_image': filtered,
                'filter_type': 'median',
                'parameters': {
                    'kernel_size': kernel_size
                },
                'metrics': {
                    'noise_reduction_percentage': float(noise_reduction)
                },
                'description': f'Median filter with {kernel_size}x{kernel_size} kernel',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply median filter: {str(e)}'}
    
    def apply_bilateral_filter(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> Dict[str, Any]:
        """Apply bilateral filter (edge-preserving smoothing)"""
        try:
            # Convert to BGR for OpenCV bilateral filter
            image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
            
            # Apply bilateral filter
            filtered_bgr = cv2.bilateralFilter(image_bgr, d, sigma_color, sigma_space)
            
            # Convert back to RGB
            filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
            
            # Calculate edge preservation metric
            original_edges = cv2.Canny(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 50, 150)
            filtered_edges = cv2.Canny(cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
            
            edge_preservation = np.sum(filtered_edges) / np.sum(original_edges) * 100 if np.sum(original_edges) > 0 else 0
            
            return {
                'filtered_image': filtered,
                'filter_type': 'bilateral',
                'parameters': {
                    'd': d,
                    'sigma_color': sigma_color,
                    'sigma_space': sigma_space
                },
                'metrics': {
                    'edge_preservation_percentage': float(edge_preservation)
                },
                'description': f'Bilateral filter - edge-preserving smoothing',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply bilateral filter: {str(e)}'}
    
    def apply_box_filter(self, kernel_size: int = 5) -> Dict[str, Any]:
        """Apply box (averaging) filter"""
        try:
            # Ensure odd kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create box filter kernel
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Apply filter
            filtered = cv2.filter2D(self.image_rgb, -1, kernel)
            
            return {
                'filtered_image': filtered,
                'filter_type': 'box',
                'parameters': {
                    'kernel_size': kernel_size
                },
                'description': f'Box filter with {kernel_size}x{kernel_size} kernel',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply box filter: {str(e)}'}
    
    def apply_sharpen_filter(self, intensity: float = 1.0) -> Dict[str, Any]:
        """Apply sharpening filter"""
        try:
            # Define sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * intensity
            
            # Apply sharpening filter
            sharpened = cv2.filter2D(self.image_rgb, -1, kernel)
            
            # Clip values to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Calculate sharpness increase
            laplacian_original = cv2.Laplacian(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            laplacian_sharpened = cv2.Laplacian(cv2.cvtColor(cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            
            sharpness_original = laplacian_original.var()
            sharpness_enhanced = laplacian_sharpened.var()
            sharpness_increase = (sharpness_enhanced - sharpness_original) / sharpness_original * 100 if sharpness_original > 0 else 0
            
            return {
                'filtered_image': sharpened,
                'filter_type': 'sharpen',
                'parameters': {
                    'intensity': intensity
                },
                'metrics': {
                    'original_sharpness': float(sharpness_original),
                    'enhanced_sharpness': float(sharpness_enhanced),
                    'sharpness_increase_percentage': float(sharpness_increase)
                },
                'description': f'Sharpening filter with intensity {intensity}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply sharpen filter: {str(e)}'}
    
    def apply_unsharp_mask(self, sigma: float = 1.0, strength: float = 1.5, threshold: float = 0) -> Dict[str, Any]:
        """Apply unsharp masking for sharpening"""
        try:
            # Convert to float for calculations
            image_float = self.image_rgb.astype(np.float64)
            
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(self.image_rgb, (0, 0), sigma)
            blurred_float = blurred.astype(np.float64)
            
            # Create mask (difference between original and blurred)
            mask = image_float - blurred_float
            
            # Apply threshold to mask
            if threshold > 0:
                mask = np.where(np.abs(mask) < threshold, 0, mask)
            
            # Apply unsharp mask
            sharpened = image_float + strength * mask
            
            # Clip to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return {
                'filtered_image': sharpened,
                'unsharp_mask': ((mask + 128).clip(0, 255)).astype(np.uint8),  # Visualize mask
                'filter_type': 'unsharp_mask',
                'parameters': {
                    'sigma': sigma,
                    'strength': strength,
                    'threshold': threshold
                },
                'description': f'Unsharp masking: sigma={sigma}, strength={strength}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply unsharp mask: {str(e)}'}
    
    def apply_non_local_means(self, h: float = 10, template_window_size: int = 7, search_window_size: int = 21) -> Dict[str, Any]:
        """Apply Non-Local Means denoising"""
        try:
            # Convert to BGR for OpenCV
            image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
            
            # Apply Non-Local Means denoising
            denoised_bgr = cv2.fastNlMeansDenoisingColored(image_bgr, None, h, h, template_window_size, search_window_size)
            
            # Convert back to RGB
            denoised = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            
            # Calculate noise reduction
            mse_original = np.mean((self.image_rgb - np.mean(self.image_rgb)) ** 2)
            mse_denoised = np.mean((denoised - np.mean(denoised)) ** 2)
            noise_reduction = (mse_original - mse_denoised) / mse_original * 100 if mse_original > 0 else 0
            
            return {
                'filtered_image': denoised,
                'filter_type': 'non_local_means',
                'parameters': {
                    'h': h,
                    'template_window_size': template_window_size,
                    'search_window_size': search_window_size
                },
                'metrics': {
                    'noise_reduction_percentage': float(noise_reduction)
                },
                'description': 'Non-Local Means denoising - advanced noise removal',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Non-Local Means: {str(e)}'}
    
    def apply_wiener_filter(self, noise_variance: float = 100) -> Dict[str, Any]:
        """Apply Wiener filter for noise reduction"""
        try:
            # Convert to frequency domain
            image_float = self.image_rgb.astype(np.float64)
            
            # Apply Wiener filter to each channel
            filtered_channels = []
            
            for i in range(3):
                channel = image_float[:, :, i]
                
                # FFT
                f_transform = np.fft.fft2(channel)
                f_shift = np.fft.fftshift(f_transform)
                
                # Estimate signal power spectrum
                signal_power = np.abs(f_shift) ** 2
                
                # Wiener filter
                wiener_filter = signal_power / (signal_power + noise_variance)
                
                # Apply filter
                filtered_f = f_shift * wiener_filter
                
                # IFFT
                f_ishift = np.fft.ifftshift(filtered_f)
                filtered_channel = np.fft.ifft2(f_ishift)
                filtered_channel = np.real(filtered_channel)
                
                filtered_channels.append(filtered_channel)
            
            # Combine channels
            filtered = np.stack(filtered_channels, axis=2)
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            
            return {
                'filtered_image': filtered,
                'filter_type': 'wiener',
                'parameters': {
                    'noise_variance': noise_variance
                },
                'description': f'Wiener filter with noise variance {noise_variance}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply Wiener filter: {str(e)}'}