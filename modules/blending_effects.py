import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class BlendingEffects:
    """Advanced image blending and visual effects module"""
    
    def __init__(self, image_path: str, image2_path: Optional[str] = None):
        """Initialize with image path(s)"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        self.height, self.width, self.channels = self.image.shape
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Load second image if provided
        self.image2 = None
        self.image2_rgb = None
        if image2_path:
            self.image2 = cv2.imread(image2_path)
            if self.image2 is not None:
                # Resize second image to match first image
                self.image2 = cv2.resize(self.image2, (self.width, self.height))
                self.image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
    
    def weighted_blend(self, alpha: float = 0.5, beta: Optional[float] = None, gamma: float = 0.0) -> Dict[str, Any]:
        """Weighted blending of two images"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for blending'}
            
            if beta is None:
                beta = 1.0 - alpha
            
            # Perform weighted blending
            blended = cv2.addWeighted(self.image_rgb, alpha, self.image2_rgb, beta, gamma)
            
            # Ensure values are in valid range
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            return {
                'blended_image': blended,
                'blend_parameters': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma
                },
                'blend_type': 'weighted',
                'formula': f'{alpha} * img1 + {beta} * img2 + {gamma}',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to perform weighted blend: {str(e)}'}
    
    def apply_negative(self) -> Dict[str, Any]:
        """Apply negative (color inversion) effect"""
        try:
            negative = 255 - self.image_rgb
            
            return {
                'negative_image': negative,
                'effect_type': 'negative',
                'description': 'Color inversion - each pixel value subtracted from 255',
                'formula': '255 - pixel_value',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply negative effect: {str(e)}'}
    
    def multiply_blend(self, factor: float = 1.0, use_second_image: bool = False) -> Dict[str, Any]:
        """Multiply blending mode"""
        try:
            if use_second_image:
                if self.image2 is None:
                    return {'error': 'Second image required for multiply blending'}
                
                # Normalize to 0-1 range for multiplication
                img1_norm = self.image_rgb.astype(np.float32) / 255.0
                img2_norm = self.image2_rgb.astype(np.float32) / 255.0
                
                # Multiply blend
                result = img1_norm * img2_norm
                
                # Convert back to 0-255 range
                result = (result * 255).astype(np.uint8)
                
                description = 'Multiply blend between two images'
                formula = '(img1 * img2) / 255'
            else:
                # Multiply with factor
                result = self.image_rgb.astype(np.float32) * factor
                result = np.clip(result, 0, 255).astype(np.uint8)
                
                description = f'Multiply image by factor {factor}'
                formula = f'pixel_value * {factor}'
            
            return {
                'multiplied_image': result,
                'effect_type': 'multiply',
                'description': description,
                'formula': formula,
                'factor': factor if not use_second_image else None,
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply multiply blend: {str(e)}'}
    
    def selective_coloring(self, target_color: str, replacement_color: Tuple[int, int, int], 
                         tolerance: int = 30) -> Dict[str, Any]:
        """Selective color replacement"""
        try:
            # Convert to HSV for better color selection
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common colors
            color_ranges = {
                'red': ([0, 50, 50], [10, 255, 255]),
                'green': ([50, 50, 50], [70, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'yellow': ([20, 50, 50], [30, 255, 255]),
                'orange': ([10, 50, 50], [20, 255, 255]),
                'purple': ([130, 50, 50], [160, 255, 255]),
                'cyan': ([80, 50, 50], [100, 255, 255])
            }
            
            if target_color.lower() in color_ranges:
                lower_bound, upper_bound = color_ranges[target_color.lower()]
                lower_bound = np.array(lower_bound)
                upper_bound = np.array(upper_bound)
                
                # Adjust bounds by tolerance
                lower_bound[1] = max(0, lower_bound[1] - tolerance)
                lower_bound[2] = max(0, lower_bound[2] - tolerance)
                upper_bound[1] = min(255, upper_bound[1] + tolerance)
                upper_bound[2] = min(255, upper_bound[2] + tolerance)
            else:
                return {'error': f'Unsupported target color: {target_color}'}
            
            # Create mask for target color
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Apply selective coloring
            result = self.image_rgb.copy()
            replacement_bgr = replacement_color[::-1]  # RGB to BGR
            result[mask > 0] = replacement_color
            
            # Calculate statistics
            total_pixels = self.width * self.height
            affected_pixels = np.sum(mask > 0)
            coverage_percentage = (affected_pixels / total_pixels) * 100
            
            return {
                'colored_image': result,
                'color_mask': mask,
                'effect_type': 'selective_coloring',
                'target_color': target_color,
                'replacement_color': replacement_color,
                'tolerance': tolerance,
                'statistics': {
                    'total_pixels': total_pixels,
                    'affected_pixels': int(affected_pixels),
                    'coverage_percentage': float(coverage_percentage)
                },
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply selective coloring: {str(e)}'}
    
    def screen_blend(self) -> Dict[str, Any]:
        """Screen blending mode with second image"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for screen blending'}
            
            # Normalize to 0-1 range
            img1_norm = self.image_rgb.astype(np.float32) / 255.0
            img2_norm = self.image2_rgb.astype(np.float32) / 255.0
            
            # Screen blend formula: 1 - (1 - img1) * (1 - img2)
            result = 1.0 - (1.0 - img1_norm) * (1.0 - img2_norm)
            
            # Convert back to 0-255 range
            result = (result * 255).astype(np.uint8)
            
            return {
                'blended_image': result,
                'effect_type': 'screen',
                'description': 'Screen blend - creates lighter result than multiply',
                'formula': '1 - (1 - img1) * (1 - img2)',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply screen blend: {str(e)}'}
    
    def overlay_blend(self) -> Dict[str, Any]:
        """Overlay blending mode"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for overlay blending'}
            
            # Normalize to 0-1 range
            img1_norm = self.image_rgb.astype(np.float32) / 255.0
            img2_norm = self.image2_rgb.astype(np.float32) / 255.0
            
            # Overlay blend formula
            result = np.where(img1_norm < 0.5,
                            2 * img1_norm * img2_norm,
                            1 - 2 * (1 - img1_norm) * (1 - img2_norm))
            
            # Convert back to 0-255 range
            result = (result * 255).astype(np.uint8)
            
            return {
                'blended_image': result,
                'effect_type': 'overlay',
                'description': 'Overlay blend - combines multiply and screen modes',
                'formula': 'if base < 0.5: 2*base*overlay else 1-2*(1-base)*(1-overlay)',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply overlay blend: {str(e)}'}
    
    def soft_light_blend(self) -> Dict[str, Any]:
        """Soft light blending mode"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for soft light blending'}
            
            # Normalize to 0-1 range
            img1_norm = self.image_rgb.astype(np.float32) / 255.0
            img2_norm = self.image2_rgb.astype(np.float32) / 255.0
            
            # Soft light blend formula
            result = np.where(img2_norm < 0.5,
                            2 * img1_norm * img2_norm + img1_norm * img1_norm * (1 - 2 * img2_norm),
                            2 * img1_norm * (1 - img2_norm) + np.sqrt(img1_norm) * (2 * img2_norm - 1))
            
            # Convert back to 0-255 range
            result = (result * 255).astype(np.uint8)
            
            return {
                'blended_image': result,
                'effect_type': 'soft_light',
                'description': 'Soft light blend - creates subtle lighting effects',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply soft light blend: {str(e)}'}
    
    def color_dodge_blend(self) -> Dict[str, Any]:
        """Color dodge blending mode"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for color dodge blending'}
            
            # Normalize to 0-1 range
            img1_norm = self.image_rgb.astype(np.float32) / 255.0
            img2_norm = self.image2_rgb.astype(np.float32) / 255.0
            
            # Color dodge formula: base / (1 - blend)
            # Avoid division by zero
            denominator = 1.0 - img2_norm
            denominator = np.where(denominator == 0, 0.001, denominator)
            
            result = img1_norm / denominator
            result = np.clip(result, 0, 1)
            
            # Convert back to 0-255 range
            result = (result * 255).astype(np.uint8)
            
            return {
                'blended_image': result,
                'effect_type': 'color_dodge',
                'description': 'Color dodge blend - creates bright, high-contrast effects',
                'formula': 'base / (1 - blend)',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply color dodge blend: {str(e)}'}
    
    def difference_blend(self) -> Dict[str, Any]:
        """Difference blending mode"""
        try:
            if self.image2 is None:
                return {'error': 'Second image required for difference blending'}
            
            # Difference blend: absolute difference
            result = cv2.absdiff(self.image_rgb, self.image2_rgb)
            
            return {
                'blended_image': result,
                'effect_type': 'difference',
                'description': 'Difference blend - absolute difference between images',
                'formula': '|img1 - img2|',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to apply difference blend: {str(e)}'}
    
    def create_vignette(self, intensity: float = 0.8, radius: float = 0.7) -> Dict[str, Any]:
        """Create vignette effect"""
        try:
            # Create coordinate matrices
            center_x, center_y = self.width // 2, self.height // 2
            y, x = np.ogrid[:self.height, :self.width]
            
            # Calculate distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Normalize distance
            max_distance = np.sqrt(center_x**2 + center_y**2)
            distance = distance / max_distance
            
            # Create vignette mask
            vignette = np.clip(1 - (distance / radius) * intensity, 0, 1)
            
            # Apply vignette to each channel
            result = self.image_rgb.copy().astype(np.float32)
            for i in range(3):
                result[:, :, i] *= vignette
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return {
                'vignette_image': result,
                'vignette_mask': (vignette * 255).astype(np.uint8),
                'effect_type': 'vignette',
                'parameters': {
                    'intensity': intensity,
                    'radius': radius
                },
                'description': 'Vignette effect - darkens edges of image',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to create vignette effect: {str(e)}'}
    
    def create_gradient_overlay(self, direction: str = 'vertical', 
                              start_color: Tuple[int, int, int] = (0, 0, 0),
                              end_color: Tuple[int, int, int] = (255, 255, 255),
                              blend_mode: str = 'overlay') -> Dict[str, Any]:
        """Create gradient overlay effect"""
        try:
            # Create gradient
            if direction.lower() == 'vertical':
                gradient = np.linspace(0, 1, self.height).reshape(-1, 1)
                gradient = np.repeat(gradient, self.width, axis=1)
            elif direction.lower() == 'horizontal':
                gradient = np.linspace(0, 1, self.width).reshape(1, -1)
                gradient = np.repeat(gradient, self.height, axis=0)
            elif direction.lower() == 'radial':
                center_x, center_y = self.width // 2, self.height // 2
                y, x = np.ogrid[:self.height, :self.width]
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                gradient = distance / max_distance
                gradient = np.clip(gradient, 0, 1)
            else:
                return {'error': f'Unsupported gradient direction: {direction}'}
            
            # Create gradient image
            gradient_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(3):
                gradient_image[:, :, i] = (start_color[i] * (1 - gradient) + 
                                         end_color[i] * gradient).astype(np.uint8)
            
            # Apply blend mode
            if blend_mode.lower() == 'overlay':
                # Normalize for overlay blend
                img_norm = self.image_rgb.astype(np.float32) / 255.0
                grad_norm = gradient_image.astype(np.float32) / 255.0
                
                result = np.where(img_norm < 0.5,
                                2 * img_norm * grad_norm,
                                1 - 2 * (1 - img_norm) * (1 - grad_norm))
                result = (result * 255).astype(np.uint8)
            
            elif blend_mode.lower() == 'multiply':
                result = ((self.image_rgb.astype(np.float32) * gradient_image.astype(np.float32)) / 255).astype(np.uint8)
            
            elif blend_mode.lower() == 'screen':
                img_norm = self.image_rgb.astype(np.float32) / 255.0
                grad_norm = gradient_image.astype(np.float32) / 255.0
                result = 1.0 - (1.0 - img_norm) * (1.0 - grad_norm)
                result = (result * 255).astype(np.uint8)
            
            else:
                # Default to normal blend (weighted)
                result = cv2.addWeighted(self.image_rgb, 0.7, gradient_image, 0.3, 0)
            
            return {
                'gradient_overlay_image': result,
                'gradient_image': gradient_image,
                'effect_type': 'gradient_overlay',
                'parameters': {
                    'direction': direction,
                    'start_color': start_color,
                    'end_color': end_color,
                    'blend_mode': blend_mode
                },
                'description': f'{direction} gradient overlay with {blend_mode} blending',
                'success': True
            }
        
        except Exception as e:
            return {'error': f'Failed to create gradient overlay: {str(e)}'}