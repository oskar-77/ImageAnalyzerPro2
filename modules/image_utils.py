import cv2
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import logging

class ImageUtils:
    """Utility functions for image processing and validation"""
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate if file is a valid image"""
        try:
            if not os.path.exists(image_path):
                return False
            
            # Try to load with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Check if image has valid dimensions
            if len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
                return False
            
            return True
        
        except Exception as e:
            logging.error(f"Image validation error: {str(e)}")
            return False
    
    @staticmethod
    def get_image_format(image_path: str) -> str:
        """Get image format from file path"""
        try:
            return image_path.split('.')[-1].lower()
        except:
            return 'unknown'
    
    @staticmethod
    def get_file_size(image_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(image_path)
        except:
            return 0
    
    @staticmethod
    def get_image_dimensions(image_path: str) -> Tuple[int, int, int]:
        """Get image dimensions (width, height, channels)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return (0, 0, 0)
            
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            return (width, height, channels)
        
        except Exception as e:
            logging.error(f"Error getting image dimensions: {str(e)}")
            return (0, 0, 0)
    
    @staticmethod
    def convert_image_format(input_path: str, output_path: str, target_format: str) -> bool:
        """Convert image to different format"""
        try:
            # Load image with PIL for better format support
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if saving as JPEG
                if target_format.lower() in ['jpg', 'jpeg'] and img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img.save(output_path, format=target_format.upper())
                return True
        
        except Exception as e:
            logging.error(f"Error converting image format: {str(e)}")
            return False
    
    @staticmethod
    def create_thumbnail(input_path: str, output_path: str, size: Tuple[int, int] = (150, 150)) -> bool:
        """Create thumbnail of image"""
        try:
            with Image.open(input_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(output_path)
                return True
        
        except Exception as e:
            logging.error(f"Error creating thumbnail: {str(e)}")
            return False
    
    @staticmethod
    def get_image_metadata(image_path: str) -> Dict[str, Any]:
        """Get comprehensive image metadata"""
        try:
            metadata = {}
            
            # Basic file info
            metadata['file_path'] = image_path
            metadata['file_name'] = os.path.basename(image_path)
            metadata['file_size'] = ImageUtils.get_file_size(image_path)
            metadata['format'] = ImageUtils.get_image_format(image_path)
            
            # Image dimensions
            width, height, channels = ImageUtils.get_image_dimensions(image_path)
            metadata['width'] = width
            metadata['height'] = height
            metadata['channels'] = channels
            metadata['total_pixels'] = width * height
            
            # Color mode
            if channels == 1:
                metadata['color_mode'] = 'Grayscale'
            elif channels == 3:
                metadata['color_mode'] = 'RGB'
            elif channels == 4:
                metadata['color_mode'] = 'RGBA'
            else:
                metadata['color_mode'] = f'{channels}-channel'
            
            # Try to get EXIF data with PIL
            try:
                with Image.open(image_path) as img:
                    exif = img._getexif()
                    if exif:
                        metadata['exif'] = exif
            except:
                pass
            
            return metadata
        
        except Exception as e:
            logging.error(f"Error getting image metadata: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def crop_image(input_path: str, output_path: str, x: int, y: int, width: int, height: int) -> bool:
        """Crop image to specified region"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                return False
            
            # Validate crop coordinates
            img_height, img_width = image.shape[:2]
            
            if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
                return False
            
            # Crop image
            cropped = image[y:y+height, x:x+width]
            
            return cv2.imwrite(output_path, cropped)
        
        except Exception as e:
            logging.error(f"Error cropping image: {str(e)}")
            return False
    
    @staticmethod
    def rotate_image(input_path: str, output_path: str, angle: float) -> bool:
        """Rotate image by specified angle"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                return False
            
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            
            return cv2.imwrite(output_path, rotated)
        
        except Exception as e:
            logging.error(f"Error rotating image: {str(e)}")
            return False
    
    @staticmethod
    def flip_image(input_path: str, output_path: str, direction: str = 'horizontal') -> bool:
        """Flip image horizontally or vertically"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                return False
            
            if direction == 'horizontal':
                flipped = cv2.flip(image, 1)
            elif direction == 'vertical':
                flipped = cv2.flip(image, 0)
            elif direction == 'both':
                flipped = cv2.flip(image, -1)
            else:
                return False
            
            return cv2.imwrite(output_path, flipped)
        
        except Exception as e:
            logging.error(f"Error flipping image: {str(e)}")
            return False
    
    @staticmethod
    def calculate_image_hash(image_path: str, hash_type: str = 'average') -> str:
        """Calculate perceptual hash of image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if hash_type == 'average':
                # Average hash
                resized = cv2.resize(gray, (8, 8))
                avg = resized.mean()
                hash_bits = resized > avg
                return ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
            
            elif hash_type == 'difference':
                # Difference hash
                resized = cv2.resize(gray, (9, 8))
                diff = resized[:, 1:] > resized[:, :-1]
                return ''.join(['1' if bit else '0' for bit in diff.flatten()])
            
            else:
                return ""
        
        except Exception as e:
            logging.error(f"Error calculating image hash: {str(e)}")
            return ""
    
    @staticmethod
    def clean_temp_files(directory: str, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified hours"""
        try:
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(filepath)
                            cleaned_count += 1
                        except:
                            pass
            
            return cleaned_count
        
        except Exception as e:
            logging.error(f"Error cleaning temp files: {str(e)}")
            return 0
    
    @staticmethod
    def create_image_grid(image_paths: List[str], output_path: str, grid_size: Tuple[int, int] = None) -> bool:
        """Create a grid of images"""
        try:
            if not image_paths:
                return False
            
            # Load all images
            images = []
            for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
            
            if not images:
                return False
            
            # Determine grid size if not specified
            if grid_size is None:
                n_images = len(images)
                cols = int(np.ceil(np.sqrt(n_images)))
                rows = int(np.ceil(n_images / cols))
                grid_size = (rows, cols)
            
            rows, cols = grid_size
            
            # Find the maximum dimensions
            max_height = max(img.shape[0] for img in images)
            max_width = max(img.shape[1] for img in images)
            
            # Create grid
            grid_height = rows * max_height
            grid_width = cols * max_width
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Place images in grid
            for i, img in enumerate(images):
                if i >= rows * cols:
                    break
                
                row = i // cols
                col = i % cols
                
                # Resize image to fit grid cell
                resized = cv2.resize(img, (max_width, max_height))
                
                # Place in grid
                y_start = row * max_height
                y_end = y_start + max_height
                x_start = col * max_width
                x_end = x_start + max_width
                
                grid[y_start:y_end, x_start:x_end] = resized
            
            return cv2.imwrite(output_path, grid)
        
        except Exception as e:
            logging.error(f"Error creating image grid: {str(e)}")
            return False
