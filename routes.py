import os
import json
import uuid
from datetime import datetime
from flask import render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils

from app import app, db
from models import ImageUpload, ImageAnalysis, ImageComparison as ComparisonModel, PixelExploration, ImageProcessingLog
from modules.image_exploration import ImageExploration
from modules.image_statistics import ImageStatistics
from modules.image_conversion import ImageConversion
from modules.image_comparison import ImageComparison
from modules.image_utils import ImageUtils
from modules.digital_images import DigitalImages
from modules.color_spaces import ColorSpaces
from modules.masks import Masks
from modules.blending_effects import BlendingEffects
from modules.noise_smoothing import NoiseSmoothing
from modules.edge_detection import EdgeDetection
from modules.histogram_enhancement import HistogramEnhancement
from modules.thresholding import Thresholding

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_numpy_to_serializable(obj):
    """Recursively convert numpy arrays and all numpy types to JSON-serializable formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer) or type(obj).__name__ in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        return int(obj)
    elif isinstance(obj, np.floating) or type(obj).__name__ in ['float16', 'float32', 'float64']:
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif type(obj).__name__ in ['complex64', 'complex128']:
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif hasattr(obj, 'item'):  # Any remaining numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Any remaining numpy arrays
        return obj.tolist()
    else:
        return obj


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))

        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename or '')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Validate image
            if not ImageUtils.validate_image(filepath):
                os.remove(filepath)
                flash('Invalid image file', 'error')
                return redirect(url_for('index'))

            return redirect(url_for('analysis', filename=filename))
        else:
            flash(
                'File type not allowed. Please upload PNG, JPG, JPEG, GIF, BMP, TIFF, or WEBP files.',
                'error')
            return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        flash('Upload failed. Please try again.', 'error')
        return redirect(url_for('index'))


@app.route('/analysis/<filename>')
def analysis(filename):
    """Analysis page for uploaded image"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        flash('Image not found', 'error')
        return redirect(url_for('index'))

    return render_template('analysis.html', filename=filename)


@app.route('/api/image_info/<filename>')
def image_info(filename):
    """Get basic image information"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Get image info
        height, width, channels = image.shape
        file_size = os.path.getsize(filepath)

        return jsonify({
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'file_size': int(file_size),
            'format': filename.split('.')[-1].upper()
        })

    except Exception as e:
        app.logger.error(f"Image info error: {str(e)}")
        return jsonify({'error': 'Failed to get image information'}), 500


@app.route('/api/pixel_value/<filename>')
def pixel_value(filename):
    """Get pixel value at specific coordinates"""
    try:
        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        explorer = ImageExploration(filepath)
        pixel_info = explorer.get_pixel_value(x, y)
        
        # Convert numpy arrays and other types to serializable format
        pixel_info = convert_numpy_to_serializable(pixel_info)

        return jsonify(pixel_info)

    except Exception as e:
        app.logger.error(f"Pixel value error: {str(e)}")
        return jsonify({'error': 'Failed to get pixel value'}), 500


@app.route('/api/statistics/<filename>')
def statistics(filename):
    """Get image statistics"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        stats = ImageStatistics(filepath)
        statistics_data = stats.get_comprehensive_statistics()
        
        # Convert numpy arrays and other types to serializable format
        statistics_data = convert_numpy_to_serializable(statistics_data)

        return jsonify(statistics_data)

    except Exception as e:
        app.logger.error(f"Statistics error: {str(e)}")
        return jsonify({'error': 'Failed to calculate statistics'}), 500


@app.route('/api/histogram/<filename>')
def histogram(filename):
    """Get histogram data for plotting"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        stats = ImageStatistics(filepath)
        histogram_data = stats.get_histogram_data()
        
        # Convert numpy arrays and other types to serializable format
        histogram_data = convert_numpy_to_serializable(histogram_data)

        return jsonify(histogram_data)

    except Exception as e:
        app.logger.error(f"Histogram error: {str(e)}")
        return jsonify({'error': 'Failed to generate histogram'}), 500


@app.route('/api/convert/<filename>')
def convert_image(filename):
    """Convert image to different formats"""
    try:
        conversion_type = request.args.get('type', 'grayscale')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        converter = ImageConversion(filepath)

        # Generate output filename
        base_name = filename.rsplit('.', 1)[0]
        output_filename = f"{base_name}_{conversion_type}_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                   output_filename)

        # Perform conversion
        success = False
        if conversion_type == 'grayscale':
            success = converter.to_grayscale(output_path)
        elif conversion_type == 'binary':
            success = converter.to_binary(output_path)
        elif conversion_type == 'hsv':
            success = converter.to_hsv(output_path)

        if success:
            return jsonify({'converted_filename': output_filename})
        else:
            return jsonify({'error': 'Conversion failed'}), 500

    except Exception as e:
        app.logger.error(f"Conversion error: {str(e)}")
        return jsonify({'error': 'Failed to convert image'}), 500


@app.route('/api/compare', methods=['POST'])
def compare_images():
    """Compare two uploaded images"""
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Two files required for comparison'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400

        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({'error': 'Invalid file types'}), 400

        # Save files
        filename1 = str(uuid.uuid4()) + '_' + secure_filename(file1.filename or '')
        filename2 = str(uuid.uuid4()) + '_' + secure_filename(file2.filename or '')

        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(filepath1)
        file2.save(filepath2)

        # Validate images
        if not (ImageUtils.validate_image(filepath1)
                and ImageUtils.validate_image(filepath2)):
            os.remove(filepath1)
            os.remove(filepath2)
            return jsonify({'error': 'Invalid image files'}), 400

        # Compare images
        comparator = ImageComparison(filepath1, filepath2)
        comparison_data = comparator.compare_images()

        # Generate difference map
        diff_filename = f"diff_{uuid.uuid4().hex[:8]}.png"
        diff_path = os.path.join(app.config['UPLOAD_FOLDER'], diff_filename)

        if comparator.generate_difference_map(diff_path):
            comparison_data['difference_map'] = diff_filename

        comparison_data['image1'] = filename1
        comparison_data['image2'] = filename2
        
        # Convert numpy arrays and other types to serializable format
        comparison_data = convert_numpy_to_serializable(comparison_data)

        return jsonify(comparison_data)

    except Exception as e:
        app.logger.error(f"Comparison error: {str(e)}")
        return jsonify({'error': 'Failed to compare images'}), 500


@app.route('/generate_report/<filename>')
def generate_report(filename):
    """Generate and download analysis report"""
    try:
        report_format = request.args.get('format', 'txt')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        # Generate comprehensive analysis
        explorer = ImageExploration(filepath)
        stats = ImageStatistics(filepath)

        # Get image info
        image = cv2.imread(filepath)
        height, width, channels = image.shape
        file_size = os.path.getsize(filepath)

        # Get statistics
        statistics_data = stats.get_comprehensive_statistics()

        # Generate report
        report_filename = f"report_{filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex[:8]}.{report_format}"
        report_path = os.path.join(app.config['REPORT_FOLDER'],
                                   report_filename)

        if report_format == 'txt':
            with open(report_path, 'w') as f:
                f.write("IMAGE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Image: {filename}\n\n")

                f.write("BASIC INFORMATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"Dimensions: {width} x {height} pixels\n")
                f.write(f"Channels: {channels}\n")
                f.write(f"File Size: {file_size:,} bytes\n\n")

                f.write("STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(
                    f"Mean Brightness: {statistics_data['brightness']['mean']:.2f}\n"
                )
                f.write(
                    f"Standard Deviation: {statistics_data['brightness']['std']:.2f}\n"
                )
                f.write(f"Min Value: {statistics_data['brightness']['min']}\n")
                f.write(f"Max Value: {statistics_data['brightness']['max']}\n")

                if 'channel_stats' in statistics_data:
                    f.write("\nCHANNEL STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    for channel, data in statistics_data[
                            'channel_stats'].items():
                        f.write(
                            f"{channel}: Mean={data['mean']:.2f}, Std={data['std']:.2f}\n"
                        )

        elif report_format == 'png':
            # Create a visual report
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,
                                                         2,
                                                         figsize=(12, 10))

            # Original image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax1.imshow(image_rgb)
            ax1.set_title(f'Original Image\n{width}x{height} pixels')
            ax1.axis('off')

            # Histogram
            if len(image.shape) == 3:
                colors = ['blue', 'green', 'red']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    ax2.plot(hist, color=color, alpha=0.7)
            else:
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                ax2.plot(hist, color='gray')

            ax2.set_title('Histogram')
            ax2.set_xlabel('Pixel Value')
            ax2.set_ylabel('Frequency')

            # Statistics text
            ax3.text(
                0.1,
                0.9,
                f"Mean Brightness: {statistics_data['brightness']['mean']:.2f}",
                transform=ax3.transAxes,
                fontsize=12)
            ax3.text(
                0.1,
                0.8,
                f"Std Deviation: {statistics_data['brightness']['std']:.2f}",
                transform=ax3.transAxes,
                fontsize=12)
            ax3.text(0.1,
                     0.7,
                     f"Min Value: {statistics_data['brightness']['min']}",
                     transform=ax3.transAxes,
                     fontsize=12)
            ax3.text(0.1,
                     0.6,
                     f"Max Value: {statistics_data['brightness']['max']}",
                     transform=ax3.transAxes,
                     fontsize=12)
            ax3.set_title('Statistics')
            ax3.axis('off')

            # Grayscale version
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ax4.imshow(gray, cmap='gray')
            ax4.set_title('Grayscale Version')
            ax4.axis('off')

            plt.tight_layout()
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()

        return send_file(report_path,
                         as_attachment=True,
                         download_name=report_filename)

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate report'}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404


@app.route('/api/crop/<filename>', methods=['POST'])
def crop_image(filename):
    """Crop image to specified region"""
    try:
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        data = request.get_json()
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 100))
        height = int(data.get('height', 100))

        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_filename = f"{base_name}_cropped_{uuid.uuid4().hex[:8]}{ext}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                   output_filename)

        # Crop image using ImageUtils
        success = ImageUtils.crop_image(filepath, output_path, x, y, width,
                                        height)

        if success:
            return jsonify({
                'success': True,
                'cropped_filename': output_filename,
                'dimensions': f"{width}x{height}"
            })
        else:
            return jsonify({'error': 'Failed to crop image'}), 500

    except Exception as e:
        app.logger.error(f"Crop error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rotate/<filename>', methods=['POST'])
def rotate_image(filename):
    """Rotate image by specified angle"""
    try:
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        data = request.get_json()
        angle = float(data.get('angle', 0))

        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_filename = f"{base_name}_rotated_{int(angle)}_{uuid.uuid4().hex[:8]}{ext}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                   output_filename)

        # Rotate image using ImageUtils
        success = ImageUtils.rotate_image(filepath, output_path, angle)

        if success:
            return jsonify({
                'success': True,
                'rotated_filename': output_filename,
                'angle': angle
            })
        else:
            return jsonify({'error': 'Failed to rotate image'}), 500

    except Exception as e:
        app.logger.error(f"Rotate error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/flip/<filename>', methods=['POST'])
def flip_image(filename):
    """Flip image horizontally or vertically"""
    try:
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404

        data = request.get_json()
        direction = data.get('direction', 'horizontal')

        if direction not in ['horizontal', 'vertical']:
            return jsonify(
                {'error':
                 'Invalid direction. Use horizontal or vertical'}), 400

        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_filename = f"{base_name}_flipped_{direction}_{uuid.uuid4().hex[:8]}{ext}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                   output_filename)

        # Flip image using ImageUtils
        success = ImageUtils.flip_image(filepath, output_path, direction)

        if success:
            return jsonify({
                'success': True,
                'flipped_filename': output_filename,
                'direction': direction
            })
        else:
            return jsonify({'error': 'Failed to flip image'}), 500

    except Exception as e:
        app.logger.error(f"Flip error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Digital Images Module Routes
@app.route('/api/digital_properties/<filename>')
def digital_properties(filename):
    """Get comprehensive digital properties of image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        digital_images = DigitalImages(filepath)
        properties = digital_images.get_digital_properties()
        
        # Convert numpy arrays and other types to serializable format
        properties = convert_numpy_to_serializable(properties)
        
        return jsonify(properties)
    
    except Exception as e:
        app.logger.error(f"Digital properties error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pixel_matrix/<filename>')
def pixel_matrix(filename):
    """Get pixel matrix around coordinates"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))
        size = int(request.args.get('size', 5))
        
        digital_images = DigitalImages(filepath)
        matrix = digital_images.get_pixel_matrix(x, y, size)
        
        # Convert numpy arrays and other types to serializable format
        matrix = convert_numpy_to_serializable(matrix)
        
        return jsonify(matrix)
    
    except Exception as e:
        app.logger.error(f"Pixel matrix error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/bit_planes/<filename>')
def bit_planes(filename):
    """Analyze bit planes of image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        digital_images = DigitalImages(filepath)
        analysis = digital_images.analyze_bit_planes()
        
        # Convert numpy arrays and other types to serializable format
        analysis = convert_numpy_to_serializable(analysis)
        
        return jsonify(analysis)
    
    except Exception as e:
        app.logger.error(f"Bit planes error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Color Spaces Module Routes
@app.route('/api/color_spaces/<filename>')
def color_spaces_conversion(filename):
    """Convert image to all color spaces"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        color_spaces = ColorSpaces(filepath)
        conversions = color_spaces.convert_all_color_spaces()
        
        # Remove large data arrays for JSON response, keep only descriptions
        if 'conversions' in conversions:
            for space_name, space_data in conversions['conversions'].items():
                if 'data' in space_data:
                    space_data.pop('data')  # Remove large numpy arrays
        
        # Convert numpy arrays and other types to serializable format
        conversions = convert_numpy_to_serializable(conversions)
        
        return jsonify(conversions)
    
    except Exception as e:
        app.logger.error(f"Color spaces error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/color_comparison/<filename>')
def color_comparison(filename):
    """Compare pixel values across color spaces"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))
        
        color_spaces = ColorSpaces(filepath)
        comparison = color_spaces.compare_color_spaces(x, y)
        
        # Convert numpy arrays and other types to serializable format
        comparison = convert_numpy_to_serializable(comparison)
        
        return jsonify(comparison)
    
    except Exception as e:
        app.logger.error(f"Color comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/color_distribution/<filename>')
def color_distribution(filename):
    """Analyze color distribution in specified color space"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        color_space = request.args.get('color_space', 'hsv')
        
        color_spaces = ColorSpaces(filepath)
        distribution = color_spaces.analyze_color_distribution(color_space)
        
        # Convert numpy arrays and other types to serializable format
        distribution = convert_numpy_to_serializable(distribution)
        
        return jsonify(distribution)
    
    except Exception as e:
        app.logger.error(f"Color distribution error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/color_palette/<filename>')
def color_palette(filename):
    """Extract dominant color palette"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        n_colors = int(request.args.get('n_colors', 8))
        color_space = request.args.get('color_space', 'rgb')
        
        color_spaces = ColorSpaces(filepath)
        palette = color_spaces.extract_color_palette(n_colors, color_space)
        
        # Convert numpy arrays and other types to serializable format
        palette = convert_numpy_to_serializable(palette)
        
        return jsonify(palette)
    
    except Exception as e:
        app.logger.error(f"Color palette error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Masks Module Routes
@app.route('/api/create_mask/<filename>', methods=['POST'])
def create_mask(filename):
    """Create geometric or color-based mask"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        data = request.get_json()
        mask_type = data.get('type', 'geometric')
        
        masks = Masks(filepath)
        
        if mask_type == 'geometric':
            shape = data.get('shape', 'circle')
            params = data.get('params', {})
            result = masks.create_geometric_mask(shape, params)
        elif mask_type == 'color':
            color_space = data.get('color_space', 'hsv')
            result = masks.create_color_mask(color_space, **data.get('params', {}))
        elif mask_type == 'threshold':
            method = data.get('method', 'global')
            result = masks.create_threshold_mask(method, **data.get('params', {}))
        else:
            return jsonify({'error': 'Invalid mask type'}), 400
        
        # Convert numpy arrays to lists for JSON serialization
        result = convert_numpy_to_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Create mask error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Edge Detection Module Routes
@app.route('/api/edge_detection/<filename>')
def edge_detection(filename):
    """Apply edge detection algorithms"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        method = request.args.get('method', 'canny')
        
        edge_detector = EdgeDetection(filepath)
        
        if method == 'canny':
            low_thresh = int(request.args.get('low_threshold', 50))
            high_thresh = int(request.args.get('high_threshold', 150))
            result = edge_detector.apply_canny(low_thresh, high_thresh)
        elif method == 'sobel':
            ksize = int(request.args.get('ksize', 3))
            result = edge_detector.apply_sobel(ksize)
        elif method == 'laplacian':
            ksize = int(request.args.get('ksize', 3))
            result = edge_detector.apply_laplacian(ksize)
        elif method == 'scharr':
            result = edge_detector.apply_scharr()
        elif method == 'prewitt':
            result = edge_detector.apply_prewitt()
        elif method == 'roberts':
            result = edge_detector.apply_roberts()
        elif method == 'log':
            sigma = float(request.args.get('sigma', 1.0))
            result = edge_detector.apply_log(sigma)
        elif method == 'compare':
            result = edge_detector.compare_edge_detectors()
        else:
            return jsonify({'error': 'Invalid edge detection method'}), 400
        
        # Convert numpy arrays to lists
        result = convert_numpy_to_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Edge detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Noise and Smoothing Module Routes
@app.route('/api/add_noise/<filename>', methods=['POST'])
def add_noise(filename):
    """Add noise to image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        data = request.get_json()
        noise_type = data.get('type', 'gaussian')
        
        noise_smoother = NoiseSmoothing(filepath)
        
        if noise_type == 'gaussian':
            mean = data.get('mean', 0)
            sigma = data.get('sigma', 25)
            result = noise_smoother.add_gaussian_noise(mean, sigma)
        elif noise_type == 'salt_pepper':
            salt_prob = data.get('salt_prob', 0.01)
            pepper_prob = data.get('pepper_prob', 0.01)
            result = noise_smoother.add_salt_pepper_noise(salt_prob, pepper_prob)
        elif noise_type == 'uniform':
            low = data.get('low', -30)
            high = data.get('high', 30)
            result = noise_smoother.add_uniform_noise(low, high)
        elif noise_type == 'speckle':
            intensity = data.get('intensity', 0.1)
            result = noise_smoother.add_speckle_noise(intensity)
        else:
            return jsonify({'error': 'Invalid noise type'}), 400
        
        # Save noisy image and return filename
        if 'noisy_image' in result:
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_filename = f"{base_name}_noisy_{noise_type}_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Convert RGB to BGR for saving
            noisy_bgr = cv2.cvtColor(result['noisy_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, noisy_bgr)
            
            result['output_filename'] = output_filename
            result.pop('noisy_image')  # Remove large array
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Add noise error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/apply_filter/<filename>', methods=['POST'])
def apply_filter(filename):
    """Apply smoothing or sharpening filters"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        data = request.get_json()
        filter_type = data.get('type', 'gaussian')
        
        noise_smoother = NoiseSmoothing(filepath)
        
        if filter_type == 'gaussian':
            kernel_size = data.get('kernel_size', 5)
            sigma_x = data.get('sigma_x', 0)
            result = noise_smoother.apply_gaussian_blur(kernel_size, sigma_x)
        elif filter_type == 'median':
            kernel_size = data.get('kernel_size', 5)
            result = noise_smoother.apply_median_filter(kernel_size)
        elif filter_type == 'bilateral':
            d = data.get('d', 9)
            sigma_color = data.get('sigma_color', 75)
            sigma_space = data.get('sigma_space', 75)
            result = noise_smoother.apply_bilateral_filter(d, sigma_color, sigma_space)
        elif filter_type == 'sharpen':
            intensity = data.get('intensity', 1.0)
            result = noise_smoother.apply_sharpen_filter(intensity)
        elif filter_type == 'unsharp':
            sigma = data.get('sigma', 1.0)
            strength = data.get('strength', 1.5)
            threshold = data.get('threshold', 0)
            result = noise_smoother.apply_unsharp_mask(sigma, strength, threshold)
        else:
            return jsonify({'error': 'Invalid filter type'}), 400
        
        # Save filtered image and return filename
        if 'filtered_image' in result:
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_filename = f"{base_name}_filtered_{filter_type}_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Convert RGB to BGR for saving
            filtered_bgr = cv2.cvtColor(result['filtered_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, filtered_bgr)
            
            result['output_filename'] = output_filename
            result.pop('filtered_image')  # Remove large array
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Apply filter error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Histogram Enhancement Module Routes
@app.route('/api/histogram_analysis/<filename>')
def histogram_analysis(filename):
    """Calculate and analyze image histogram"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        color_space = request.args.get('color_space', 'rgb')
        
        hist_enhancer = HistogramEnhancement(filepath)
        analysis = hist_enhancer.calculate_histogram(color_space)
        
        # Convert numpy arrays and other types to serializable format
        analysis = convert_numpy_to_serializable(analysis)
        
        return jsonify(analysis)
    
    except Exception as e:
        app.logger.error(f"Histogram analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/histogram_equalization/<filename>', methods=['POST'])
def histogram_equalization(filename):
    """Apply histogram equalization"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        data = request.get_json()
        method = data.get('method', 'global')
        
        hist_enhancer = HistogramEnhancement(filepath)
        
        if method == 'clahe':
            clip_limit = data.get('clip_limit', 2.0)
            tile_size = data.get('tile_grid_size', [8, 8])
            result = hist_enhancer.apply_clahe_advanced(clip_limit, tuple(tile_size))
        else:
            result = hist_enhancer.apply_histogram_equalization(method)
        
        # Save enhanced image
        if 'enhanced_image' in result:
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_filename = f"{base_name}_enhanced_{method}_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            # Convert RGB to BGR for saving
            enhanced_bgr = cv2.cvtColor(result['enhanced_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_bgr)
            
            result['output_filename'] = output_filename
            result.pop('enhanced_image')  # Remove large array
        
        # Convert numpy arrays and other types to serializable format
        result = convert_numpy_to_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Histogram equalization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Thresholding Module Routes
@app.route('/api/thresholding/<filename>', methods=['POST'])
def thresholding(filename):
    """Apply various thresholding techniques"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        data = request.get_json()
        method = data.get('method', 'global')
        
        thresholder = Thresholding(filepath)
        
        if method == 'global':
            threshold = data.get('threshold', 127)
            max_value = data.get('max_value', 255)
            thresh_type = data.get('threshold_type', 'binary')
            result = thresholder.apply_global_threshold(threshold, max_value, thresh_type)
        elif method == 'otsu':
            result = thresholder.apply_otsu_threshold()
        elif method == 'adaptive':
            max_value = data.get('max_value', 255)
            adaptive_method = data.get('adaptive_method', 'gaussian')
            thresh_type = data.get('threshold_type', 'binary')
            block_size = data.get('block_size', 11)
            c_constant = data.get('c_constant', 2)
            result = thresholder.apply_adaptive_threshold(max_value, adaptive_method, thresh_type, block_size, c_constant)
        elif method == 'triangle':
            result = thresholder.apply_triangle_threshold()
        elif method == 'li':
            result = thresholder.apply_li_threshold()
        elif method == 'multi':
            levels = data.get('levels', 3)
            result = thresholder.apply_multi_threshold(levels)
        elif method == 'compare':
            result = thresholder.compare_thresholding_methods()
        else:
            return jsonify({'error': 'Invalid thresholding method'}), 400
        
        # Save thresholded image
        if 'thresholded_image' in result:
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_filename = f"{base_name}_threshold_{method}_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            cv2.imwrite(output_path, result['thresholded_image'])
            
            result['output_filename'] = output_filename
            result.pop('thresholded_image')  # Remove large array
        
        # Convert any remaining numpy arrays to serializable format
        result = convert_numpy_to_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Thresholding error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    app.logger.error(f"Internal error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))
