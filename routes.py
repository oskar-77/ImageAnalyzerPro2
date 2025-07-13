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

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
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
        filename1 = str(uuid.uuid4()) + '_' + secure_filename(file1.filename)
        filename2 = str(uuid.uuid4()) + '_' + secure_filename(file2.filename)

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


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    app.logger.error(f"Internal error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))
