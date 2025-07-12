# Image Analysis Dashboard

A comprehensive Python web application for image analysis featuring pixel exploration, statistics, conversions, and comparisons with interactive visualizations.

## Features

### Core Image Analysis
- **Pixel Value Exploration**: Click on any pixel to explore RGB, HSV, LAB values and brightness information
- **Image Statistics**: Comprehensive brightness, histogram, and pixel distribution analysis
- **Texture Analysis**: Edge detection, entropy, and texture feature extraction
- **Color Analysis**: Color temperature, diversity, and distribution metrics

### Image Conversions
- **Format Conversions**: RGB to Grayscale, Binary, HSV, LAB
- **Image Enhancements**: Brightness/contrast adjustment, gamma correction
- **Filtering**: Gaussian blur, edge detection (Canny, Sobel, Laplacian)
- **Transformations**: Resize, rotate, flip operations

### Image Comparison
- **Similarity Metrics**: SSIM (Structural Similarity Index), PSNR, MSE
- **Visual Comparison**: Side-by-side display with difference maps
- **Histogram Comparison**: Color channel correlation analysis
- **Comprehensive Reports**: Detailed comparison metrics and interpretations

### Interactive Dashboard
- **Responsive Design**: Modern Bootstrap 5 interface with dark theme
- **Interactive Charts**: Plotly.js visualizations for histograms and statistics
- **Real-time Analysis**: Live pixel exploration and instant feedback
- **Progress Indicators**: Loading states and status messages
- **Report Generation**: Download analysis reports in TXT and PNG formats

## Technology Stack

### Backend
- **Flask**: Web framework for API and routing
- **OpenCV (cv2)**: Core image processing operations
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Statistical visualizations and report generation
- **scikit-image**: Advanced image analysis metrics (SSIM, PSNR)
- **Pillow**: Image format handling and metadata extraction
- **SciPy**: Statistical analysis and texture features

### Frontend
- **HTML5 & CSS3**: Modern web standards
- **Bootstrap 5**: Responsive UI framework with dark theme
- **Plotly.js**: Interactive data visualizations
- **Vanilla JavaScript**: Client-side interactivity
- **Font Awesome**: Icon library

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download the project files**
   ```bash
   # All files should be in the current directory
   ls -la
   ```

2. **Install dependencies**
   ```bash
   pip install flask opencv-python numpy matplotlib scikit-image pillow scipy plotly
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p static/uploads static/reports
   ```

4. **Set environment variables (optional)**
   ```bash
   export SESSION_SECRET="your-secret-key-here"
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

### Production Deployment

For production deployment, consider:

1. **Set a secure session secret**
   ```bash
   export SESSION_SECRET="your-very-secure-secret-key"
   ```

2. **Configure file upload limits**
   - Current limit: 16MB per file
   - Modify `MAX_CONTENT_LENGTH` in `app.py` if needed

3. **Set up file cleanup**
   - Implement periodic cleanup of old uploaded files
   - Use the `ImageUtils.clean_temp_files()` method

4. **Configure logging**
   - Set appropriate log levels for production
   - Consider using a proper logging service

## Usage Guide

### Single Image Analysis

1. **Upload an Image**
   - Click "Upload & Analyze" on the home page
   - Select an image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
   - Maximum file size: 16MB

2. **Explore Pixels**
   - Click anywhere on the uploaded image
   - View RGB, HSV, LAB values and brightness
   - See color swatches and hex values

3. **View Statistics**
   - Click the "Statistics" tab
   - Review brightness statistics, channel analysis
   - Examine texture features and color distribution

4. **Generate Histograms**
   - Click the "Histogram" tab
   - View interactive color channel histograms
   - Analyze pixel value distributions

5. **Apply Conversions**
   - Click the "Conversions" tab
   - Convert to Grayscale, Binary, or HSV
   - Download converted images

6. **Generate Reports**
   - Click the "Reports" tab
   - Download TXT reports with statistics
   - Download PNG reports with visualizations

### Image Comparison

1. **Select Two Images**
   - Use the "Image Comparison" section on the home page
   - Choose two images to compare

2. **View Comparison Results**
   - SSIM: Structural similarity (0-100%)
   - PSNR: Peak signal-to-noise ratio (dB)
   - MSE: Mean squared error
   - Overall similarity percentage

3. **Analyze Difference Maps**
   - Visual representation of pixel differences
   - Color-coded difference intensity
   - Side-by-side comparison view

## API Endpoints

### Image Analysis
- `GET /api/image_info/<filename>` - Basic image information
- `GET /api/pixel_value/<filename>?x=<x>&y=<y>` - Pixel value at coordinates
- `GET /api/statistics/<filename>` - Comprehensive image statistics
- `GET /api/histogram/<filename>` - Histogram data for plotting

### Image Processing
- `GET /api/convert/<filename>?type=<type>` - Convert image format
- `POST /api/compare` - Compare two images
- `GET /api/report/<filename>?format=<format>` - Generate analysis report

### File Management
- `POST /upload` - Upload image file
- `GET /analysis/<filename>` - Analysis page for uploaded image

## Project Structure

