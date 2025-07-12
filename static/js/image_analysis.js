/**
 * Image Analysis Dashboard JavaScript
 * Handles interactive functionality for image analysis
 */

class ImageAnalysisDashboard {
    constructor() {
        this.currentImage = null;
        this.imageInfo = null;
        this.statisticsData = null;
        this.histogramData = null;
        this.isLoading = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.setupTooltips();
    }
    
    setupEventListeners() {
        // Image click handler for pixel exploration
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('image-clickable')) {
                this.handleImageClick(e);
            }
        });
        
        // Tab change handlers
        document.addEventListener('shown.bs.tab', (e) => {
            this.handleTabChange(e);
        });
        
        // Form submission handlers
        document.addEventListener('submit', (e) => {
            if (e.target.id === 'compareForm') {
                e.preventDefault();
                this.handleImageComparison(e);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }
    
    setupFileUpload() {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        fileInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.validateFileInput(e);
            });
        });
        
        // Drag and drop functionality
        const uploadAreas = document.querySelectorAll('.file-upload-area');
        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });
            
            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
            });
            
            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
                this.handleFileDrop(e);
            });
        });
    }
    
    setupTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    handleImageClick(event) {
        const image = event.target;
        const rect = image.getBoundingClientRect();
        const scaleX = image.naturalWidth / image.width;
        const scaleY = image.naturalHeight / image.height;
        
        const x = Math.round((event.clientX - rect.left) * scaleX);
        const y = Math.round((event.clientY - rect.top) * scaleY);
        
        // Show pixel marker
        this.showPixelMarker(event.clientX - rect.left, event.clientY - rect.top, image);
        
        // Get pixel value
        this.getPixelValue(x, y);
    }
    
    showPixelMarker(x, y, image) {
        // Remove existing marker
        const existingMarker = document.querySelector('.pixel-marker');
        if (existingMarker) {
            existingMarker.remove();
        }
        
        // Create new marker
        const marker = document.createElement('div');
        marker.className = 'pixel-marker';
        marker.style.left = `${x}px`;
        marker.style.top = `${y}px`;
        
        // Add to image container
        const container = image.closest('.position-relative') || image.parentElement;
        if (container) {
            container.style.position = 'relative';
            container.appendChild(marker);
        }
    }
    
    async getPixelValue(x, y) {
        if (!this.currentImage) return;
        
        try {
            this.showLoading('Getting pixel value...');
            
            const response = await fetch(`/api/pixel_value/${this.currentImage}?x=${x}&y=${y}`);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.displayPixelValue(data);
            }
        } catch (error) {
            console.error('Error getting pixel value:', error);
            this.showError('Failed to get pixel value');
        } finally {
            this.hideLoading();
        }
    }
    
    displayPixelValue(data) {
        const pixelContainer = document.getElementById('pixelValues');
        const colorContainer = document.getElementById('colorInfo');
        
        if (!pixelContainer || !colorContainer) return;
        
        // Update pixel values
        pixelContainer.innerHTML = this.generatePixelValueHTML(data);
        
        // Update color information
        colorContainer.innerHTML = this.generateColorInfoHTML(data);
        
        // Add animation
        pixelContainer.classList.add('fade-in');
        colorContainer.classList.add('fade-in');
        
        setTimeout(() => {
            pixelContainer.classList.remove('fade-in');
            colorContainer.classList.remove('fade-in');
        }, 300);
    }
    
    generatePixelValueHTML(data) {
        return `
            <div class="pixel-value-grid">
                <div class="pixel-value-item">
                    <div class="value">(${data.coordinates.x}, ${data.coordinates.y})</div>
                    <div class="label">Position</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.rgb.r}</div>
                    <div class="label">Red</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.rgb.g}</div>
                    <div class="label">Green</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.rgb.b}</div>
                    <div class="label">Blue</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.hsv.h}</div>
                    <div class="label">Hue</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.hsv.s}</div>
                    <div class="label">Saturation</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.hsv.v}</div>
                    <div class="label">Value</div>
                </div>
                <div class="pixel-value-item">
                    <div class="value">${data.brightness.toFixed(2)}</div>
                    <div class="label">Brightness</div>
                </div>
            </div>
        `;
    }
    
    generateColorInfoHTML(data) {
        return `
            <div class="d-flex align-items-center mb-3">
                <div class="color-swatch me-3" style="width: 60px; height: 60px; background-color: ${data.hex};"></div>
                <div>
                    <div class="fw-bold">${data.hex}</div>
                    <div class="text-muted">rgb(${data.rgb.r}, ${data.rgb.g}, ${data.rgb.b})</div>
                    <div class="text-muted">hsv(${data.hsv.h}, ${data.hsv.s}, ${data.hsv.v})</div>
                </div>
            </div>
            <div class="row g-2">
                <div class="col-6">
                    <div class="bg-body-secondary p-2 rounded">
                        <strong>LAB L:</strong> ${data.lab.l}
                    </div>
                </div>
                <div class="col-6">
                    <div class="bg-body-secondary p-2 rounded">
                        <strong>LAB A:</strong> ${data.lab.a}
                    </div>
                </div>
            </div>
        `;
    }
    
    handleTabChange(event) {
        const tabId = event.target.getAttribute('data-bs-target');
        
        switch (tabId) {
            case '#statistics':
                if (!this.statisticsData) {
                    this.loadStatistics();
                }
                break;
            case '#histogram':
                if (!this.histogramData) {
                    this.loadHistogram();
                }
                break;
        }
    }
    
    async loadStatistics() {
        if (!this.currentImage) return;
        
        try {
            this.showLoading('Loading statistics...');
            
            const response = await fetch(`/api/statistics/${this.currentImage}`);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.statisticsData = data;
                this.displayStatistics(data);
            }
        } catch (error) {
            console.error('Error loading statistics:', error);
            this.showError('Failed to load statistics');
        } finally {
            this.hideLoading();
        }
    }
    
    async loadHistogram() {
        if (!this.currentImage) return;
        
        try {
            this.showLoading('Loading histogram...');
            
            const response = await fetch(`/api/histogram/${this.currentImage}`);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.histogramData = data;
                this.displayHistogram(data);
            }
        } catch (error) {
            console.error('Error loading histogram:', error);
            this.showError('Failed to load histogram');
        } finally {
            this.hideLoading();
        }
    }
    
    displayHistogram(data) {
        const container = document.getElementById('histogramContent');
        if (!container) return;
        
        const traces = [];
        
        if (data.red && data.green && data.blue) {
            traces.push(
                {
                    x: data.red.bins,
                    y: data.red.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Red',
                    line: { color: 'red', width: 2 }
                },
                {
                    x: data.green.bins,
                    y: data.green.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Green',
                    line: { color: 'green', width: 2 }
                },
                {
                    x: data.blue.bins,
                    y: data.blue.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Blue',
                    line: { color: 'blue', width: 2 }
                }
            );
        } else if (data.grayscale) {
            traces.push({
                x: data.grayscale.bins,
                y: data.grayscale.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Grayscale',
                line: { color: 'gray', width: 2 }
            });
        }
        
        const layout = {
            title: 'Image Histogram',
            xaxis: { title: 'Pixel Value' },
            yaxis: { title: 'Frequency' },
            template: 'plotly_dark',
            margin: { t: 50, r: 50, b: 50, l: 50 }
        };
        
        Plotly.newPlot(container, traces, layout, { responsive: true });
    }
    
    async handleImageComparison(event) {
        const form = event.target;
        const formData = new FormData(form);
        const submitBtn = form.querySelector('button[type="submit"]');
        
        try {
            this.showButtonLoading(submitBtn, 'Comparing...');
            
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.displayComparisonResults(data);
            }
        } catch (error) {
            console.error('Error comparing images:', error);
            this.showError('Failed to compare images');
        } finally {
            this.hideButtonLoading(submitBtn, 'Compare Images');
        }
    }
    
    displayComparisonResults(data) {
        const modal = document.getElementById('comparisonModal');
        const resultsContainer = document.getElementById('comparisonResults');
        
        if (!modal || !resultsContainer) return;
        
        resultsContainer.innerHTML = this.generateComparisonResultsHTML(data);
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
    
    generateComparisonResultsHTML(data) {
        return `
            <div class="row">
                <div class="col-md-4">
                    <h6>Original Images</h6>
                    <div class="mb-3">
                        <img src="/static/uploads/${data.image1}" class="img-fluid rounded" alt="Image 1">
                    </div>
                    <div class="mb-3">
                        <img src="/static/uploads/${data.image2}" class="img-fluid rounded" alt="Image 2">
                    </div>
                </div>
                <div class="col-md-8">
                    <h6>Similarity Metrics</h6>
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="comparison-metric">
                                <div class="metric-value">${(data.ssim.ssim * 100).toFixed(2)}%</div>
                                <div class="metric-label">SSIM</div>
                                <small class="text-muted">${data.ssim.interpretation}</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="comparison-metric">
                                <div class="metric-value">${data.psnr.psnr.toFixed(2)} dB</div>
                                <div class="metric-label">PSNR</div>
                                <small class="text-muted">${data.psnr.interpretation}</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="comparison-metric">
                                <div class="metric-value">${data.mse.mse.toFixed(2)}</div>
                                <div class="metric-label">MSE</div>
                                <small class="text-muted">${data.mse.interpretation}</small>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="comparison-metric">
                                <div class="metric-value">${data.overall_similarity.percentage.toFixed(1)}%</div>
                                <div class="metric-label">Overall</div>
                                <small class="text-muted">${data.overall_similarity.interpretation}</small>
                            </div>
                        </div>
                    </div>
                    ${data.difference_map ? `
                        <div class="mt-4">
                            <h6>Difference Map</h6>
                            <img src="/static/uploads/${data.difference_map}" class="img-fluid rounded" alt="Difference Map">
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    handleKeyboardShortcuts(event) {
        // Ctrl+U for upload
        if (event.ctrlKey && event.key === 'u') {
            event.preventDefault();
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput) fileInput.click();
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) bsModal.hide();
            });
        }
    }
    
    validateFileInput(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const maxSize = 16 * 1024 * 1024; // 16MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
        
        if (file.size > maxSize) {
            this.showError('File size must be less than 16MB');
            event.target.value = '';
            return;
        }
        
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (JPEG, PNG, GIF, BMP, WEBP)');
            event.target.value = '';
            return;
        }
        
        this.showSuccess('File selected successfully');
    }
    
    handleFileDrop(event) {
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = event.target.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.files = files;
                this.validateFileInput({ target: fileInput });
            }
        }
    }
    
    showLoading(message = 'Loading...') {
        this.isLoading = true;
        
        // Create loading overlay
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="text-light">${message}</div>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }
    
    hideLoading() {
        this.isLoading = false;
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
    
    showButtonLoading(button, message) {
        button.disabled = true;
        button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${message}`;
    }
    
    hideButtonLoading(button, originalText) {
        button.disabled = false;
        button.innerHTML = originalText;
    }
    
    showError(message) {
        this.showToast(message, 'error');
    }
    
    showSuccess(message) {
        this.showToast(message, 'success');
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to toast container or create one
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast after hiding
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
    
    setCurrentImage(filename) {
        this.currentImage = filename;
        // Reset cached data
        this.statisticsData = null;
        this.histogramData = null;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.imageAnalysisDashboard = new ImageAnalysisDashboard();
    
    // Set current image if we're on analysis page
    const analysisPage = document.querySelector('[data-current-image]');
    if (analysisPage) {
        const filename = analysisPage.getAttribute('data-current-image');
        window.imageAnalysisDashboard.setCurrentImage(filename);
    }
});

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add fade-in animation class
const style = document.createElement('style');
style.textContent = `
    .fade-in {
        animation: fadeIn 0.3s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
