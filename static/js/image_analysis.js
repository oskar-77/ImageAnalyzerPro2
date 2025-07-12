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
            if (e.target.id === 'comparisonForm') {
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
    
    displayStatistics(data) {
        const container = document.getElementById('statisticsContent');
        if (!container) return;
        
        let html = `
            <div class="row g-4">
                <!-- Brightness Statistics -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-sun me-2"></i>Brightness Analysis</h6>
                        </div>
                        <div class="card-body">
                            <div class="row g-3">
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 mb-0 text-primary">${data.brightness.mean.toFixed(2)}</div>
                                        <small class="text-muted">Mean</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 mb-0 text-success">${data.brightness.std.toFixed(2)}</div>
                                        <small class="text-muted">Std Dev</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 mb-0 text-info">${data.brightness.min}</div>
                                        <small class="text-muted">Min</small>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="text-center">
                                        <div class="h4 mb-0 text-warning">${data.brightness.max}</div>
                                        <small class="text-muted">Max</small>
                                    </div>
                                </div>
                            </div>
                            <div class="mt-3">
                                <div class="progress">
                                    <div class="progress-bar bg-primary" style="width: ${(data.brightness.mean / 255 * 100).toFixed(1)}%"></div>
                                </div>
                                <small class="text-muted">Average brightness level</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Color Channel Statistics -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-palette me-2"></i>Color Channels</h6>
                        </div>
                        <div class="card-body">
        `;
        
        if (data.channels && data.channels.channel_stats) {
            Object.entries(data.channels.channel_stats).forEach(([channel, stats]) => {
                const color = channel.toLowerCase() === 'red' ? 'danger' : 
                             channel.toLowerCase() === 'green' ? 'success' : 
                             channel.toLowerCase() === 'blue' ? 'primary' : 'secondary';
                
                html += `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="fw-bold text-${color}">${channel}</span>
                        <span class="badge bg-${color}">${stats.mean.toFixed(1)}</span>
                    </div>
                `;
            });
        }
        
        html += `
                        </div>
                    </div>
                </div>
                
                <!-- Texture Features -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-texture me-2"></i>Texture Analysis</h6>
                        </div>
                        <div class="card-body">
        `;
        
        if (data.texture && !data.texture.error) {
            html += `
                <div class="row g-2">
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${(data.texture.edge_density * 100).toFixed(1)}%</div>
                            <small class="text-muted">Edge Density</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-success">${data.texture.edge_strength.toFixed(2)}</div>
                            <small class="text-muted">Edge Strength</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-info">${data.texture.laplacian_variance.toFixed(2)}</div>
                            <small class="text-muted">Laplacian Var</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-warning">${data.texture.local_binary_pattern_uniformity.toFixed(3)}</div>
                            <small class="text-muted">LBP Uniformity</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += `
                        </div>
                    </div>
                </div>
                
                <!-- Color Analysis -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-eye-dropper me-2"></i>Color Analysis</h6>
                        </div>
                        <div class="card-body">
        `;
        
        if (data.color && !data.color.error && data.color.message !== 'Color analysis not applicable to grayscale images') {
            html += `
                <div class="row g-2">
                    <div class="col-12">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span>Color Temperature</span>
                            <span class="badge bg-${data.color.color_temperature === 'warm' ? 'warning' : 'info'}">${data.color.color_temperature}</span>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${data.color.unique_colors.toLocaleString()}</div>
                            <small class="text-muted">Unique Colors</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-success">${(data.color.color_diversity * 100).toFixed(2)}%</div>
                            <small class="text-muted">Color Diversity</small>
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <h6 class="mb-2">HSV Analysis</h6>
                    <div class="row g-2">
                        <div class="col-4">
                            <div class="text-center">
                                <div class="fw-bold text-danger">${data.color.hsv_analysis.hue_mean.toFixed(1)}</div>
                                <small class="text-muted">Hue</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="text-center">
                                <div class="fw-bold text-success">${data.color.hsv_analysis.saturation_mean.toFixed(1)}</div>
                                <small class="text-muted">Saturation</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="text-center">
                                <div class="fw-bold text-primary">${data.color.hsv_analysis.value_mean.toFixed(1)}</div>
                                <small class="text-muted">Value</small>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `<div class="alert alert-info">Color analysis not available for grayscale images</div>`;
        }
        
        html += `
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
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
                    name: 'Red Channel',
                    line: { color: '#ff4444', width: 2 },
                    fill: 'tonexty'
                },
                {
                    x: data.green.bins,
                    y: data.green.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Green Channel',
                    line: { color: '#44ff44', width: 2 },
                    fill: 'tonexty'
                },
                {
                    x: data.blue.bins,
                    y: data.blue.values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Blue Channel',
                    line: { color: '#4444ff', width: 2 },
                    fill: 'tonexty'
                }
            );
        } else if (data.grayscale) {
            traces.push({
                x: data.grayscale.bins,
                y: data.grayscale.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Grayscale',
                line: { color: '#888888', width: 3 },
                fill: 'tozeroy'
            });
        }
        
        const layout = {
            title: {
                text: 'Image Histogram Analysis',
                font: { size: 18, color: '#ffffff' }
            },
            xaxis: { 
                title: 'Pixel Value (0-255)',
                gridcolor: '#444444',
                color: '#ffffff'
            },
            yaxis: { 
                title: 'Frequency',
                gridcolor: '#444444',
                color: '#ffffff'
            },
            template: 'plotly_dark',
            margin: { t: 60, r: 50, b: 60, l: 60 },
            showlegend: true,
            legend: {
                x: 0.7,
                y: 0.9,
                bgcolor: 'rgba(0,0,0,0.5)',
                bordercolor: '#444444',
                borderwidth: 1
            }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(container, traces, layout, config);
    }
    
    async handleImageComparison(event) {
        event.preventDefault();
        
        const form = event.target;
        const file2Input = form.querySelector('input[name="file2"]');
        const submitBtn = form.querySelector('button[type="submit"]');
        
        if (!file2Input.files[0]) {
            this.showError('Please select an image to compare');
            return;
        }
        
        try {
            this.showButtonLoading(submitBtn, 'Comparing...');
            
            // Create FormData with both files
            const formData = new FormData();
            
            // Get the current image file
            const currentImageResponse = await fetch(`/static/uploads/${this.currentImage}`);
            const currentImageBlob = await currentImageResponse.blob();
            const currentImageFile = new File([currentImageBlob], this.currentImage, { type: currentImageBlob.type });
            
            formData.append('file1', currentImageFile);
            formData.append('file2', file2Input.files[0]);
            
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
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
        
        if (file.size > maxSize) {
            this.showError('File size exceeds 16MB limit');
            event.target.value = '';
            return false;
        }
        
        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, TIFF, or WEBP files.');
            event.target.value = '';
            return false;
        }
        
        return true;
    }
    
    handleFileDrop(event) {
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = event.target.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
        }
    }
    
    showLoading(message = 'Loading...') {
        this.isLoading = true;
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="text-light">${message}</div>
            </div>
        `;
        document.body.appendChild(loadingOverlay);
    }
    
    hideLoading() {
        this.isLoading = false;
        const loadingOverlay = document.querySelector('.loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }
    }
    
    showButtonLoading(button, message) {
        const originalText = button.innerHTML;
        button.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${message}`;
        button.disabled = true;
        button.setAttribute('data-original-text', originalText);
    }
    
    hideButtonLoading(button, originalText) {
        button.innerHTML = originalText || button.getAttribute('data-original-text') || button.innerHTML;
        button.disabled = false;
        button.removeAttribute('data-original-text');
    }
    
    showError(message) {
        this.showToast(message, 'error');
    }
    
    showSuccess(message) {
        this.showToast(message, 'success');
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        const toastId = 'toast-' + Date.now();
        
        const bgClass = type === 'error' ? 'bg-danger' : 
                       type === 'success' ? 'bg-success' : 
                       type === 'warning' ? 'bg-warning' : 'bg-info';
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white ${bgClass} border-0`;
        toast.id = toastId;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
    
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }
    
    setCurrentImage(filename) {
        this.currentImage = filename;
    }
}

// Image conversion functions
async function convertImage(type) {
    const dashboard = window.imageDashboard;
    if (!dashboard || !dashboard.currentImage) {
        dashboard.showError('No image loaded');
        return;
    }
    
    try {
        dashboard.showLoading(`Converting to ${type}...`);
        
        const response = await fetch(`/api/convert/${dashboard.currentImage}?type=${type}`);
        const data = await response.json();
        
        if (data.error) {
            dashboard.showError(data.error);
        } else {
            dashboard.showSuccess(`Image converted to ${type}`);
            displayConvertedImage(data.converted_filename, type);
        }
    } catch (error) {
        console.error('Error converting image:', error);
        dashboard.showError('Failed to convert image');
    } finally {
        dashboard.hideLoading();
    }
}

// Image cropping functionality
let cropMode = false;
let cropStartX = 0;
let cropStartY = 0;
let cropEndX = 0;
let cropEndY = 0;
let cropBox = null;

function enableCropMode() {
    const dashboard = window.imageDashboard;
    if (!dashboard || !dashboard.currentImage) {
        dashboard.showError('No image loaded');
        return;
    }
    
    cropMode = true;
    const mainImage = document.getElementById('mainImage');
    if (mainImage) {
        mainImage.style.cursor = 'crosshair';
        dashboard.showToast('Click and drag to select area to crop', 'info');
        
        // Add crop event listeners
        mainImage.addEventListener('mousedown', startCrop);
        mainImage.addEventListener('mousemove', drawCrop);
        mainImage.addEventListener('mouseup', endCrop);
        
        // Add crop cancel button
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-outline-danger btn-sm crop-cancel-btn';
        cancelBtn.innerHTML = '<i class="fas fa-times me-1"></i>Cancel Crop';
        cancelBtn.style.position = 'absolute';
        cancelBtn.style.top = '10px';
        cancelBtn.style.right = '10px';
        cancelBtn.style.zIndex = '1000';
        cancelBtn.onclick = cancelCrop;
        
        const imageContainer = mainImage.parentElement;
        imageContainer.style.position = 'relative';
        imageContainer.appendChild(cancelBtn);
    }
}

function startCrop(e) {
    if (!cropMode) return;
    
    const rect = e.target.getBoundingClientRect();
    cropStartX = e.clientX - rect.left;
    cropStartY = e.clientY - rect.top;
    
    // Create crop box
    cropBox = document.createElement('div');
    cropBox.className = 'crop-box';
    cropBox.style.position = 'absolute';
    cropBox.style.border = '2px dashed #007bff';
    cropBox.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
    cropBox.style.pointerEvents = 'none';
    cropBox.style.zIndex = '999';
    
    const imageContainer = e.target.parentElement;
    imageContainer.appendChild(cropBox);
}

function drawCrop(e) {
    if (!cropMode || !cropBox) return;
    
    const rect = e.target.getBoundingClientRect();
    cropEndX = e.clientX - rect.left;
    cropEndY = e.clientY - rect.top;
    
    const x = Math.min(cropStartX, cropEndX);
    const y = Math.min(cropStartY, cropEndY);
    const width = Math.abs(cropEndX - cropStartX);
    const height = Math.abs(cropEndY - cropStartY);
    
    cropBox.style.left = x + 'px';
    cropBox.style.top = y + 'px';
    cropBox.style.width = width + 'px';
    cropBox.style.height = height + 'px';
}

function endCrop(e) {
    if (!cropMode || !cropBox) return;
    
    const rect = e.target.getBoundingClientRect();
    const imageNaturalWidth = e.target.naturalWidth;
    const imageNaturalHeight = e.target.naturalHeight;
    const imageDisplayWidth = e.target.clientWidth;
    const imageDisplayHeight = e.target.clientHeight;
    
    // Calculate scale factors
    const scaleX = imageNaturalWidth / imageDisplayWidth;
    const scaleY = imageNaturalHeight / imageDisplayHeight;
    
    // Convert to image coordinates
    const x = Math.min(cropStartX, cropEndX) * scaleX;
    const y = Math.min(cropStartY, cropEndY) * scaleY;
    const width = Math.abs(cropEndX - cropStartX) * scaleX;
    const height = Math.abs(cropEndY - cropStartY) * scaleY;
    
    if (width > 10 && height > 10) {
        // Show crop confirmation
        showCropConfirmation(Math.round(x), Math.round(y), Math.round(width), Math.round(height));
    } else {
        cancelCrop();
    }
}

function showCropConfirmation(x, y, width, height) {
    const dashboard = window.imageDashboard;
    
    const confirmDiv = document.createElement('div');
    confirmDiv.className = 'crop-confirmation';
    confirmDiv.innerHTML = `
        <div class="alert alert-info">
            <strong>Crop Selection:</strong> ${width}x${height} at (${x}, ${y})
            <div class="mt-2">
                <button class="btn btn-success btn-sm me-2" onclick="applyCrop(${x}, ${y}, ${width}, ${height})">
                    <i class="fas fa-check me-1"></i>Apply Crop
                </button>
                <button class="btn btn-secondary btn-sm" onclick="cancelCrop()">
                    <i class="fas fa-times me-1"></i>Cancel
                </button>
            </div>
        </div>
    `;
    
    const imageContainer = document.getElementById('mainImage').parentElement;
    imageContainer.appendChild(confirmDiv);
}

async function applyCrop(x, y, width, height) {
    const dashboard = window.imageDashboard;
    
    try {
        dashboard.showLoading('Cropping image...');
        
        const response = await fetch(`/api/crop/${dashboard.currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ x, y, width, height })
        });
        
        const data = await response.json();
        
        if (data.error) {
            dashboard.showError(data.error);
        } else {
            dashboard.showSuccess('Image cropped successfully');
            displayConvertedImage(data.cropped_filename, 'cropped');
        }
    } catch (error) {
        console.error('Error cropping image:', error);
        dashboard.showError('Failed to crop image');
    } finally {
        cancelCrop();
        dashboard.hideLoading();
    }
}

function cancelCrop() {
    cropMode = false;
    const mainImage = document.getElementById('mainImage');
    if (mainImage) {
        mainImage.style.cursor = 'default';
        mainImage.removeEventListener('mousedown', startCrop);
        mainImage.removeEventListener('mousemove', drawCrop);
        mainImage.removeEventListener('mouseup', endCrop);
    }
    
    // Remove crop elements
    const cropElements = document.querySelectorAll('.crop-box, .crop-cancel-btn, .crop-confirmation');
    cropElements.forEach(element => element.remove());
    
    cropBox = null;
}

// Image rotation and flipping
async function rotateImage(angle) {
    const dashboard = window.imageDashboard;
    if (!dashboard || !dashboard.currentImage) {
        dashboard.showError('No image loaded');
        return;
    }
    
    try {
        dashboard.showLoading('Rotating image...');
        
        const response = await fetch(`/api/rotate/${dashboard.currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ angle })
        });
        
        const data = await response.json();
        
        if (data.error) {
            dashboard.showError(data.error);
        } else {
            dashboard.showSuccess(`Image rotated ${angle}°`);
            displayConvertedImage(data.rotated_filename, `rotated_${angle}`);
        }
    } catch (error) {
        console.error('Error rotating image:', error);
        dashboard.showError('Failed to rotate image');
    } finally {
        dashboard.hideLoading();
    }
}

async function flipImage(direction) {
    const dashboard = window.imageDashboard;
    if (!dashboard || !dashboard.currentImage) {
        dashboard.showError('No image loaded');
        return;
    }
    
    try {
        dashboard.showLoading('Flipping image...');
        
        const response = await fetch(`/api/flip/${dashboard.currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ direction })
        });
        
        const data = await response.json();
        
        if (data.error) {
            dashboard.showError(data.error);
        } else {
            dashboard.showSuccess(`Image flipped ${direction}ly`);
            displayConvertedImage(data.flipped_filename, `flipped_${direction}`);
        }
    } catch (error) {
        console.error('Error flipping image:', error);
        dashboard.showError('Failed to flip image');
    } finally {
        dashboard.hideLoading();
    }
}

function displayConvertedImage(filename, type) {
    const container = document.getElementById('convertedImages');
    if (!container) return;
    
    // Clear existing content or initialize
    if (container.innerHTML.includes('alert-info')) {
        container.innerHTML = '<div class="converted-images"></div>';
    }
    
    const imagesGrid = container.querySelector('.converted-images') || container;
    
    const imageCard = document.createElement('div');
    imageCard.className = 'converted-image-card';
    imageCard.innerHTML = `
        <img src="/static/uploads/${filename}" alt="${type} conversion" class="img-fluid">
        <div class="card-body">
            <h6 class="card-title">${type.charAt(0).toUpperCase() + type.slice(1)}</h6>
            <a href="/static/uploads/${filename}" download class="btn btn-outline-primary btn-sm">
                <i class="fas fa-download me-1"></i>Download
            </a>
        </div>
    `;
    
    imagesGrid.appendChild(imageCard);
}

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

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.imageDashboard = new ImageAnalysisDashboard();
    
    // Set current image from URL if on analysis page
    const pathParts = window.location.pathname.split('/');
    if (pathParts[1] === 'analysis' && pathParts[2]) {
        window.imageDashboard.setCurrentImage(pathParts[2]);
    }
});



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
