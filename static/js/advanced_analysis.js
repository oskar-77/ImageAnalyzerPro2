/**
 * Advanced Image Analysis JavaScript
 * Handles new advanced image processing features
 */

let currentImage = '';
let currentPanel = 'basic';

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeAdvancedAnalysis();
});

function initializeAdvancedAnalysis() {
    // Get current image filename from the image element
    const mainImage = document.getElementById('mainImage');
    if (mainImage) {
        currentImage = mainImage.dataset.currentImage;
        
        // Setup threshold slider
        setupThresholdSlider();
        
        // Initialize with basic panel
        showAnalysisPanel('basic');
    }
}

// Panel Management Functions
function showAnalysisPanel(panelName) {
    // Hide all panels
    const panels = document.querySelectorAll('.analysis-panel');
    panels.forEach(panel => panel.classList.add('d-none'));
    
    // Show selected panel
    const selectedPanel = document.getElementById(panelName + 'Panel');
    if (selectedPanel) {
        selectedPanel.classList.remove('d-none');
        currentPanel = panelName;
        
        // Update panel title
        updatePanelTitle(panelName);
        
        // Update button states
        updateButtonStates(panelName);
    }
}

function updatePanelTitle(panelName) {
    const titleElement = document.getElementById('analysisPanelTitle');
    const titles = {
        'basic': '<i class="fas fa-info-circle me-2"></i>معلومات الصورة',
        'digital': '<i class="fas fa-digital-tachograph me-2"></i>الخصائص الرقمية',
        'colors': '<i class="fas fa-palette me-2"></i>فضاءات الألوان',
        'histogram': '<i class="fas fa-chart-bar me-2"></i>تحليل الهستوغرام',
        'edges': '<i class="fas fa-border-style me-2"></i>كشف الحواف',
        'filters': '<i class="fas fa-filter me-2"></i>المرشحات والضوضاء',
        'threshold': '<i class="fas fa-adjust me-2"></i>العتبة والتقسيم',
        'masks': '<i class="fas fa-mask me-2"></i>الأقنعة'
    };
    
    if (titleElement && titles[panelName]) {
        titleElement.innerHTML = titles[panelName];
    }
}

function updateButtonStates(activePanelName) {
    // Remove active state from all buttons
    const buttons = document.querySelectorAll('.btn-group-vertical .btn');
    buttons.forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-outline-primary');
    });
    
    // Add active state to clicked button
    const activeButton = document.querySelector(`[onclick="showAnalysisPanel('${activePanelName}')"]`);
    if (activeButton) {
        activeButton.classList.remove('btn-outline-primary');
        activeButton.classList.add('btn-primary');
    }
}

function setupThresholdSlider() {
    const slider = document.getElementById('thresholdValue');
    const display = document.getElementById('thresholdValueDisplay');
    
    if (slider && display) {
        slider.addEventListener('input', function() {
            display.textContent = slider.value;
        });
    }
}

// Digital Properties Functions
async function loadDigitalProperties() {
    showLoading('digitalProperties');
    
    try {
        const response = await fetch(`/api/digital_properties/${currentImage}`);
        const data = await response.json();
        
        if (data.error) {
            showError('digitalProperties', data.error);
            return;
        }
        
        displayDigitalProperties(data);
    } catch (error) {
        showError('digitalProperties', 'فشل في تحميل الخصائص الرقمية');
    }
}

function displayDigitalProperties(data) {
    const container = document.getElementById('digitalProperties');
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>الخصائص الأساسية</h6>
                <div class="row g-2">
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${data.dimensions.width}</div>
                            <small class="text-muted">العرض</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${data.dimensions.height}</div>
                            <small class="text-muted">الارتفاع</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${data.dimensions.channels}</div>
                            <small class="text-muted">القنوات</small>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="bg-body-secondary p-2 rounded text-center">
                            <div class="fw-bold text-primary">${data.bit_depth}</div>
                            <small class="text-muted">عمق البت</small>
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <h6>نسبة العرض إلى الارتفاع</h6>
                    <p class="mb-1">${data.aspect_ratio.ratio} (${data.aspect_ratio.description})</p>
                </div>
                <div class="mt-3">
                    <h6>إحصائيات البكسل</h6>
                    <p class="mb-1">إجمالي البكسل: ${data.total_pixels.toLocaleString()}</p>
                    <p class="mb-1">متوسط قيم البكسل: ${data.pixel_statistics.mean.toFixed(2)}</p>
                    <p class="mb-1">الانحراف المعياري: ${data.pixel_statistics.std.toFixed(2)}</p>
                </div>
            </div>
        </div>
    `;
}

// Color Spaces Functions
async function loadColorSpaces() {
    showLoading('colorSpacesInfo');
    
    try {
        const response = await fetch(`/api/color_spaces/${currentImage}`);
        const data = await response.json();
        
        if (data.error) {
            showError('colorSpacesInfo', data.error);
            return;
        }
        
        displayColorSpaces(data);
    } catch (error) {
        showError('colorSpacesInfo', 'فشل في تحميل فضاءات الألوان');
    }
}

function displayColorSpaces(data) {
    const container = document.getElementById('colorSpacesInfo');
    let html = '<div class="row">';
    
    Object.entries(data.conversions).forEach(([space, info]) => {
        html += `
            <div class="col-md-6 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">${space.toUpperCase()}</h6>
                        <p class="card-text">${info.description}</p>
                        <div class="mt-2">
                            <small class="text-muted">قنوات: ${info.channels.join(', ')}</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

async function extractColorPalette() {
    showLoading('colorPalette');
    
    try {
        const response = await fetch(`/api/color_palette/${currentImage}?n_colors=8&color_space=rgb`);
        const data = await response.json();
        
        if (data.error) {
            showError('colorPalette', data.error);
            return;
        }
        
        displayColorPalette(data);
    } catch (error) {
        showError('colorPalette', 'فشل في استخراج لوحة الألوان');
    }
}

function displayColorPalette(data) {
    const container = document.getElementById('colorPalette');
    let html = '<h6>لوحة الألوان المهيمنة</h6><div class="row">';
    
    data.palette.forEach((color, index) => {
        const rgbColor = `rgb(${color.color[0]}, ${color.color[1]}, ${color.color[2]})`;
        html += `
            <div class="col-3 mb-2">
                <div class="text-center">
                    <div style="width: 40px; height: 40px; background-color: ${rgbColor}; border: 1px solid #ccc; border-radius: 4px; margin: 0 auto;"></div>
                    <small class="text-muted">${color.percentage.toFixed(1)}%</small>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// Histogram Functions
async function analyzeHistogram() {
    const colorSpace = document.getElementById('histogramColorSpace').value;
    showLoading('histogramAnalysis');
    
    try {
        const response = await fetch(`/api/histogram_analysis/${currentImage}?color_space=${colorSpace}`);
        const data = await response.json();
        
        if (data.error) {
            showError('histogramAnalysis', data.error);
            return;
        }
        
        displayHistogramAnalysis(data);
    } catch (error) {
        showError('histogramAnalysis', 'فشل في تحليل الهستوغرام');
    }
}

function displayHistogramAnalysis(data) {
    const container = document.getElementById('histogramAnalysis');
    let html = `
        <h6>تحليل الهستوغرام - ${data.color_space}</h6>
        <div class="row g-2">
    `;
    
    Object.entries(data.channels).forEach(([channel, info]) => {
        html += `
            <div class="col-12 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">${channel}</h6>
                        <div class="row">
                            <div class="col-6">
                                <small>المتوسط: ${info.statistics.mean.toFixed(2)}</small><br>
                                <small>الوسيط: ${info.statistics.median.toFixed(2)}</small><br>
                                <small>المنوال: ${info.statistics.mode}</small>
                            </div>
                            <div class="col-6">
                                <small>الحد الأدنى: ${info.statistics.min}</small><br>
                                <small>الحد الأقصى: ${info.statistics.max}</small><br>
                                <small>المدى: ${info.statistics.dynamic_range}</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

async function applyHistogramEqualization(method) {
    showLoading('histogramEnhancement');
    
    try {
        const response = await fetch(`/api/histogram_equalization/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ method: method })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError('histogramEnhancement', data.error);
            return;
        }
        
        displayHistogramEnhancement(data);
    } catch (error) {
        showError('histogramEnhancement', 'فشل في تحسين التباين');
    }
}

function displayHistogramEnhancement(data) {
    const container = document.getElementById('histogramEnhancement');
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>نتائج تحسين التباين</h6>
                <p><strong>الطريقة:</strong> ${data.method}</p>
                <p><strong>التباين الأصلي:</strong> ${data.metrics.original_contrast.toFixed(2)}</p>
                <p><strong>التباين المحسن:</strong> ${data.metrics.enhanced_contrast.toFixed(2)}</p>
                <p><strong>نسبة التحسن:</strong> ${data.metrics.contrast_improvement_percentage.toFixed(2)}%</p>
                ${data.output_filename ? `
                    <div class="mt-3">
                        <img src="/static/uploads/${data.output_filename}" class="img-fluid rounded" alt="Enhanced Image">
                        <div class="mt-2">
                            <button class="btn btn-sm btn-primary" onclick="updateMainImage('${data.output_filename}')">
                                استخدام هذه الصورة
                            </button>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Edge Detection Functions
async function detectEdges() {
    const method = document.getElementById('edgeMethod').value;
    showLoading('edgeDetection');
    
    try {
        const response = await fetch(`/api/edge_detection/${currentImage}?method=${method}`);
        const data = await response.json();
        
        if (data.error) {
            showError('edgeDetection', data.error);
            return;
        }
        
        displayEdgeDetection(data);
    } catch (error) {
        showError('edgeDetection', 'فشل في كشف الحواف');
    }
}

function displayEdgeDetection(data) {
    const container = document.getElementById('edgeDetection');
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>نتائج كشف الحواف</h6>
                <p><strong>الخوارزمية:</strong> ${data.algorithm}</p>
                <p><strong>الوصف:</strong> ${data.description}</p>
                ${data.edge_statistics ? `
                    <div class="mt-3">
                        <h6>إحصائيات الحواف</h6>
                        <p>بكسل الحواف: ${data.edge_statistics.edge_pixels.toLocaleString()}</p>
                        <p>كثافة الحواف: ${data.edge_statistics.edge_density_percentage.toFixed(2)}%</p>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Noise and Filter Functions
async function addNoise() {
    const noiseType = document.getElementById('noiseType').value;
    showLoading('filterResults');
    
    try {
        const response = await fetch(`/api/add_noise/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                type: noiseType,
                mean: 0,
                sigma: 25,
                salt_prob: 0.01,
                pepper_prob: 0.01
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError('filterResults', data.error);
            return;
        }
        
        displayFilterResult(data, 'ضوضاء');
    } catch (error) {
        showError('filterResults', 'فشل في إضافة الضوضاء');
    }
}

async function applyFilter() {
    const filterType = document.getElementById('filterType').value;
    showLoading('filterResults');
    
    try {
        const response = await fetch(`/api/apply_filter/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                type: filterType,
                kernel_size: 5,
                sigma_x: 0
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError('filterResults', data.error);
            return;
        }
        
        displayFilterResult(data, 'مرشح');
    } catch (error) {
        showError('filterResults', 'فشل في تطبيق المرشح');
    }
}

function displayFilterResult(data, operation) {
    const container = document.getElementById('filterResults');
    const filename = data.output_filename;
    
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>نتيجة ${operation}</h6>
                <p><strong>النوع:</strong> ${data.filter_type || data.noise_type}</p>
                <p><strong>الوصف:</strong> ${data.description}</p>
                ${filename ? `
                    <div class="mt-3">
                        <img src="/static/uploads/${filename}" class="img-fluid rounded" alt="Processed Image">
                        <div class="mt-2">
                            <button class="btn btn-sm btn-primary" onclick="updateMainImage('${filename}')">
                                استخدام هذه الصورة
                            </button>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Thresholding Functions
async function applyThresholding() {
    const method = document.getElementById('thresholdMethod').value;
    const threshold = document.getElementById('thresholdValue').value;
    showLoading('thresholdResults');
    
    try {
        const requestBody = { method: method };
        if (method === 'global') {
            requestBody.threshold = parseInt(threshold);
        }
        
        const response = await fetch(`/api/thresholding/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError('thresholdResults', data.error);
            return;
        }
        
        displayThresholdingResult(data);
    } catch (error) {
        showError('thresholdResults', 'فشل في تطبيق العتبة');
    }
}

function displayThresholdingResult(data) {
    const container = document.getElementById('thresholdResults');
    const filename = data.output_filename;
    
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>نتيجة العتبة</h6>
                <p><strong>الطريقة:</strong> ${data.description}</p>
                ${data.statistics ? `
                    <div class="mt-3">
                        <h6>الإحصائيات</h6>
                        <p>بكسل المقدمة: ${data.statistics.foreground_percentage.toFixed(2)}%</p>
                        <p>بكسل الخلفية: ${data.statistics.background_percentage.toFixed(2)}%</p>
                    </div>
                ` : ''}
                ${filename ? `
                    <div class="mt-3">
                        <img src="/static/uploads/${filename}" class="img-fluid rounded" alt="Thresholded Image">
                        <div class="mt-2">
                            <button class="btn btn-sm btn-primary" onclick="updateMainImage('${filename}')">
                                استخدام هذه الصورة
                            </button>
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Mask Functions
async function createMask() {
    const maskType = document.getElementById('maskType').value;
    showLoading('maskResults');
    
    try {
        const requestBody = { 
            type: maskType,
            params: {}
        };
        
        if (maskType === 'geometric') {
            requestBody.shape = 'circle';
            requestBody.params = {
                center_x: 100,
                center_y: 100,
                radius: 50
            };
        }
        
        const response = await fetch(`/api/create_mask/${currentImage}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError('maskResults', data.error);
            return;
        }
        
        displayMaskResult(data);
    } catch (error) {
        showError('maskResults', 'فشل في إنشاء القناع');
    }
}

function displayMaskResult(data) {
    const container = document.getElementById('maskResults');
    container.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h6>نتيجة القناع</h6>
                ${data.mask_info ? `
                    <p><strong>النوع:</strong> ${data.mask_info.shape || data.mask_info.color_space}</p>
                ` : ''}
                ${data.statistics ? `
                    <div class="mt-3">
                        <h6>الإحصائيات</h6>
                        <p>بكسل القناع: ${data.statistics.mask_pixels.toLocaleString()}</p>
                        <p>نسبة التغطية: ${data.statistics.coverage_percentage.toFixed(2)}%</p>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Utility Functions
function updateMainImage(filename) {
    const mainImage = document.getElementById('mainImage');
    mainImage.src = `/static/uploads/${filename}`;
    mainImage.dataset.currentImage = filename;
    currentImage = filename;
}

function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">جاري التحميل...</span>
                </div>
            </div>
        `;
    }
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
}

// Color comparison for pixel clicks
async function loadColorComparison(x, y) {
    try {
        const response = await fetch(`/api/color_comparison/${currentImage}?x=${x}&y=${y}`);
        const data = await response.json();
        
        if (data.error) {
            return;
        }
        
        displayColorComparison(data);
    } catch (error) {
        console.error('Error loading color comparison:', error);
    }
}

function displayColorComparison(data) {
    const container = document.getElementById('colorComparison');
    let html = '<h6>مقارنة فضاءات الألوان</h6>';
    
    Object.entries(data.color_spaces).forEach(([space, values]) => {
        html += `
            <div class="mb-2">
                <strong>${space}:</strong> 
                ${Array.isArray(values) ? values.map(v => Math.round(v)).join(', ') : Math.round(values)}
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Pixel matrix for digital analysis
async function loadPixelMatrix(x, y) {
    try {
        const response = await fetch(`/api/pixel_matrix/${currentImage}?x=${x}&y=${y}&size=5`);
        const data = await response.json();
        
        if (data.error) {
            return;
        }
        
        displayPixelMatrix(data);
    } catch (error) {
        console.error('Error loading pixel matrix:', error);
    }
}

function displayPixelMatrix(data) {
    const container = document.getElementById('channelAnalysis');
    container.innerHTML = `
        <h6>مصفوفة البكسل المحيطة</h6>
        <div class="mt-2">
            <small class="text-muted">مصفوفة ${data.matrix_size}x${data.matrix_size} حول النقطة (${data.center_x}, ${data.center_y})</small>
        </div>
    `;
}