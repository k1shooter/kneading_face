/**
 * Facial Expression Transformer - Frontend JavaScript
 * Handles drag-and-drop uploads, API communication, and UI interactions
 */

class FacialExpressionApp {
    constructor() {
        this.currentFile = null;
        this.currentConversionId = null;
        this.pollingInterval = null;
        this.apiToken = null;
        
        // Configuration
        this.config = {
            maxFileSize: 10 * 1024 * 1024, // 10MB
            allowedTypes: ['image/jpeg', 'image/png', 'image/webp'],
            pollingInterval: 2000, // 2 seconds
            maxPollingTime: 300000, // 5 minutes
            apiEndpoints: {
                upload: '/upload',
                status: '/status/',
                health: '/api/health',
                transform: '/api/transform',
                history: '/api/history',
                token: '/api/token'
            }
        };
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }
    
    /**
     * Initialize the application
     */
    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.checkApiHealth();
        this.loadConversionHistory();
        
        // Initialize tooltips and UI components
        this.initializeTooltips();
        this.setupExpressionCards();
        
        console.log('Facial Expression App initialized');
    }
    
    /**
     * Set up all event listeners
     */
    setupEventListeners() {
        // File input change
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // Upload button
        const uploadBtn = document.getElementById('upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.triggerFileSelect());
        }
        
        // Transform button
        const transformBtn = document.getElementById('transform-btn');
        if (transformBtn) {
            transformBtn.addEventListener('click', () => this.startTransformation());
        }
        
        // Clear button
        const clearBtn = document.getElementById('clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearUpload());
        }
        
        // Expression selection
        const expressionCards = document.querySelectorAll('.expression-card');
        expressionCards.forEach(card => {
            card.addEventListener('click', () => this.selectExpression(card));
        });
        
        // Settings toggles
        const settingsToggles = document.querySelectorAll('.setting-toggle input');
        settingsToggles.forEach(toggle => {
            toggle.addEventListener('change', () => this.updateSettings());
        });
        
        // History refresh
        const refreshHistoryBtn = document.getElementById('refresh-history');
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', () => this.loadConversionHistory());
        }
        
        // Download result button
        const downloadBtn = document.getElementById('download-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadResult());
        }
        
        // Share result button
        const shareBtn = document.getElementById('share-btn');
        if (shareBtn) {
            shareBtn.addEventListener('click', () => this.shareResult());
        }
    }
    
    /**
     * Set up drag and drop functionality
     */
    setupDragAndDrop() {
        const dropZone = document.getElementById('drop-zone');
        if (!dropZone) return;
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.highlight(dropZone), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.unhighlight(dropZone), false);
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);
    }
    
    /**
     * Prevent default drag behaviors
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    /**
     * Highlight drop zone
     */
    highlight(element) {
        element.classList.add('drag-over');
    }
    
    /**
     * Remove highlight from drop zone
     */
    unhighlight(element) {
        element.classList.remove('drag-over');
    }
    
    /**
     * Handle file drop
     */
    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }
    
    /**
     * Trigger file selection dialog
     */
    triggerFileSelect() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    /**
     * Handle file selection from input
     */
    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }
    
    /**
     * Process selected/dropped file
     */
    handleFile(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }
        
        this.currentFile = file;
        this.displayFilePreview(file);
        this.updateUI('file-selected');
        
        // Auto-select default expression if none selected
        const selectedExpression = document.querySelector('.expression-card.selected');
        if (!selectedExpression) {
            const defaultExpression = document.querySelector('.expression-card[data-expression="happy"]');
            if (defaultExpression) {
                this.selectExpression(defaultExpression);
            }
        }
    }
    
    /**
     * Validate uploaded file
     */
    validateFile(file) {
        // Check file size
        if (file.size > this.config.maxFileSize) {
            this.showToast(`File too large. Maximum size is ${this.formatFileSize(this.config.maxFileSize)}`, 'error');
            return false;
        }
        
        // Check file type
        if (!this.config.allowedTypes.includes(file.type)) {
            this.showToast('Invalid file type. Please upload JPEG, PNG, or WebP images.', 'error');
            return false;
        }
        
        // Check if it's actually an image
        return this.validateImageFile(file);
    }
    
    /**
     * Validate that file is actually an image
     */
    validateImageFile(file) {
        return new Promise((resolve) => {
            const img = new Image();
            const url = URL.createObjectURL(file);
            
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(true);
            };
            
            img.onerror = () => {
                URL.revokeObjectURL(url);
                this.showToast('Invalid image file. Please select a valid image.', 'error');
                resolve(false);
            };
            
            img.src = url;
        });
    }
    
    /**
     * Display file preview
     */
    displayFilePreview(file) {
        const preview = document.getElementById('image-preview');
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        
        if (preview) {
            const url = URL.createObjectURL(file);
            preview.src = url;
            preview.style.display = 'block';
            
            // Clean up previous URL
            if (preview.dataset.url) {
                URL.revokeObjectURL(preview.dataset.url);
            }
            preview.dataset.url = url;
        }
        
        if (fileName) {
            fileName.textContent = file.name;
        }
        
        if (fileSize) {
            fileSize.textContent = this.formatFileSize(file.size);
        }
    }
    
    /**
     * Select expression
     */
    selectExpression(card) {
        // Remove previous selection
        document.querySelectorAll('.expression-card').forEach(c => {
            c.classList.remove('selected');
        });
        
        // Add selection to clicked card
        card.classList.add('selected');
        
        // Update transform button state
        this.updateTransformButton();
        
        // Show expression details
        const expression = card.dataset.expression;
        const description = card.dataset.description;
        this.showExpressionDetails(expression, description);
    }
    
    /**
     * Show expression details
     */
    showExpressionDetails(expression, description) {
        const detailsElement = document.getElementById('expression-details');
        if (detailsElement) {
            detailsElement.innerHTML = `
                <h4>Selected Expression: ${expression.charAt(0).toUpperCase() + expression.slice(1)}</h4>
                <p>${description || 'Transform your image with this expression.'}</p>
            `;
            detailsElement.style.display = 'block';
        }
    }
    
    /**
     * Update transform button state
     */
    updateTransformButton() {
        const transformBtn = document.getElementById('transform-btn');
        const selectedExpression = document.querySelector('.expression-card.selected');
        
        if (transformBtn) {
            const canTransform = this.currentFile && selectedExpression;
            transformBtn.disabled = !canTransform;
            transformBtn.textContent = canTransform ? 'Transform Image' : 'Select Image & Expression';
        }
    }
    
    /**
     * Start image transformation
     */
    async startTransformation() {
        if (!this.currentFile) {
            this.showToast('Please select an image first.', 'warning');
            return;
        }
        
        const selectedExpression = document.querySelector('.expression-card.selected');
        if (!selectedExpression) {
            this.showToast('Please select an expression.', 'warning');
            return;
        }
        
        const expression = selectedExpression.dataset.expression;
        const settings = this.getTransformationSettings();
        
        this.updateUI('processing');
        
        try {
            // Create form data
            const formData = new FormData();
            formData.append('image', this.currentFile);
            formData.append('expression', expression);
            formData.append('settings', JSON.stringify(settings));
            
            // Start transformation
            const response = await fetch(this.config.apiEndpoints.upload, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                this.currentConversionId = result.conversion_id;
                this.startStatusPolling();
                this.showToast('Transformation started! Please wait...', 'info');
            } else {
                throw new Error(result.error || 'Transformation failed');
            }
            
        } catch (error) {
            console.error('Transformation error:', error);
            this.showToast(`Transformation failed: ${error.message}`, 'error');
            this.updateUI('error');
        }
    }
    
    /**
     * Get transformation settings
     */
    getTransformationSettings() {
        const settings = {};
        
        // Get all setting toggles
        const toggles = document.querySelectorAll('.setting-toggle input');
        toggles.forEach(toggle => {
            settings[toggle.name] = toggle.checked;
        });
        
        // Get intensity slider if present
        const intensitySlider = document.getElementById('intensity-slider');
        if (intensitySlider) {
            settings.intensity = parseFloat(intensitySlider.value);
        }
        
        return settings;
    }
    
    /**
     * Start polling for transformation status
     */
    startStatusPolling() {
        if (!this.currentConversionId) return;
        
        let pollCount = 0;
        const maxPolls = this.config.maxPollingTime / this.config.pollingInterval;
        
        this.pollingInterval = setInterval(async () => {
            pollCount++;
            
            try {
                const response = await fetch(`${this.config.apiEndpoints.status}${this.currentConversionId}`);
                const status = await response.json();
                
                this.updateProcessingStatus(status);
                
                if (status.status === 'completed') {
                    this.handleTransformationComplete(status);
                    this.stopStatusPolling();
                } else if (status.status === 'failed' || pollCount >= maxPolls) {
                    this.handleTransformationError(status.error || 'Transformation timed out');
                    this.stopStatusPolling();
                }
                
            } catch (error) {
                console.error('Status polling error:', error);
                if (pollCount >= maxPolls) {
                    this.handleTransformationError('Status check failed');
                    this.stopStatusPolling();
                }
            }
        }, this.config.pollingInterval);
    }
    
    /**
     * Stop status polling
     */
    stopStatusPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    /**
     * Update processing status display
     */
    updateProcessingStatus(status) {
        const statusElement = document.getElementById('processing-status');
        const progressBar = document.getElementById('progress-bar');
        const statusText = document.getElementById('status-text');
        
        if (statusElement) {
            statusElement.style.display = 'block';
        }
        
        if (progressBar) {
            const progress = status.progress || 0;
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }
        
        if (statusText) {
            statusText.textContent = status.message || 'Processing...';
        }
    }
    
    /**
     * Handle transformation completion
     */
    handleTransformationComplete(status) {
        this.updateUI('completed');
        this.displayResults(status);
        this.showToast('Transformation completed successfully!', 'success');
        this.loadConversionHistory(); // Refresh history
    }
    
    /**
     * Handle transformation error
     */
    handleTransformationError(error) {
        this.updateUI('error');
        this.showToast(`Transformation failed: ${error}`, 'error');
    }
    
    /**
     * Display transformation results
     */
    displayResults(status) {
        const resultsSection = document.getElementById('results-section');
        const originalImage = document.getElementById('original-image');
        const transformedImage = document.getElementById('transformed-image');
        
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        if (originalImage && this.currentFile) {
            originalImage.src = URL.createObjectURL(this.currentFile);
        }
        
        if (transformedImage && status.result_url) {
            transformedImage.src = status.result_url;
            transformedImage.dataset.downloadUrl = status.download_url || status.result_url;
        }
        
        // Update metadata
        this.updateResultMetadata(status);
    }
    
    /**
     * Update result metadata
     */
    updateResultMetadata(status) {
        const metadata = document.getElementById('result-metadata');
        if (metadata && status.metadata) {
            const meta = status.metadata;
            metadata.innerHTML = `
                <div class="metadata-item">
                    <span class="label">Expression:</span>
                    <span class="value">${meta.expression || 'Unknown'}</span>
                </div>
                <div class="metadata-item">
                    <span class="label">Processing Time:</span>
                    <span class="value">${meta.processing_time || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="label">Model Version:</span>
                    <span class="value">${meta.model_version || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="label">Resolution:</span>
                    <span class="value">${meta.resolution || 'N/A'}</span>
                </div>
            `;
        }
    }
    
    /**
     * Download transformation result
     */
    downloadResult() {
        const transformedImage = document.getElementById('transformed-image');
        if (transformedImage && transformedImage.dataset.downloadUrl) {
            const link = document.createElement('a');
            link.href = transformedImage.dataset.downloadUrl;
            link.download = `transformed_${Date.now()}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    
    /**
     * Share transformation result
     */
    shareResult() {
        const transformedImage = document.getElementById('transformed-image');
        if (transformedImage && transformedImage.src) {
            if (navigator.share) {
                // Use Web Share API if available
                navigator.share({
                    title: 'Facial Expression Transformation',
                    text: 'Check out my transformed image!',
                    url: window.location.href
                }).catch(console.error);
            } else {
                // Fallback to copying URL
                navigator.clipboard.writeText(window.location.href).then(() => {
                    this.showToast('Link copied to clipboard!', 'success');
                }).catch(() => {
                    this.showToast('Unable to copy link', 'error');
                });
            }
        }
    }
    
    /**
     * Clear current upload
     */
    clearUpload() {
        this.currentFile = null;
        this.currentConversionId = null;
        this.stopStatusPolling();
        
        // Clear file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Clear preview
        const preview = document.getElementById('image-preview');
        if (preview) {
            if (preview.dataset.url) {
                URL.revokeObjectURL(preview.dataset.url);
            }
            preview.src = '';
            preview.style.display = 'none';
        }
        
        // Clear expression selection
        document.querySelectorAll('.expression-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        this.updateUI('initial');
        this.showToast('Upload cleared', 'info');
    }
    
    /**
     * Update UI state
     */
    updateUI(state) {
        const elements = {
            uploadSection: document.getElementById('upload-section'),
            processingSection: document.getElementById('processing-section'),
            resultsSection: document.getElementById('results-section'),
            transformBtn: document.getElementById('transform-btn'),
            clearBtn: document.getElementById('clear-btn')
        };
        
        // Hide all sections first
        Object.values(elements).forEach(el => {
            if (el) el.style.display = 'none';
        });
        
        switch (state) {
            case 'initial':
                if (elements.uploadSection) elements.uploadSection.style.display = 'block';
                break;
                
            case 'file-selected':
                if (elements.uploadSection) elements.uploadSection.style.display = 'block';
                this.updateTransformButton();
                break;
                
            case 'processing':
                if (elements.processingSection) elements.processingSection.style.display = 'block';
                break;
                
            case 'completed':
                if (elements.resultsSection) elements.resultsSection.style.display = 'block';
                if (elements.clearBtn) elements.clearBtn.style.display = 'inline-block';
                break;
                
            case 'error':
                if (elements.uploadSection) elements.uploadSection.style.display = 'block';
                this.updateTransformButton();
                break;
        }
    }
    
    /**
     * Load conversion history
     */
    async loadConversionHistory() {
        try {
            const response = await fetch(this.config.apiEndpoints.history);
            const data = await response.json();
            
            if (data.success) {
                this.displayConversionHistory(data.history);
            }
        } catch (error) {
            console.error('Failed to load conversion history:', error);
        }
    }
    
    /**
     * Display conversion history
     */
    displayConversionHistory(history) {
        const historyContainer = document.getElementById('history-container');
        if (!historyContainer) return;
        
        if (!history || history.length === 0) {
            historyContainer.innerHTML = '<p class="no-history">No conversion history yet.</p>';
            return;
        }
        
        const historyHTML = history.map(item => `
            <div class="history-item" data-id="${item.id}">
                <div class="history-thumbnail">
                    <img src="${item.thumbnail_url}" alt="Conversion ${item.id}" loading="lazy">
                </div>
                <div class="history-details">
                    <div class="history-expression">${item.expression}</div>
                    <div class="history-date">${this.formatDate(item.created_at)}</div>
                    <div class="history-status ${item.status}">${item.status}</div>
                </div>
                <div class="history-actions">
                    <button class="btn-small" onclick="app.viewHistoryItem('${item.id}')">View</button>
                    <button class="btn-small" onclick="app.downloadHistoryItem('${item.id}')">Download</button>
                </div>
            </div>
        `).join('');
        
        historyContainer.innerHTML = historyHTML;
    }
    
    /**
     * View history item
     */
    viewHistoryItem(id) {
        // Implementation for viewing history item
        console.log('Viewing history item:', id);
    }
    
    /**
     * Download history item
     */
    downloadHistoryItem(id) {
        // Implementation for downloading history item
        console.log('Downloading history item:', id);
    }
    
    /**
     * Check API health
     */
    async checkApiHealth() {
        try {
            const response = await fetch(this.config.apiEndpoints.health);
            const health = await response.json();
            
            this.updateHealthStatus(health);
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateHealthStatus({ status: 'error', message: 'API unavailable' });
        }
    }
    
    /**
     * Update health status display
     */
    updateHealthStatus(health) {
        const statusIndicator = document.getElementById('api-status');
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${health.status}`;
            statusIndicator.title = health.message || health.status;
        }
    }
    
    /**
     * Initialize tooltips
     */
    initializeTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip);
            element.addEventListener('mouseleave', this.hideTooltip);
        });
    }
    
    /**
     * Show tooltip
     */
    showTooltip(e) {
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = e.target.dataset.tooltip;
        
        document.body.appendChild(tooltip);
        
        const rect = e.target.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
        
        e.target.tooltipElement = tooltip;
    }
    
    /**
     * Hide tooltip
     */
    hideTooltip(e) {
        if (e.target.tooltipElement) {
            document.body.removeChild(e.target.tooltipElement);
            e.target.tooltipElement = null;
        }
    }
    
    /**
     * Setup expression cards
     */
    setupExpressionCards() {
        const cards = document.querySelectorAll('.expression-card');
        cards.forEach(card => {
            // Add hover effects
            card.addEventListener('mouseenter', () => {
                if (!card.classList.contains('selected')) {
                    card.style.transform = 'translateY(-5px)';
                }
            });
            
            card.addEventListener('mouseleave', () => {
                if (!card.classList.contains('selected')) {
                    card.style.transform = 'translateY(0)';
                }
            });
        });
    }
    
    /**
     * Update settings
     */
    updateSettings() {
        const settings = this.getTransformationSettings();
        console.log('Settings updated:', settings);
        
        // Save settings to localStorage
        localStorage.setItem('expressionAppSettings', JSON.stringify(settings));
    }
    
    /**
     * Load saved settings
     */
    loadSavedSettings() {
        try {
            const saved = localStorage.getItem('expressionAppSettings');
            if (saved) {
                const settings = JSON.parse(saved);
                
                // Apply saved settings to UI
                Object.entries(settings).forEach(([key, value]) => {
                    const element = document.querySelector(`[name="${key}"]`);
                    if (element) {
                        if (element.type === 'checkbox') {
                            element.checked = value;
                        } else {
                            element.value = value;
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Failed to load saved settings:', error);
        }
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        // Remove existing toasts
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());
        
        // Create new toast
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">&times;</button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
        
        // Animate in
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
    }
    
    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    /**
     * Format date
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
}

// Initialize the application
const app = new FacialExpressionApp();

// Export for global access
window.app = app;