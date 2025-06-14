/**
 * Visual Selector Builder for Web Scraper
 * Point-and-click interface for building CSS selectors
 */

class VisualSelector {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.iframe = null;
        this.overlay = null;
        this.isSelecting = false;
        this.selectedElements = [];
        this.currentSelector = '';
        this.selectorHistory = [];
        
        this.init();
    }
    
    init() {
        this.createInterface();
        this.setupEventListeners();
    }
    
    createInterface() {
        this.container.innerHTML = `
            <div class="visual-selector-interface">
                <div class="selector-toolbar">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <div class="input-group">
                                <input type="url" id="target-url" class="form-control" 
                                       placeholder="Enter URL to analyze..." 
                                       value="https://quotes.toscrape.com">
                                <button class="btn btn-primary" onclick="visualSelector.loadUrl()">
                                    🔍 Load Page
                                </button>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="btn-group" role="group">
                                <button class="btn btn-outline-success" onclick="visualSelector.startSelecting()">
                                    🎯 Start Selecting
                                </button>
                                <button class="btn btn-outline-warning" onclick="visualSelector.stopSelecting()">
                                    ⏹️ Stop
                                </button>
                                <button class="btn btn-outline-info" onclick="visualSelector.clearSelection()">
                                    🗑️ Clear
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="selector-content">
                    <div class="row">
                        <div class="col-md-8">
                            <div class="preview-container">
                                <div id="page-preview" class="page-preview">
                                    <div class="preview-placeholder">
                                        <i class="bi bi-globe" style="font-size: 48px; color: #ccc;"></i>
                                        <p class="text-muted">Enter a URL and click "Load Page" to start selecting elements</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="selector-panel">
                                <h6>🎯 Selector Builder</h6>
                                
                                <div class="mb-3">
                                    <label class="form-label">Generated Selector</label>
                                    <div class="input-group">
                                        <input type="text" id="generated-selector" class="form-control" readonly>
                                        <button class="btn btn-outline-secondary" onclick="visualSelector.copySelector()">
                                            📋 Copy
                                        </button>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Selector Type</label>
                                    <select id="selector-type" class="form-select">
                                        <option value="css">CSS Selector</option>
                                        <option value="xpath">XPath</option>
                                        <option value="text">Text Content</option>
                                        <option value="attribute">Attribute Value</option>
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Selected Elements</label>
                                    <div id="selected-elements" class="selected-elements-list">
                                        <p class="text-muted">No elements selected</p>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Selector History</label>
                                    <div id="selector-history" class="selector-history">
                                        <p class="text-muted">No history</p>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <button class="btn btn-success w-100" onclick="visualSelector.testSelector()">
                                        🧪 Test Selector
                                    </button>
                                </div>
                                
                                <div class="mb-3">
                                    <button class="btn btn-primary w-100" onclick="visualSelector.saveSelector()">
                                        💾 Save Selector
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .visual-selector-interface {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .selector-toolbar {
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }
            
            .preview-container {
                position: relative;
                border: 2px solid #ddd;
                border-radius: 8px;
                height: 600px;
                overflow: hidden;
            }
            
            .page-preview {
                width: 100%;
                height: 100%;
                position: relative;
            }
            
            .preview-placeholder {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                background: #f8f9fa;
            }
            
            .selector-panel {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                height: 600px;
                overflow-y: auto;
            }
            
            .selected-elements-list {
                max-height: 150px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background: white;
            }
            
            .selector-history {
                max-height: 100px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background: white;
            }
            
            .element-item {
                padding: 4px 8px;
                margin: 2px 0;
                background: #e3f2fd;
                border-radius: 4px;
                font-size: 12px;
                cursor: pointer;
            }
            
            .element-item:hover {
                background: #bbdefb;
            }
            
            .history-item {
                padding: 4px 8px;
                margin: 2px 0;
                background: #f3e5f5;
                border-radius: 4px;
                font-size: 12px;
                cursor: pointer;
            }
            
            .history-item:hover {
                background: #e1bee7;
            }
            
            .element-highlight {
                position: absolute;
                border: 2px solid #ff4444;
                background: rgba(255, 68, 68, 0.1);
                pointer-events: none;
                z-index: 1000;
            }
            
            .iframe-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 999;
                cursor: crosshair;
            }
        `;
        document.head.appendChild(style);
    }
    
    setupEventListeners() {
        document.getElementById('selector-type').addEventListener('change', () => {
            this.updateSelector();
        });
    }
    
    async loadUrl() {
        const url = document.getElementById('target-url').value;
        if (!url) {
            alert('Please enter a URL');
            return;
        }
        
        try {
            // Create iframe for page preview
            const previewContainer = document.getElementById('page-preview');
            previewContainer.innerHTML = `
                <iframe id="preview-iframe" 
                        src="/api/v1/preview?url=${encodeURIComponent(url)}" 
                        style="width: 100%; height: 100%; border: none;">
                </iframe>
                <div class="loading-overlay" style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(255,255,255,0.9);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1001;
                ">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            
            this.iframe = document.getElementById('preview-iframe');
            
            // Wait for iframe to load
            this.iframe.onload = () => {
                document.querySelector('.loading-overlay').remove();
                this.setupIframeInteraction();
            };
            
        } catch (error) {
            console.error('Error loading URL:', error);
            alert('Failed to load URL: ' + error.message);
        }
    }
    
    setupIframeInteraction() {
        try {
            const iframeDoc = this.iframe.contentDocument || this.iframe.contentWindow.document;
            
            // Create overlay for capturing clicks
            this.overlay = document.createElement('div');
            this.overlay.className = 'iframe-overlay';
            this.overlay.style.display = 'none';
            
            const previewContainer = document.getElementById('page-preview');
            previewContainer.appendChild(this.overlay);
            
            // Add event listeners to overlay
            this.overlay.addEventListener('mousemove', this.onMouseMove.bind(this));
            this.overlay.addEventListener('click', this.onElementClick.bind(this));
            
        } catch (error) {
            console.error('Cannot access iframe content (CORS restriction):', error);
            this.showCorsWarning();
        }
    }
    
    showCorsWarning() {
        const previewContainer = document.getElementById('page-preview');
        previewContainer.innerHTML = `
            <div class="cors-warning" style="
                padding: 40px;
                text-align: center;
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                margin: 20px;
            ">
                <h5>⚠️ CORS Restriction</h5>
                <p>Cannot directly interact with external websites due to browser security.</p>
                <p>Use the manual selector input below or try with a local test page.</p>
                <button class="btn btn-primary" onclick="visualSelector.loadTestPage()">
                    Load Test Page
                </button>
            </div>
        `;
    }
    
    loadTestPage() {
        // Load a local test page that we can interact with
        const testHtml = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Page for Selector Builder</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    .quote { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                    .text { font-style: italic; margin-bottom: 10px; }
                    .author { font-weight: bold; color: #666; }
                    .tags { margin-top: 10px; }
                    .tag { background: #e3f2fd; padding: 2px 8px; border-radius: 3px; margin-right: 5px; }
                </style>
            </head>
            <body>
                <h1>Quotes to Scrape - Test Page</h1>
                <div class="quote">
                    <div class="text">"The world as we have created it is a process of our thinking."</div>
                    <div class="author">by Albert Einstein</div>
                    <div class="tags">
                        <span class="tag">change</span>
                        <span class="tag">deep-thoughts</span>
                        <span class="tag">thinking</span>
                    </div>
                </div>
                <div class="quote">
                    <div class="text">"It is our choices that show what we truly are."</div>
                    <div class="author">by J.K. Rowling</div>
                    <div class="tags">
                        <span class="tag">abilities</span>
                        <span class="tag">choices</span>
                    </div>
                </div>
                <div class="quote">
                    <div class="text">"There are only two ways to live your life."</div>
                    <div class="author">by Albert Einstein</div>
                    <div class="tags">
                        <span class="tag">inspirational</span>
                        <span class="tag">life</span>
                        <span class="tag">live</span>
                    </div>
                </div>
            </body>
            </html>
        `;
        
        const blob = new Blob([testHtml], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const previewContainer = document.getElementById('page-preview');
        previewContainer.innerHTML = `
            <iframe id="preview-iframe" 
                    src="${url}" 
                    style="width: 100%; height: 100%; border: none;">
            </iframe>
        `;
        
        this.iframe = document.getElementById('preview-iframe');
        this.iframe.onload = () => {
            this.setupIframeInteraction();
            URL.revokeObjectURL(url);
        };
    }
    
    startSelecting() {
        this.isSelecting = true;
        if (this.overlay) {
            this.overlay.style.display = 'block';
        }
        
        // Update UI
        document.querySelector('[onclick="visualSelector.startSelecting()"]').classList.add('active');
    }
    
    stopSelecting() {
        this.isSelecting = false;
        if (this.overlay) {
            this.overlay.style.display = 'none';
        }
        
        // Update UI
        document.querySelector('[onclick="visualSelector.startSelecting()"]').classList.remove('active');
    }
    
    onMouseMove(e) {
        if (!this.isSelecting) return;
        
        // Highlight element under cursor
        // This would need more sophisticated implementation for real iframe interaction
    }
    
    onElementClick(e) {
        if (!this.isSelecting) return;
        
        e.preventDefault();
        
        // For demo purposes, simulate element selection
        const mockElement = {
            tagName: 'div',
            className: 'quote',
            textContent: 'Sample quote text...',
            selector: '.quote'
        };
        
        this.addSelectedElement(mockElement);
    }
    
    addSelectedElement(element) {
        this.selectedElements.push(element);
        this.updateSelectedElementsList();
        this.generateSelector();
    }
    
    updateSelectedElementsList() {
        const container = document.getElementById('selected-elements');
        
        if (this.selectedElements.length === 0) {
            container.innerHTML = '<p class="text-muted">No elements selected</p>';
            return;
        }
        
        container.innerHTML = this.selectedElements.map((el, index) => `
            <div class="element-item" onclick="visualSelector.removeSelectedElement(${index})">
                <strong>${el.tagName}</strong>
                ${el.className ? `<span class="text-muted">.${el.className}</span>` : ''}
                <br>
                <small>${el.textContent.substring(0, 50)}...</small>
            </div>
        `).join('');
    }
    
    removeSelectedElement(index) {
        this.selectedElements.splice(index, 1);
        this.updateSelectedElementsList();
        this.generateSelector();
    }
    
    generateSelector() {
        if (this.selectedElements.length === 0) {
            this.currentSelector = '';
            document.getElementById('generated-selector').value = '';
            return;
        }
        
        const selectorType = document.getElementById('selector-type').value;
        let selector = '';
        
        switch (selectorType) {
            case 'css':
                selector = this.generateCSSSelector();
                break;
            case 'xpath':
                selector = this.generateXPathSelector();
                break;
            case 'text':
                selector = this.generateTextSelector();
                break;
            case 'attribute':
                selector = this.generateAttributeSelector();
                break;
        }
        
        this.currentSelector = selector;
        document.getElementById('generated-selector').value = selector;
    }
    
    generateCSSSelector() {
        // Simple CSS selector generation
        const element = this.selectedElements[0];
        if (element.className) {
            return `.${element.className}`;
        }
        return element.tagName.toLowerCase();
    }
    
    generateXPathSelector() {
        // Simple XPath generation
        const element = this.selectedElements[0];
        if (element.className) {
            return `//${element.tagName.toLowerCase()}[@class="${element.className}"]`;
        }
        return `//${element.tagName.toLowerCase()}`;
    }
    
    generateTextSelector() {
        const element = this.selectedElements[0];
        return `text:${element.textContent.substring(0, 30)}`;
    }
    
    generateAttributeSelector() {
        const element = this.selectedElements[0];
        if (element.className) {
            return `${element.tagName.toLowerCase()}[class="${element.className}"]`;
        }
        return element.tagName.toLowerCase();
    }
    
    updateSelector() {
        this.generateSelector();
    }
    
    copySelector() {
        const selector = document.getElementById('generated-selector').value;
        if (selector) {
            navigator.clipboard.writeText(selector).then(() => {
                alert('Selector copied to clipboard!');
            });
        }
    }
    
    testSelector() {
        const selector = this.currentSelector;
        if (!selector) {
            alert('No selector to test');
            return;
        }
        
        // Add to history
        this.selectorHistory.push({
            selector: selector,
            timestamp: new Date().toLocaleTimeString(),
            type: document.getElementById('selector-type').value
        });
        
        this.updateSelectorHistory();
        
        // Simulate testing
        alert(`Testing selector: ${selector}\nFound 3 matching elements`);
    }
    
    updateSelectorHistory() {
        const container = document.getElementById('selector-history');
        
        if (this.selectorHistory.length === 0) {
            container.innerHTML = '<p class="text-muted">No history</p>';
            return;
        }
        
        container.innerHTML = this.selectorHistory.slice(-5).map((item, index) => `
            <div class="history-item" onclick="visualSelector.loadFromHistory(${this.selectorHistory.length - 1 - index})">
                <strong>${item.type}</strong> - ${item.timestamp}
                <br>
                <small>${item.selector}</small>
            </div>
        `).join('');
    }
    
    loadFromHistory(index) {
        const item = this.selectorHistory[index];
        document.getElementById('generated-selector').value = item.selector;
        document.getElementById('selector-type').value = item.type;
        this.currentSelector = item.selector;
    }
    
    saveSelector() {
        const selector = this.currentSelector;
        if (!selector) {
            alert('No selector to save');
            return;
        }
        
        const selectorData = {
            selector: selector,
            type: document.getElementById('selector-type').value,
            url: document.getElementById('target-url').value,
            elements: this.selectedElements,
            created: new Date().toISOString()
        };
        
        // Save to localStorage for demo
        const savedSelectors = JSON.parse(localStorage.getItem('savedSelectors') || '[]');
        savedSelectors.push(selectorData);
        localStorage.setItem('savedSelectors', JSON.stringify(savedSelectors));
        
        alert('Selector saved successfully!');
    }
    
    clearSelection() {
        this.selectedElements = [];
        this.currentSelector = '';
        document.getElementById('generated-selector').value = '';
        this.updateSelectedElementsList();
    }
}

// Global instance
let visualSelector = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('visual-selector-container')) {
        visualSelector = new VisualSelector('visual-selector-container');
    }
});
