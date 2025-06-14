/**
 * Visual Workflow Builder for Web Scraper
 * Drag-and-drop interface for creating scraping workflows
 */

class WorkflowBuilder {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = null;
        this.ctx = null;
        this.nodes = new Map();
        this.connections = [];
        this.selectedNode = null;
        this.draggedNode = null;
        this.dragOffset = { x: 0, y: 0 };
        this.isConnecting = false;
        this.connectionStart = null;
        this.nodeIdCounter = 0;
        this.scale = 1;
        this.panOffset = { x: 0, y: 0 };
        
        this.init();
    }
    
    init() {
        this.createCanvas();
        this.createToolbar();
        this.createNodePalette();
        this.createPropertiesPanel();
        this.setupEventListeners();
        this.loadAgentTypes();
    }
    
    createCanvas() {
        // Create canvas container
        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'workflow-canvas-container';
        canvasContainer.style.cssText = `
            position: relative;
            width: 100%;
            height: 600px;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        `;
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1200;
        this.canvas.height = 600;
        this.canvas.style.cssText = `
            cursor: grab;
            display: block;
        `;
        
        this.ctx = this.canvas.getContext('2d');
        canvasContainer.appendChild(this.canvas);
        this.container.appendChild(canvasContainer);
        
        // Initial draw
        this.draw();
    }
    
    createToolbar() {
        const toolbar = document.createElement('div');
        toolbar.className = 'workflow-toolbar';
        toolbar.style.cssText = `
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        `;
        
        const buttons = [
            { text: '🔄 Reset', action: () => this.resetWorkflow() },
            { text: '💾 Save', action: () => this.saveWorkflow() },
            { text: '📁 Load', action: () => this.loadWorkflow() },
            { text: '▶️ Execute', action: () => this.executeWorkflow() },
            { text: '🔍 Zoom In', action: () => this.zoomIn() },
            { text: '🔍 Zoom Out', action: () => this.zoomOut() },
            { text: '🎯 Fit to Screen', action: () => this.fitToScreen() }
        ];
        
        buttons.forEach(btn => {
            const button = document.createElement('button');
            button.className = 'btn btn-sm btn-outline-primary';
            button.textContent = btn.text;
            button.onclick = btn.action;
            toolbar.appendChild(button);
        });
        
        this.container.appendChild(toolbar);
    }
    
    createNodePalette() {
        const palette = document.createElement('div');
        palette.className = 'node-palette';
        palette.style.cssText = `
            position: absolute;
            left: 10px;
            top: 80px;
            width: 200px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        `;
        
        const title = document.createElement('h6');
        title.textContent = '🧩 Agent Nodes';
        title.style.marginBottom = '15px';
        palette.appendChild(title);
        
        // Agent node types
        const agentTypes = [
            { name: 'Scraper', icon: '🕷️', color: '#007bff' },
            { name: 'Parser', icon: '📝', color: '#28a745' },
            { name: 'Storage', icon: '💾', color: '#6f42c1' },
            { name: 'JavaScript', icon: '⚡', color: '#ffc107' },
            { name: 'Authentication', icon: '🔐', color: '#dc3545' },
            { name: 'Anti-Detection', icon: '🥷', color: '#fd7e14' },
            { name: 'Data Transform', icon: '🔄', color: '#20c997' },
            { name: 'Quality Check', icon: '✅', color: '#17a2b8' }
        ];
        
        agentTypes.forEach(agent => {
            const nodeItem = document.createElement('div');
            nodeItem.className = 'node-palette-item';
            nodeItem.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px;
                margin-bottom: 5px;
                border: 1px solid #eee;
                border-radius: 4px;
                cursor: grab;
                transition: all 0.2s;
            `;
            
            nodeItem.innerHTML = `
                <span style="font-size: 16px;">${agent.icon}</span>
                <span style="font-size: 12px; font-weight: 500;">${agent.name}</span>
            `;
            
            nodeItem.addEventListener('mouseenter', () => {
                nodeItem.style.backgroundColor = '#f8f9fa';
                nodeItem.style.borderColor = agent.color;
            });
            
            nodeItem.addEventListener('mouseleave', () => {
                nodeItem.style.backgroundColor = 'white';
                nodeItem.style.borderColor = '#eee';
            });
            
            nodeItem.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', JSON.stringify(agent));
            });
            
            nodeItem.draggable = true;
            palette.appendChild(nodeItem);
        });
        
        this.container.appendChild(palette);
    }
    
    createPropertiesPanel() {
        const panel = document.createElement('div');
        panel.className = 'properties-panel';
        panel.style.cssText = `
            position: absolute;
            right: 10px;
            top: 80px;
            width: 250px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        `;
        
        panel.innerHTML = `
            <h6>⚙️ Node Properties</h6>
            <div id="node-properties-content">
                <p class="text-muted">Select a node to edit properties</p>
            </div>
        `;
        
        this.container.appendChild(panel);
        this.propertiesPanel = panel;
    }
    
    setupEventListeners() {
        // Canvas events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));
        
        // Drag and drop
        this.canvas.addEventListener('dragover', (e) => e.preventDefault());
        this.canvas.addEventListener('drop', this.onDrop.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.onKeyDown.bind(this));
    }
    
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.panOffset.x;
        const y = (e.clientY - rect.top) / this.scale - this.panOffset.y;
        
        const clickedNode = this.getNodeAt(x, y);
        
        if (clickedNode) {
            this.selectedNode = clickedNode;
            this.draggedNode = clickedNode;
            this.dragOffset = {
                x: x - clickedNode.x,
                y: y - clickedNode.y
            };
            this.updatePropertiesPanel(clickedNode);
        } else {
            this.selectedNode = null;
            this.updatePropertiesPanel(null);
        }
        
        this.draw();
    }
    
    onMouseMove(e) {
        if (this.draggedNode) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / this.scale - this.panOffset.x;
            const y = (e.clientY - rect.top) / this.scale - this.panOffset.y;
            
            this.draggedNode.x = x - this.dragOffset.x;
            this.draggedNode.y = y - this.dragOffset.y;
            
            this.draw();
        }
    }
    
    onMouseUp(e) {
        this.draggedNode = null;
    }
    
    onWheel(e) {
        e.preventDefault();
        const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
        this.scale = Math.max(0.1, Math.min(3, this.scale * scaleFactor));
        this.draw();
    }
    
    onDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.panOffset.x;
        const y = (e.clientY - rect.top) / this.scale - this.panOffset.y;
        
        const clickedNode = this.getNodeAt(x, y);
        if (clickedNode) {
            this.editNodeProperties(clickedNode);
        }
    }
    
    onKeyDown(e) {
        if (e.key === 'Delete' && this.selectedNode) {
            this.deleteNode(this.selectedNode.id);
        }
    }
    
    onDrop(e) {
        e.preventDefault();
        const agentData = JSON.parse(e.dataTransfer.getData('text/plain'));
        
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.panOffset.x;
        const y = (e.clientY - rect.top) / this.scale - this.panOffset.y;
        
        this.addNode(agentData, x, y);
    }
    
    addNode(agentData, x, y) {
        const node = {
            id: ++this.nodeIdCounter,
            type: agentData.name,
            icon: agentData.icon,
            color: agentData.color,
            x: x,
            y: y,
            width: 120,
            height: 60,
            inputs: [],
            outputs: [],
            properties: this.getDefaultProperties(agentData.name)
        };
        
        this.nodes.set(node.id, node);
        this.draw();
    }
    
    getDefaultProperties(agentType) {
        const defaults = {
            'Scraper': { url: '', headers: {}, timeout: 30 },
            'Parser': { selectors: {}, format: 'json' },
            'Storage': { format: 'json', filename: 'output' },
            'JavaScript': { enabled: true, timeout: 10 },
            'Authentication': { type: 'none', credentials: {} },
            'Anti-Detection': { enabled: true, delay: 1 },
            'Data Transform': { operations: [] },
            'Quality Check': { rules: [] }
        };
        
        return defaults[agentType] || {};
    }
    
    draw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Apply transformations
        this.ctx.save();
        this.ctx.scale(this.scale, this.scale);
        this.ctx.translate(this.panOffset.x, this.panOffset.y);
        
        // Draw grid
        this.drawGrid();
        
        // Draw connections
        this.drawConnections();
        
        // Draw nodes
        this.drawNodes();
        
        this.ctx.restore();
    }
    
    drawGrid() {
        const gridSize = 20;
        this.ctx.strokeStyle = '#f0f0f0';
        this.ctx.lineWidth = 1;
        
        for (let x = 0; x < this.canvas.width / this.scale; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height / this.scale);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height / this.scale; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width / this.scale, y);
            this.ctx.stroke();
        }
    }
    
    drawNodes() {
        this.nodes.forEach(node => {
            // Node background
            this.ctx.fillStyle = node === this.selectedNode ? '#e3f2fd' : 'white';
            this.ctx.strokeStyle = node === this.selectedNode ? node.color : '#ddd';
            this.ctx.lineWidth = node === this.selectedNode ? 3 : 1;
            
            this.ctx.fillRect(node.x, node.y, node.width, node.height);
            this.ctx.strokeRect(node.x, node.y, node.width, node.height);
            
            // Node icon and text
            this.ctx.fillStyle = 'black';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(node.icon, node.x + 20, node.y + 25);
            
            this.ctx.font = '12px Arial';
            this.ctx.fillText(node.type, node.x + node.width/2, node.y + 45);
            
            // Connection points
            this.drawConnectionPoints(node);
        });
    }
    
    drawConnectionPoints(node) {
        this.ctx.fillStyle = '#007bff';
        
        // Input point (left)
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y + node.height/2, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Output point (right)
        this.ctx.beginPath();
        this.ctx.arc(node.x + node.width, node.y + node.height/2, 5, 0, 2 * Math.PI);
        this.ctx.fill();
    }
    
    drawConnections() {
        this.ctx.strokeStyle = '#007bff';
        this.ctx.lineWidth = 2;
        
        this.connections.forEach(conn => {
            const fromNode = this.nodes.get(conn.from);
            const toNode = this.nodes.get(conn.to);
            
            if (fromNode && toNode) {
                this.ctx.beginPath();
                this.ctx.moveTo(fromNode.x + fromNode.width, fromNode.y + fromNode.height/2);
                this.ctx.lineTo(toNode.x, toNode.y + toNode.height/2);
                this.ctx.stroke();
            }
        });
    }
    
    getNodeAt(x, y) {
        for (let [id, node] of this.nodes) {
            if (x >= node.x && x <= node.x + node.width &&
                y >= node.y && y <= node.y + node.height) {
                return node;
            }
        }
        return null;
    }
    
    updatePropertiesPanel(node) {
        const content = document.getElementById('node-properties-content');
        
        if (!node) {
            content.innerHTML = '<p class="text-muted">Select a node to edit properties</p>';
            return;
        }
        
        content.innerHTML = `
            <div class="mb-3">
                <label class="form-label">Node Type</label>
                <input type="text" class="form-control" value="${node.type}" readonly>
            </div>
            <div class="mb-3">
                <label class="form-label">Properties</label>
                <textarea class="form-control" rows="8" id="node-properties-json">${JSON.stringify(node.properties, null, 2)}</textarea>
            </div>
            <button class="btn btn-primary btn-sm" onclick="workflowBuilder.saveNodeProperties(${node.id})">Save</button>
        `;
    }
    
    saveNodeProperties(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        try {
            const propertiesText = document.getElementById('node-properties-json').value;
            node.properties = JSON.parse(propertiesText);
            console.log('Node properties saved:', node.properties);
        } catch (e) {
            alert('Invalid JSON in properties');
        }
    }
    
    // Workflow management methods
    resetWorkflow() {
        this.nodes.clear();
        this.connections = [];
        this.selectedNode = null;
        this.nodeIdCounter = 0;
        this.updatePropertiesPanel(null);
        this.draw();
    }
    
    saveWorkflow() {
        const workflow = {
            nodes: Array.from(this.nodes.values()),
            connections: this.connections,
            metadata: {
                created: new Date().toISOString(),
                version: '1.0'
            }
        };
        
        const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'workflow.json';
        a.click();
        URL.revokeObjectURL(url);
    }
    
    loadWorkflow() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const workflow = JSON.parse(e.target.result);
                        this.loadWorkflowData(workflow);
                    } catch (err) {
                        alert('Invalid workflow file');
                    }
                };
                reader.readAsText(file);
            }
        };
        input.click();
    }
    
    loadWorkflowData(workflow) {
        this.resetWorkflow();
        
        // Load nodes
        workflow.nodes.forEach(nodeData => {
            const node = { ...nodeData };
            this.nodes.set(node.id, node);
            this.nodeIdCounter = Math.max(this.nodeIdCounter, node.id);
        });
        
        // Load connections
        this.connections = workflow.connections || [];
        
        this.draw();
    }
    
    async executeWorkflow() {
        const workflow = {
            nodes: Array.from(this.nodes.values()),
            connections: this.connections
        };
        
        try {
            const response = await fetch('/api/v1/workflows/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(workflow)
            });
            
            const result = await response.json();
            console.log('Workflow execution result:', result);
            alert('Workflow executed successfully!');
        } catch (error) {
            console.error('Workflow execution failed:', error);
            alert('Workflow execution failed: ' + error.message);
        }
    }
    
    zoomIn() {
        this.scale = Math.min(3, this.scale * 1.2);
        this.draw();
    }
    
    zoomOut() {
        this.scale = Math.max(0.1, this.scale / 1.2);
        this.draw();
    }
    
    fitToScreen() {
        this.scale = 1;
        this.panOffset = { x: 0, y: 0 };
        this.draw();
    }
    
    deleteNode(nodeId) {
        this.nodes.delete(nodeId);
        this.connections = this.connections.filter(conn => 
            conn.from !== nodeId && conn.to !== nodeId
        );
        this.selectedNode = null;
        this.updatePropertiesPanel(null);
        this.draw();
    }
    
    async loadAgentTypes() {
        try {
            const response = await fetch('/api/v1/agents');
            const agents = await response.json();
            console.log('Available agents:', agents);
        } catch (error) {
            console.error('Failed to load agent types:', error);
        }
    }
}

// Global instance
let workflowBuilder = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('workflow-builder-container')) {
        workflowBuilder = new WorkflowBuilder('workflow-builder-container');
    }
});
