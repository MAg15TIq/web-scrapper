// WebSocket functionality for real-time updates

class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.isConnecting = false;
        this.subscribers = new Map();
        this.heartbeatInterval = null;
    }

    connect() {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.isConnecting = true;
        console.log('Connecting to WebSocket...');

        try {
            // Add client ID to WebSocket URL for better connection management
            const clientId = 'dashboard_' + Date.now();
            const wsUrl = CONFIG.WS_URL + '/' + clientId;

            this.ws = new WebSocket(wsUrl);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.handleConnectionError();
        }
    }

    setupEventHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            globalState.isConnected = true;
            
            // Start heartbeat
            this.startHeartbeat();
            
            // Subscribe to default channels
            this.subscribe('system_status');
            this.subscribe('job_updates');
            this.subscribe('agent_updates');
            
            // Notify subscribers
            this.notifySubscribers('connection', { status: 'connected' });
            
            // Update UI
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnecting = false;
            globalState.isConnected = false;

            // Stop heartbeat
            this.stopHeartbeat();

            // Update UI
            this.updateConnectionStatus(false);

            // Notify subscribers
            this.notifySubscribers('connection', { status: 'disconnected' });

            // Show user-friendly message for common disconnect reasons
            if (event.code === 1006) {
                console.warn('WebSocket connection lost - backend may be offline');
                showAlert('Real-time updates unavailable. Backend services may be starting up.', 'info', 5000);
            }

            // Attempt to reconnect
            if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.scheduleReconnect();
            } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                showAlert('Unable to establish real-time connection. Using static data mode.', 'warning', 10000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError();
        };
    }

    handleMessage(data) {
        const { type, channel, payload } = data;
        
        switch (type) {
            case 'job_update':
                this.handleJobUpdate(payload);
                break;
            case 'agent_update':
                this.handleAgentUpdate(payload);
                break;
            case 'system_metrics':
                this.handleSystemMetrics(payload);
                break;
            case 'notification':
                this.handleNotification(payload);
                break;
            case 'pong':
                // Heartbeat response
                break;
            default:
                console.log('Unknown message type:', type);
        }
        
        // Notify channel subscribers
        if (channel) {
            this.notifySubscribers(channel, payload);
        }
    }

    handleJobUpdate(payload) {
        const { job_id, status, progress, result, error } = payload;
        
        // Update global state
        if (status === 'running') {
            globalState.activeJobs++;
        } else if (status === 'completed' || status === 'failed' || status === 'cancelled') {
            globalState.activeJobs = Math.max(0, globalState.activeJobs - 1);
        }
        
        // Update UI elements
        updateStatusIndicators();
        
        // Show notification for important status changes
        if (status === 'completed') {
            showAlert(`Job ${job_id} completed successfully`, 'success');
        } else if (status === 'failed') {
            showAlert(`Job ${job_id} failed: ${error || 'Unknown error'}`, 'danger');
        }
        
        // Trigger custom event for job pages
        document.dispatchEvent(new CustomEvent('jobUpdate', { detail: payload }));
    }

    handleAgentUpdate(payload) {
        const { agent_id, status, metrics } = payload;
        
        // Update global state
        updateStatusIndicators();
        
        // Trigger custom event for agent pages
        document.dispatchEvent(new CustomEvent('agentUpdate', { detail: payload }));
    }

    handleSystemMetrics(payload) {
        // Update global state with system metrics
        globalState.systemMetrics = payload;
        
        // Trigger custom event for monitoring pages
        document.dispatchEvent(new CustomEvent('systemMetrics', { detail: payload }));
    }

    handleNotification(payload) {
        const { message, level, title } = payload;
        showAlert(message, level || 'info');
    }

    subscribe(channel, callback = null) {
        if (!this.subscribers.has(channel)) {
            this.subscribers.set(channel, new Set());
        }
        
        if (callback) {
            this.subscribers.get(channel).add(callback);
        }
        
        // Send subscription message
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.send({
                type: 'subscribe',
                channel: channel
            });
        }
    }

    unsubscribe(channel, callback = null) {
        if (!this.subscribers.has(channel)) {
            return;
        }
        
        if (callback) {
            this.subscribers.get(channel).delete(callback);
        } else {
            this.subscribers.delete(channel);
        }
        
        // Send unsubscription message
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.send({
                type: 'unsubscribe',
                channel: channel
            });
        }
    }

    notifySubscribers(channel, data) {
        if (this.subscribers.has(channel)) {
            this.subscribers.get(channel).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('Subscriber callback error:', error);
                }
            });
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            this.send({ type: 'ping' });
        }, 30000); // Send ping every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            this.connect();
        }, delay);
    }

    handleConnectionError() {
        this.isConnecting = false;
        globalState.isConnected = false;
        this.updateConnectionStatus(false);
    }

    updateConnectionStatus(connected) {
        const statusElements = document.querySelectorAll('.connection-status');
        statusElements.forEach(element => {
            if (connected) {
                element.classList.remove('text-danger');
                element.classList.add('text-success');
                element.innerHTML = '<i class="bi bi-wifi"></i> Connected';
            } else {
                element.classList.remove('text-success');
                element.classList.add('text-danger');
                element.innerHTML = '<i class="bi bi-wifi-off"></i> Disconnected';
            }
        });
    }

    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'User initiated disconnect');
        }
        this.stopHeartbeat();
    }
}

// Global WebSocket manager instance
let wsManager = null;

function initializeWebSocket() {
    if (!wsManager) {
        wsManager = new WebSocketManager();
    }
    wsManager.connect();
}

function getWebSocketManager() {
    return wsManager;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (wsManager) {
        wsManager.disconnect();
    }
});
