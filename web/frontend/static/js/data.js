// Data management page JavaScript functionality

let datasetsData = [];
let exportHistory = [];
let selectedDatasetId = null;
let currentPreviewMode = 'table';
let dataChart = null;

// Initialize data page
document.addEventListener('DOMContentLoaded', function() {
    initializeDataPage();
    setupEventListeners();
    loadDataManagementData();
});

function initializeDataPage() {
    console.log('Initializing data management page...');
    initializeCharts();
}

function setupEventListeners() {
    // Handle URL hash for direct actions
    const hash = window.location.hash;
    if (hash === '#export') {
        setTimeout(() => {
            const modal = new bootstrap.Modal(document.getElementById('exportDataModal'));
            modal.show();
        }, 500);
    }
}

async function loadDataManagementData() {
    try {
        // Load datasets and export history with component-specific loading
        await Promise.allSettled([
            loadDatasets(),
            loadExportHistory()
        ]);

        updateDataOverview();
        updateDatasetsTable();
        updateExportHistoryTable();
        updateDataFormatChart();

    } catch (error) {
        console.error('Failed to load data management data:', error);
        showAlert('Failed to load data management data. Please check if the backend is running.', 'danger', 8000);
    }
}

async function loadDatasets() {
    try {
        // Fetch real datasets from API with component-specific loading
        const response = await apiRequest('/datasets', {}, 'datasets-table-container');
        datasetsData = response.datasets || [];

        // Populate export dataset dropdown
        populateDatasetDropdown();

    } catch (error) {
        console.error('Failed to load datasets:', error);
        datasetsData = [];

        // Show empty state
        const container = document.getElementById('datasets-table-container');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-database-x display-4 text-warning"></i>
                    <h5 class="mt-3 text-muted">Unable to Load Datasets</h5>
                    <p class="text-muted">Please check your connection and try again.</p>
                    <button class="btn btn-primary" onclick="loadDatasets()">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
        }
    }
}

async function loadExportHistory() {
    try {
        // Fetch real export history from API - NO MOCK DATA
        const response = await apiRequest('/exports');
        exportHistory = response.exports || [];
        
    } catch (error) {
        console.error('Failed to load export history:', error);
        exportHistory = [];
    }
}

function updateDataOverview() {
    const totalRecords = datasetsData.reduce((sum, ds) => sum + ds.records_count, 0);
    const totalSize = datasetsData.reduce((sum, ds) => sum + ds.size_bytes, 0);
    const recentExports = exportHistory.filter(exp => 
        exp.created_at > new Date(Date.now() - 86400000)
    ).length;
    
    document.getElementById('total-records').textContent = totalRecords.toLocaleString();
    document.getElementById('total-datasets').textContent = datasetsData.length;
    document.getElementById('data-size').textContent = formatBytes(totalSize);
    document.getElementById('recent-exports').textContent = recentExports;
}

function updateDatasetsTable() {
    const container = document.getElementById('datasets-table-container');
    if (!container) return;
    
    if (datasetsData.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-database display-4 text-muted"></i>
                <p class="text-muted mt-2">No datasets available</p>
                <p class="text-muted">Run some scraping jobs to generate data</p>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'name',
            title: 'Dataset Name',
            render: (value, row) => `
                <div>
                    <strong><a href="#" onclick="showDatasetDetails('${row.id}')">${value}</a></strong>
                    <br>
                    <small class="text-muted">Job: ${row.job_id}</small>
                </div>
            `
        },
        {
            field: 'format',
            title: 'Format',
            render: (value) => `<span class="badge bg-primary">${value.toUpperCase()}</span>`
        },
        {
            field: 'records_count',
            title: 'Records',
            render: (value) => value.toLocaleString()
        },
        {
            field: 'size_bytes',
            title: 'Size',
            render: (value) => formatBytes(value)
        },
        {
            field: 'last_updated',
            title: 'Last Updated',
            render: (value) => formatDateTime(value)
        },
        {
            field: 'actions',
            title: 'Actions',
            render: (value, row) => `
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="previewDataset('${row.id}')">
                        <i class="bi bi-eye"></i>
                    </button>
                    <button class="btn btn-outline-success" onclick="exportDataset('${row.id}')">
                        <i class="bi bi-download"></i>
                    </button>
                    <button class="btn btn-outline-danger" onclick="deleteDataset('${row.id}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            `
        }
    ];
    
    createDataTable('datasets-table-container', datasetsData, columns);
}

function updateExportHistoryTable() {
    const container = document.getElementById('export-history-container');
    if (!container) return;
    
    if (exportHistory.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-clock-history display-4 text-muted"></i>
                <p class="text-muted mt-2">No export history</p>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'dataset_name',
            title: 'Dataset',
            render: (value, row) => `
                <div>
                    <strong>${value}</strong>
                    <br>
                    <small class="text-muted">${row.filename}</small>
                </div>
            `
        },
        {
            field: 'format',
            title: 'Format',
            render: (value) => `<span class="badge bg-secondary">${value.toUpperCase()}</span>`
        },
        {
            field: 'records_count',
            title: 'Records',
            render: (value) => value.toLocaleString()
        },
        {
            field: 'file_size',
            title: 'File Size',
            render: (value) => formatBytes(value)
        },
        {
            field: 'created_at',
            title: 'Exported',
            render: (value) => formatDateTime(value)
        },
        {
            field: 'status',
            title: 'Status',
            render: (value) => {
                const statusClass = value === 'completed' ? 'success' : 
                                  value === 'failed' ? 'danger' : 'warning';
                return `<span class="badge bg-${statusClass}">${value.toUpperCase()}</span>`;
            }
        },
        {
            field: 'actions',
            title: 'Actions',
            render: (value, row) => `
                <div class="btn-group btn-group-sm">
                    ${row.status === 'completed' ? 
                        `<a href="${row.download_url}" class="btn btn-outline-primary" download>
                            <i class="bi bi-download"></i>
                        </a>` : ''
                    }
                    <button class="btn btn-outline-danger" onclick="deleteExport('${row.id}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            `
        }
    ];
    
    createDataTable('export-history-container', exportHistory, columns);
}

function initializeCharts() {
    // Data Format Distribution Chart
    dataChart = createChart('dataFormatChart', 'doughnut', {
        labels: ['JSON', 'CSV', 'Excel', 'XML'],
        datasets: [{
            data: [0, 0, 0, 0],
            backgroundColor: [
                CONFIG.CHART_COLORS.primary,
                CONFIG.CHART_COLORS.success,
                CONFIG.CHART_COLORS.warning,
                CONFIG.CHART_COLORS.info
            ]
        }]
    }, {
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    });
}

function updateDataFormatChart() {
    if (!dataChart) return;
    
    const formatCounts = {
        json: datasetsData.filter(ds => ds.format === 'json').length,
        csv: datasetsData.filter(ds => ds.format === 'csv').length,
        excel: datasetsData.filter(ds => ds.format === 'excel').length,
        xml: datasetsData.filter(ds => ds.format === 'xml').length
    };
    
    dataChart.data.datasets[0].data = [
        formatCounts.json,
        formatCounts.csv,
        formatCounts.excel,
        formatCounts.xml
    ];
    dataChart.update();
}

function populateDatasetDropdown() {
    const select = document.getElementById('export-dataset');
    if (!select) return;
    
    // Clear existing options except the first one
    select.innerHTML = '<option value="">Choose dataset...</option>';
    
    datasetsData.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.id;
        option.textContent = `${dataset.name} (${dataset.records_count.toLocaleString()} records)`;
        select.appendChild(option);
    });
}

async function previewDataset(datasetId) {
    const dataset = datasetsData.find(ds => ds.id === datasetId);
    if (!dataset) return;

    selectedDatasetId = datasetId;

    try {
        showLoading();

        // Load real preview data from API - NO MOCK DATA
        const previewData = await loadDatasetPreview(datasetId);

        const container = document.getElementById('data-preview-container');

        if (currentPreviewMode === 'table') {
            showTablePreview(previewData, dataset);
        } else {
            showJsonPreview(previewData);
        }
    } catch (error) {
        console.error('Failed to preview dataset:', error);
        showAlert('Failed to load dataset preview', 'danger');
    } finally {
        hideLoading();
    }
}

async function loadDatasetPreview(datasetId) {
    try {
        // Fetch real preview data from API - NO MOCK DATA
        const response = await apiRequest(`/datasets/${datasetId}/preview`);
        return response.data || [];
    } catch (error) {
        console.error('Failed to load dataset preview:', error);
        showAlert('Failed to load dataset preview', 'danger');
        return [];
    }
}

function showTablePreview(data, dataset) {
    const container = document.getElementById('data-preview-container');
    
    if (data.length === 0) {
        container.innerHTML = '<p class="text-muted">No data to preview</p>';
        return;
    }
    
    const columns = Object.keys(dataset.schema).map(field => ({
        field: field,
        title: field.charAt(0).toUpperCase() + field.slice(1),
        render: (value) => {
            if (value instanceof Date) {
                return formatDateTime(value);
            }
            return String(value);
        }
    }));
    
    createDataTable('data-preview-container', data, columns);
}

function showJsonPreview(data) {
    const container = document.getElementById('data-preview-container');
    container.innerHTML = `
        <pre class="bg-light p-3 rounded" style="max-height: 400px; overflow-y: auto;">
            <code>${JSON.stringify(data, null, 2)}</code>
        </pre>
    `;
}

function togglePreviewMode(mode) {
    currentPreviewMode = mode;
    
    // Update button states
    document.getElementById('table-view-btn').classList.toggle('active', mode === 'table');
    document.getElementById('json-view-btn').classList.toggle('active', mode === 'json');
    
    // Refresh preview if dataset is selected
    if (selectedDatasetId) {
        previewDataset(selectedDatasetId);
    }
}

async function showDatasetDetails(datasetId) {
    const dataset = datasetsData.find(ds => ds.id === datasetId);
    if (!dataset) return;
    
    selectedDatasetId = datasetId;
    
    const detailsContainer = document.getElementById('data-details-content');
    detailsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Dataset Information</h6>
                <table class="table table-sm">
                    <tr><td><strong>Name:</strong></td><td>${dataset.name}</td></tr>
                    <tr><td><strong>Job ID:</strong></td><td>${dataset.job_id}</td></tr>
                    <tr><td><strong>Format:</strong></td><td>${dataset.format.toUpperCase()}</td></tr>
                    <tr><td><strong>Records:</strong></td><td>${dataset.records_count.toLocaleString()}</td></tr>
                    <tr><td><strong>Size:</strong></td><td>${formatBytes(dataset.size_bytes)}</td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6>Timestamps</h6>
                <table class="table table-sm">
                    <tr><td><strong>Created:</strong></td><td>${formatDateTime(dataset.created_at)}</td></tr>
                    <tr><td><strong>Last Updated:</strong></td><td>${formatDateTime(dataset.last_updated)}</td></tr>
                </table>
            </div>
        </div>
        
        <div class="mt-3">
            <h6>Schema</h6>
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>Field Name</th>
                            <th>Data Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(dataset.schema).map(([field, type]) => 
                            `<tr><td><code>${field}</code></td><td><span class="badge bg-secondary">${type}</span></td></tr>`
                        ).join('')}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="mt-3">
            <h6>Sample Data</h6>
            <div id="modal-preview-container">
                <div class="text-center py-2">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading preview...</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('dataDetailsModal'));
    modal.show();
    
    // Load real preview data
    setTimeout(async () => {
        try {
            const previewData = await loadDatasetPreview(dataset.id);
            const previewContainer = document.getElementById('modal-preview-container');

            if (previewData.length > 0) {
                const columns = Object.keys(dataset.schema).map(field => ({
                    field: field,
                    title: field,
                    render: (value) => {
                        if (value instanceof Date) {
                            return formatDateTime(value);
                        }
                        return String(value);
                    }
                }));

                createDataTable('modal-preview-container', previewData.slice(0, 5), columns);
            } else {
                previewContainer.innerHTML = '<p class="text-muted">No data to preview</p>';
            }
        } catch (error) {
            const previewContainer = document.getElementById('modal-preview-container');
            previewContainer.innerHTML = '<p class="text-danger">Failed to load preview data</p>';
        }
    }, 500);
}

async function performExport() {
    try {
        const datasetId = document.getElementById('export-dataset').value;
        const format = document.getElementById('export-format').value;
        const filename = document.getElementById('export-filename').value;
        const limit = document.getElementById('export-limit').value;
        const offset = document.getElementById('export-offset').value;
        const includeMetadata = document.getElementById('include-metadata').checked;
        const compress = document.getElementById('compress-export').checked;
        const filters = document.getElementById('export-filters').value;
        
        if (!datasetId || !format) {
            showAlert('Please select a dataset and format', 'warning');
            return;
        }
        
        let parsedFilters = {};
        if (filters.trim()) {
            try {
                parsedFilters = JSON.parse(filters);
            } catch (e) {
                showAlert('Invalid JSON in filters', 'danger');
                return;
            }
        }
        
        const exportData = {
            dataset_id: datasetId,
            format: format,
            filename: filename || null,
            limit: limit ? parseInt(limit) : null,
            offset: parseInt(offset) || 0,
            include_metadata: includeMetadata,
            compress: compress,
            filters: parsedFilters
        };

        // Start real export process via API
        showLoading();
        const response = await apiRequest('/exports', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(exportData)
        });

        showAlert('Export started successfully. You will be notified when it\'s ready.', 'success');

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('exportDataModal'));
        modal.hide();

        // Reset form
        document.getElementById('export-form').reset();

        // Refresh export history
        await loadExportHistory();
        updateExportHistoryTable();
        
    } catch (error) {
        console.error('Failed to export data:', error);
        showAlert('Failed to export data', 'danger');
    }
}

// Public functions
function refreshData() {
    loadDataManagementData();
    showAlert('Data refreshed', 'success', 2000);
}

function exportData() {
    const modal = new bootstrap.Modal(document.getElementById('exportDataModal'));
    modal.show();
}

function exportDataset(datasetId) {
    selectedDatasetId = datasetId;
    document.getElementById('export-dataset').value = datasetId;
    exportData();
}

function exportSelectedDataset() {
    if (selectedDatasetId) {
        exportDataset(selectedDatasetId);
        
        // Close details modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('dataDetailsModal'));
        modal.hide();
    }
}

function filterByFormat(format) {
    // Implementation for filtering datasets by format
    showAlert(`Filter by ${format} format - coming soon`, 'info');
}

async function deleteDataset(datasetId) {
    if (!confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
        return;
    }

    try {
        showLoading();

        // Delete via API - NO MOCK DATA
        await apiRequest(`/datasets/${datasetId}`, {
            method: 'DELETE'
        });

        // Refresh data after deletion
        await loadDataManagementData();

        showAlert('Dataset deleted successfully', 'success');

    } catch (error) {
        console.error('Failed to delete dataset:', error);
        showAlert('Failed to delete dataset', 'danger');
    } finally {
        hideLoading();
    }
}

async function deleteExport(exportId) {
    if (!confirm('Are you sure you want to delete this export?')) {
        return;
    }

    try {
        showLoading();

        // Delete via API - NO MOCK DATA
        await apiRequest(`/exports/${exportId}`, {
            method: 'DELETE'
        });

        // Refresh export history after deletion
        await loadExportHistory();
        updateExportHistoryTable();

        showAlert('Export deleted successfully', 'success');

    } catch (error) {
        console.error('Failed to delete export:', error);
        showAlert('Failed to delete export', 'danger');
    } finally {
        hideLoading();
    }
}
