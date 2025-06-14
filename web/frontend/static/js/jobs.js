// Jobs page JavaScript functionality

let jobsData = [];
let currentPage = 1;
let pageSize = 10;
let currentFilter = 'all';
let currentSearch = '';
let selectedJobId = null;

// Initialize jobs page
document.addEventListener('DOMContentLoaded', function() {
    initializeJobsPage();
    setupEventListeners();
    loadJobs();
});

function initializeJobsPage() {
    console.log('Initializing jobs page...');
    
    // Set up real-time updates
    setupRealTimeUpdates();
    
    // Handle URL hash for direct actions
    handleUrlHash();
}

function setupEventListeners() {
    // Listen for WebSocket updates
    document.addEventListener('jobUpdate', handleJobUpdate);
    
    // Search input
    const searchInput = document.getElementById('job-search');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchJobs();
            }
        });
    }
    
    // Form submission
    const createForm = document.getElementById('create-job-form');
    if (createForm) {
        createForm.addEventListener('submit', function(e) {
            e.preventDefault();
            createJob();
        });
    }
}

function setupRealTimeUpdates() {
    const wsManager = getWebSocketManager();
    if (wsManager) {
        wsManager.subscribe('job_updates', handleJobUpdate);
    }
}

function handleUrlHash() {
    const hash = window.location.hash;
    if (hash === '#create') {
        // Show create job modal
        const modal = new bootstrap.Modal(document.getElementById('createJobModal'));
        modal.show();
    }
}

async function loadJobs() {
    try {
        const filters = {
            page: currentPage,
            size: pageSize
        };

        if (currentFilter !== 'all') {
            filters.status = currentFilter;
        }

        if (currentSearch) {
            filters.search = currentSearch;
        }

        const params = new URLSearchParams(filters);
        const response = await apiRequest(`/jobs?${params}`, {}, 'jobs-table-container');
        jobsData = response.jobs || [];

        updateJobsTable();
        updateJobStatistics();
        updatePagination(response.total_count || 0);

    } catch (error) {
        console.error('Failed to load jobs:', error);
        showAlert('Failed to load jobs. Please check if the backend is running.', 'danger', 8000);

        // Show empty state
        const container = document.getElementById('jobs-table-container');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="bi bi-exclamation-triangle display-4 text-warning"></i>
                    <h5 class="mt-3 text-muted">Unable to Load Jobs</h5>
                    <p class="text-muted">Please check your connection and try again.</p>
                    <button class="btn btn-primary" onclick="loadJobs()">
                        <i class="bi bi-arrow-clockwise"></i> Retry
                    </button>
                </div>
            `;
        }
    }
}

function updateJobsTable() {
    const container = document.getElementById('jobs-table-container');
    if (!container) return;
    
    if (jobsData.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-inbox display-4 text-muted"></i>
                <p class="text-muted mt-2">No jobs found</p>
                <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createJobModal">
                    Create Your First Job
                </button>
            </div>
        `;
        return;
    }
    
    const columns = [
        {
            field: 'name',
            title: 'Job Name',
            render: (value, row) => `
                <div>
                    <strong><a href="#" onclick="showJobDetails('${row.id}')">${value}</a></strong>
                    <br>
                    <small class="text-muted">${row.job_type}</small>
                </div>
            `
        },
        {
            field: 'status',
            title: 'Status',
            render: (value) => `<span class="job-status ${value}">${value.toUpperCase()}</span>`
        },
        {
            field: 'progress',
            title: 'Progress',
            render: (value, row) => {
                const progressClass = row.status === 'failed' ? 'bg-danger' : 
                                    row.status === 'completed' ? 'bg-success' : 'bg-primary';
                return `
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar ${progressClass}" style="width: ${value}%">
                            ${value}%
                        </div>
                    </div>
                `;
            }
        },
        {
            field: 'created_at',
            title: 'Created',
            render: (value) => formatDateTime(value)
        },
        {
            field: 'priority',
            title: 'Priority',
            render: (value) => {
                const badgeClass = value === 'urgent' ? 'bg-danger' :
                                 value === 'high' ? 'bg-warning' :
                                 value === 'low' ? 'bg-secondary' : 'bg-primary';
                return `<span class="badge ${badgeClass}">${value.toUpperCase()}</span>`;
            }
        },
        {
            field: 'actions',
            title: 'Actions',
            render: (value, row) => `
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="showJobDetails('${row.id}')">
                        <i class="bi bi-eye"></i>
                    </button>
                    ${row.status === 'running' ? 
                        `<button class="btn btn-outline-danger" onclick="cancelJob('${row.id}')">
                            <i class="bi bi-stop"></i>
                        </button>` : ''
                    }
                    ${row.status === 'failed' ? 
                        `<button class="btn btn-outline-warning" onclick="retryJob('${row.id}')">
                            <i class="bi bi-arrow-clockwise"></i>
                        </button>` : ''
                    }
                </div>
            `
        }
    ];
    
    createDataTable('jobs-table-container', jobsData, columns);
}

function updateJobStatistics() {
    const stats = {
        total: jobsData.length,
        running: jobsData.filter(j => j.status === 'running').length,
        completed: jobsData.filter(j => j.status === 'completed').length,
        failed: jobsData.filter(j => j.status === 'failed').length
    };
    
    document.getElementById('total-jobs-count').textContent = stats.total;
    document.getElementById('running-jobs-count').textContent = stats.running;
    document.getElementById('completed-jobs-count').textContent = stats.completed;
    document.getElementById('failed-jobs-count').textContent = stats.failed;
}

function updatePagination(totalCount) {
    const totalPages = Math.ceil(totalCount / pageSize);
    const paginationContainer = document.getElementById('jobs-pagination');
    
    if (!paginationContainer || totalPages <= 1) {
        paginationContainer.innerHTML = '';
        return;
    }
    
    let paginationHTML = '';
    
    // Previous button
    paginationHTML += `
        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage - 1})">Previous</a>
        </li>
    `;
    
    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    for (let i = startPage; i <= endPage; i++) {
        paginationHTML += `
            <li class="page-item ${i === currentPage ? 'active' : ''}">
                <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
            </li>
        `;
    }
    
    // Next button
    paginationHTML += `
        <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
            <a class="page-link" href="#" onclick="changePage(${currentPage + 1})">Next</a>
        </li>
    `;
    
    paginationContainer.innerHTML = paginationHTML;
}

async function createJob() {
    try {
        const form = document.getElementById('create-job-form');
        const formData = new FormData(form);
        
        // Collect selectors
        const selectors = {};
        const selectorNames = document.querySelectorAll('input[name="selector-name"]');
        const selectorValues = document.querySelectorAll('input[name="selector-value"]');
        
        for (let i = 0; i < selectorNames.length; i++) {
            const name = selectorNames[i].value.trim();
            const value = selectorValues[i].value.trim();
            if (name && value) {
                selectors[name] = value;
            }
        }
        
        const jobData = {
            name: document.getElementById('job-name').value,
            description: document.getElementById('job-description').value,
            job_type: document.getElementById('job-type').value,
            priority: document.getElementById('priority').value,
            parameters: {
                url: document.getElementById('target-url').value,
                selectors: selectors,
                max_pages: parseInt(document.getElementById('max-pages').value) || 1,
                delay: parseFloat(document.getElementById('delay').value) || 1,
                output_format: document.getElementById('output-format').value,
                render_js: document.getElementById('render-js').checked,
                anti_detection: document.getElementById('anti-detection').checked,
                clean_data: document.getElementById('clean-data').checked
            },
            created_by: 'current_user' // This should come from authentication
        };
        
        const response = await apiRequest('/jobs', {
            method: 'POST',
            body: JSON.stringify(jobData)
        });
        
        if (response) {
            showAlert('Job created successfully', 'success');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('createJobModal'));
            modal.hide();
            
            // Reset form
            form.reset();
            
            // Reload jobs
            loadJobs();
        }
        
    } catch (error) {
        console.error('Failed to create job:', error);
        showAlert('Failed to create job', 'danger');
    }
}

async function showJobDetails(jobId) {
    try {
        selectedJobId = jobId;
        const job = jobsData.find(j => j.id === jobId);
        
        if (!job) {
            showAlert('Job not found', 'danger');
            return;
        }
        
        const detailsContainer = document.getElementById('job-details-content');
        detailsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Basic Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Name:</strong></td><td>${job.name}</td></tr>
                        <tr><td><strong>Type:</strong></td><td>${job.job_type}</td></tr>
                        <tr><td><strong>Status:</strong></td><td><span class="job-status ${job.status}">${job.status}</span></td></tr>
                        <tr><td><strong>Priority:</strong></td><td>${job.priority}</td></tr>
                        <tr><td><strong>Progress:</strong></td><td>${job.progress}%</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Timestamps</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Created:</strong></td><td>${formatDateTime(job.created_at)}</td></tr>
                        <tr><td><strong>Started:</strong></td><td>${formatDateTime(job.started_at)}</td></tr>
                        <tr><td><strong>Completed:</strong></td><td>${formatDateTime(job.completed_at)}</td></tr>
                    </table>
                </div>
            </div>
            
            ${job.description ? `
                <div class="mt-3">
                    <h6>Description</h6>
                    <p>${job.description}</p>
                </div>
            ` : ''}
            
            <div class="mt-3">
                <h6>Parameters</h6>
                <pre class="bg-light p-3 rounded"><code>${JSON.stringify(job.parameters, null, 2)}</code></pre>
            </div>
            
            ${job.result ? `
                <div class="mt-3">
                    <h6>Result</h6>
                    <pre class="bg-light p-3 rounded"><code>${JSON.stringify(job.result, null, 2)}</code></pre>
                </div>
            ` : ''}
            
            ${job.error_message ? `
                <div class="mt-3">
                    <h6>Error</h6>
                    <div class="alert alert-danger">${job.error_message}</div>
                </div>
            ` : ''}
        `;
        
        // Show/hide cancel button based on job status
        const cancelBtn = document.getElementById('cancel-job-btn');
        if (cancelBtn) {
            cancelBtn.style.display = job.status === 'running' ? 'block' : 'none';
        }
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
        modal.show();
        
    } catch (error) {
        console.error('Failed to show job details:', error);
        showAlert('Failed to load job details', 'danger');
    }
}

// Event handlers and utility functions
function handleJobUpdate(event) {
    const jobData = event.detail;
    
    // Update job in local data
    const jobIndex = jobsData.findIndex(j => j.id === jobData.job_id);
    if (jobIndex !== -1) {
        jobsData[jobIndex] = { ...jobsData[jobIndex], ...jobData };
        updateJobsTable();
        updateJobStatistics();
    }
}

function addSelector() {
    const container = document.getElementById('selectors-container');
    const newRow = document.createElement('div');
    newRow.className = 'row mb-2';
    newRow.innerHTML = `
        <div class="col-md-4">
            <input type="text" class="form-control" placeholder="Field name" name="selector-name">
        </div>
        <div class="col-md-6">
            <input type="text" class="form-control" placeholder="CSS selector" name="selector-value">
        </div>
        <div class="col-md-2">
            <button type="button" class="btn btn-outline-danger btn-sm" onclick="removeSelector(this)">
                <i class="bi bi-trash"></i>
            </button>
        </div>
    `;
    container.appendChild(newRow);
}

function removeSelector(button) {
    button.closest('.row').remove();
}

function filterJobs(status) {
    currentFilter = status;
    currentPage = 1;
    loadJobs();
}

function searchJobs() {
    currentSearch = document.getElementById('job-search').value;
    currentPage = 1;
    loadJobs();
}

function changePage(page) {
    if (page < 1) return;
    currentPage = page;
    loadJobs();
}

function refreshJobs() {
    loadJobs();
    showAlert('Jobs refreshed', 'success', 2000);
}

async function cancelSelectedJob() {
    if (!selectedJobId) return;
    
    try {
        await cancelJob(selectedJobId);
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('jobDetailsModal'));
        modal.hide();
        
        // Reload jobs
        loadJobs();
        
    } catch (error) {
        console.error('Failed to cancel job:', error);
    }
}

async function retryJob(jobId) {
    // Implementation for retrying a failed job
    showAlert('Retry functionality coming soon', 'info');
}
