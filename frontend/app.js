// Context Budget Optimizer Frontend JavaScript

const API_BASE = '';

// DOM Elements
const queryInput = document.getElementById('query-input');
const budgetSlider = document.getElementById('budget-slider');
const budgetValue = document.getElementById('budget-value');
const submitBtn = document.getElementById('submit-btn');
const clearBtn = document.getElementById('clear-btn');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const answerContent = document.getElementById('answer-content');
const chunksContainer = document.getElementById('chunks-container');
const toggleChunksBtn = document.getElementById('toggle-chunks-btn');
const chunkFilter = document.getElementById('chunk-filter');

// State
let currentQueryId = null;
let currentChunks = [];
let chunksChart = null;
let budgetChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Budget slider
    budgetSlider.addEventListener('input', (e) => {
        budgetValue.textContent = e.target.value;
        // Auto-estimate cost when budget changes (if query exists)
        if (queryInput.value.trim()) {
            debounceEstimateCost();
        }
    });

    // Estimate cost when query changes
    queryInput.addEventListener('input', debounceEstimateCost);

    // Submit button
    submitBtn.addEventListener('click', handleSubmit);

    // Clear button
    clearBtn.addEventListener('click', handleClear);

    // Toggle chunks
    toggleChunksBtn.addEventListener('click', toggleChunks);

    // Filter chunks
    chunkFilter.addEventListener('change', renderChunks);

    // Enter key to submit
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleSubmit();
        }
    });
});

async function handleSubmit() {
    const query = queryInput.value.trim();
    const budget = parseInt(budgetSlider.value);

    if (!query) {
        showError('Please enter a query');
        return;
    }

    // Hide previous results
    hideAllSections();
    showLoading();

    // Estimate cost first
    try {
        const estimateResponse = await fetch(`${API_BASE}/api/estimate-cost`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                budget: budget
            })
        });

        if (estimateResponse.ok) {
            const estimate = await estimateResponse.json();
            displayCostEstimate(estimate);
        }
    } catch (error) {
        console.error('Error estimating cost:', error);
    }

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                budget: budget
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to process query');
        }

        const data = await response.json();
        currentQueryId = data.query_id;

        // Load chunks
        await loadChunks(data.query_id);

        // Display results
        displayResults(data);
        showResults();

    } catch (error) {
        showError(error.message);
    }
}

async function loadChunks(queryId) {
    try {
        const response = await fetch(`${API_BASE}/api/chunks/${queryId}`);
        if (response.ok) {
            const chunks = await response.json();
            currentChunks = chunks;
        }
    } catch (error) {
        console.error('Error loading chunks:', error);
    }
}

function displayResults(data) {
    // Answer
    answerContent.textContent = data.answer;

    // Model and tokens
    document.getElementById('model-info').textContent = `Model: ${data.model}`;
    const usage = data.usage;
    document.getElementById('tokens-info').textContent = 
        `Tokens: ${usage.total_tokens} (${usage.prompt_tokens} prompt + ${usage.completion_tokens} completion)`;

    // Optimization stats
    const opt = data.optimization;
    document.getElementById('chunks-evaluated').textContent = opt.chunks_evaluated || '-';
    document.getElementById('chunks-selected').textContent = opt.chunks_selected || '-';
    document.getElementById('budget-used').textContent = 
        opt.budget_used ? `${opt.budget_used.toFixed(1)}%` : '-';
    document.getElementById('tokens-used').textContent = opt.total_tokens || '-';

    // Update charts
    updateCharts(opt);

    // Hide cost estimate (show actual results)
    document.getElementById('cost-estimate-section').classList.add('hidden');

    // Render chunks
    renderChunks();
}

function updateCharts(opt) {
    // Chunks chart
    const chunksCtx = document.getElementById('chunks-chart').getContext('2d');
    if (chunksChart) {
        chunksChart.destroy();
    }
    chunksChart = new Chart(chunksCtx, {
        type: 'doughnut',
        data: {
            labels: ['Selected', 'Excluded'],
            datasets: [{
                data: [opt.chunks_selected || 0, opt.chunks_excluded || 0],
                backgroundColor: ['#4caf50', '#f44336'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#E8F4F6',
                        font: {
                            family: 'Quantico',
                            size: 12
                        },
                        padding: 15
                    }
                },
                title: {
                    display: true,
                    text: 'Chunk Selection',
                    color: '#E8F4F6',
                    font: {
                        family: 'Quantico',
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 10
                }
            }
        }
    });

    // Budget chart
    const budgetCtx = document.getElementById('budget-chart').getContext('2d');
    if (budgetChart) {
        budgetChart.destroy();
    }
    const budgetUsed = opt.budget_used || 0;
    const budgetRemaining = 100 - budgetUsed;
    budgetChart = new Chart(budgetCtx, {
        type: 'bar',
        data: {
            labels: ['Used', 'Remaining'],
            datasets: [{
                label: 'Budget Usage (%)',
                data: [budgetUsed, budgetRemaining],
                backgroundColor: ['#134E5E', '#0B3037'],
                borderColor: ['#E8F4F6', '#A8C8D0'],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Budget Utilization',
                    color: '#E8F4F6',
                    font: {
                        family: 'Quantico',
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 10
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#A8C8D0',
                        font: {
                            family: 'Quantico',
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(232, 244, 246, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#A8C8D0',
                        font: {
                            family: 'Quantico',
                            size: 11
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function displayCostEstimate(estimate) {
    document.getElementById('est-prompt-tokens').textContent = estimate.estimated_prompt_tokens || '-';
    document.getElementById('est-completion-tokens').textContent = estimate.estimated_completion_tokens || '-';
    document.getElementById('est-total-tokens').textContent = estimate.estimated_total_tokens || '-';
    document.getElementById('cost-estimate-section').classList.remove('hidden');
}

function renderChunks() {
    if (!currentChunks.length) {
        chunksContainer.innerHTML = '<p>No chunks available</p>';
        return;
    }

    const filter = chunkFilter.value;
    let chunksToShow = currentChunks;

    if (filter === 'selected') {
        chunksToShow = currentChunks.filter(c => c.included);
    } else if (filter === 'excluded') {
        chunksToShow = currentChunks.filter(c => !c.included);
    }

    // Sort by similarity score
    chunksToShow.sort((a, b) => b.similarity_score - a.similarity_score);

    chunksContainer.innerHTML = chunksToShow.map(chunk => {
        const status = chunk.included ? 'selected' : 'excluded';
        const badgeClass = chunk.included ? 'selected' : 'excluded';
        const badgeText = chunk.included ? 'SELECTED' : 'EXCLUDED';

        return `
            <div class="chunk-card ${status}">
                <div class="chunk-header">
                    <span class="badge ${badgeClass}">${badgeText}</span>
                    <div class="chunk-meta">
                        <span>Score: ${chunk.similarity_score.toFixed(3)}</span>
                        <span>Tokens: ${chunk.token_count}</span>
                        ${chunk.value_per_token ? `<span>Value: ${chunk.value_per_token.toFixed(4)}</span>` : ''}
                    </div>
                </div>
                ${chunk.inclusion_reason ? `<div style="font-size: 0.85em; color: #666; margin-bottom: 8px;">Reason: ${chunk.inclusion_reason}</div>` : ''}
                <div class="chunk-text">${escapeHtml(chunk.text)}</div>
            </div>
        `;
    }).join('');

    if (chunksToShow.length === 0) {
        chunksContainer.innerHTML = '<p>No chunks match the selected filter</p>';
    }
}

function toggleChunks() {
    const isHidden = chunksContainer.classList.contains('hidden');
    if (isHidden) {
        chunksContainer.classList.remove('hidden');
        toggleChunksBtn.textContent = 'Hide Chunks';
    } else {
        chunksContainer.classList.add('hidden');
        toggleChunksBtn.textContent = 'Show Chunks';
    }
}

function handleClear() {
    queryInput.value = '';
    budgetSlider.value = 2000;
    budgetValue.textContent = '2000';
    hideAllSections();
    currentQueryId = null;
    currentChunks = [];
}

function showLoading() {
    hideAllSections();
    loadingSection.classList.remove('hidden');
    submitBtn.disabled = true;
}

function showResults() {
    hideAllSections();
    resultsSection.classList.remove('hidden');
    submitBtn.disabled = false;
}

function showError(message) {
    hideAllSections();
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    submitBtn.disabled = false;
}

function hideAllSections() {
    loadingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Debounce for cost estimation
let estimateTimeout;
function debounceEstimateCost() {
    clearTimeout(estimateTimeout);
    estimateTimeout = setTimeout(async () => {
        const query = queryInput.value.trim();
        const budget = parseInt(budgetSlider.value);
        
        if (!query) {
            document.getElementById('cost-estimate-section').classList.add('hidden');
            return;
        }
        
        try {
            const response = await fetch(`${API_BASE}/api/estimate-cost`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query, budget })
            });
            
            if (response.ok) {
                const estimate = await response.json();
                displayCostEstimate(estimate);
            }
        } catch (error) {
            console.error('Error estimating cost:', error);
        }
    }, 500); // Wait 500ms after user stops typing
}
