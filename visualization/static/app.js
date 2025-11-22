/**
 * NPCPU Dashboard Application
 *
 * Handles WebSocket connection, real-time updates, and visualization
 * for the NPCPU simulation dashboard.
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
    connected: false,
    paused: false,
    speed: 1.0,
    currentSimId: 'main',
    selectedOrganism: null,
    populationHistory: [],
    maxHistoryLength: 100
};

// =============================================================================
// WebSocket Connection
// =============================================================================

let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 2000;

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        state.connected = true;
        reconnectAttempts = 0;
        updateConnectionStatus(true);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.connected = false;
        updateConnectionStatus(false);

        // Attempt reconnection
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`Reconnecting (attempt ${reconnectAttempts})...`);
            setTimeout(connect, reconnectDelay);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleMessage(message);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

function sendMessage(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

function handleMessage(message) {
    switch (message.type) {
        case 'state_update':
            updateState(message.state);
            break;
        case 'control_update':
            updateControls(message);
            break;
        default:
            console.log('Unknown message type:', message.type);
    }
}

// =============================================================================
// UI Updates
// =============================================================================

function updateConnectionStatus(connected) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('connectionText');

    if (connected) {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

function updateState(stateData) {
    // Update statistics
    document.getElementById('tickCount').textContent = stateData.tick || 0;
    document.getElementById('popSize').textContent = stateData.population_size || 0;
    document.getElementById('avgAge').textContent = Math.round(stateData.avg_age || 0);
    document.getElementById('avgEnergy').textContent = Math.round((stateData.avg_energy || 0) * 100) + '%';
    document.getElementById('avgHealth').textContent = Math.round((stateData.avg_health || 0) * 100) + '%';
    document.getElementById('birthDeathRatio').textContent =
        `${stateData.total_births || 0}/${stateData.total_deaths || 0}`;

    // Update organism count
    document.getElementById('orgCount').textContent = stateData.organisms ? stateData.organisms.length : 0;

    // Update organisms list
    updateOrganismsList(stateData.organisms || []);

    // Update event timeline
    updateEventTimeline(stateData.events || []);

    // Update world canvas
    renderWorld(stateData.organisms || []);

    // Update charts
    updateCharts(stateData);

    // Store population history
    state.populationHistory.push({
        tick: stateData.tick,
        size: stateData.population_size,
        avgEnergy: stateData.avg_energy,
        avgHealth: stateData.avg_health
    });

    if (state.populationHistory.length > state.maxHistoryLength) {
        state.populationHistory.shift();
    }
}

function updateControls(controlData) {
    state.paused = controlData.paused;
    state.speed = controlData.speed;

    const pauseBtn = document.getElementById('pauseBtn');
    pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', state.paused);

    const speedSlider = document.getElementById('speedSlider');
    speedSlider.value = state.speed;
    document.getElementById('speedLabel').textContent = state.speed.toFixed(1) + 'x';
}

function updateOrganismsList(organisms) {
    const container = document.getElementById('organismsList');

    // Sort by age (oldest first)
    const sorted = [...organisms].sort((a, b) => b.age - a.age);

    container.innerHTML = sorted.map(org => `
        <div class="organism-item ${state.selectedOrganism === org.id ? 'selected' : ''}"
             onclick="selectOrganism('${org.id}')">
            <div class="organism-name">${org.name}</div>
            <div class="organism-info">
                <span>Age: ${org.age}</span>
                <span>${org.phase}</span>
                <span>${org.alive ? 'Alive' : 'Dead'}</span>
            </div>
            <div class="organism-bars">
                <div class="bar-container" title="Energy">
                    <div class="bar energy" style="width: ${(org.energy || 0) * 100}%"></div>
                </div>
                <div class="bar-container" title="Health">
                    <div class="bar health" style="width: ${(org.health || 0) * 100}%"></div>
                </div>
            </div>
        </div>
    `).join('');
}

function updateEventTimeline(events) {
    const container = document.getElementById('eventTimeline');

    // Show most recent events first
    const sorted = [...events].reverse().slice(0, 20);

    container.innerHTML = sorted.map(event => {
        const typeClass = event.type.replace('.', '-');
        const time = new Date(event.timestamp).toLocaleTimeString();

        return `
            <div class="event-item ${typeClass}">
                <div class="event-type">${formatEventType(event.type)}</div>
                <div>${formatEventData(event)}</div>
                <div class="event-time">${time}</div>
            </div>
        `;
    }).join('');
}

function formatEventType(type) {
    return type.split('.').map(part =>
        part.charAt(0).toUpperCase() + part.slice(1)
    ).join(' ');
}

function formatEventData(event) {
    const data = event.data || {};
    if (data.organism_name) {
        return data.organism_name;
    }
    if (data.cause) {
        return `Cause: ${data.cause}`;
    }
    return '';
}

// =============================================================================
// Control Functions
// =============================================================================

function togglePause() {
    state.paused = !state.paused;
    sendMessage({
        type: state.paused ? 'pause' : 'resume',
        sim_id: state.currentSimId
    });

    const pauseBtn = document.getElementById('pauseBtn');
    pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('active', state.paused);
}

function stepOnce() {
    // Request a single state update
    sendMessage({
        type: 'get_state',
        sim_id: state.currentSimId
    });
}

function setSpeed(value) {
    state.speed = parseFloat(value);
    sendMessage({
        type: 'set_speed',
        sim_id: state.currentSimId,
        speed: state.speed
    });
    document.getElementById('speedLabel').textContent = state.speed.toFixed(1) + 'x';
}

function selectOrganism(id) {
    state.selectedOrganism = state.selectedOrganism === id ? null : id;
    // Re-render to show selection
    const items = document.querySelectorAll('.organism-item');
    items.forEach(item => {
        const itemId = item.getAttribute('onclick').match(/'([^']+)'/)[1];
        item.classList.toggle('selected', itemId === state.selectedOrganism);
    });
}

// =============================================================================
// World Rendering
// =============================================================================

let worldCanvas, worldCtx;

function initWorldCanvas() {
    worldCanvas = document.getElementById('worldCanvas');
    worldCtx = worldCanvas.getContext('2d');

    // Handle resize
    function resize() {
        const container = worldCanvas.parentElement;
        worldCanvas.width = container.clientWidth;
        worldCanvas.height = container.clientHeight - 40; // Account for title
    }

    resize();
    window.addEventListener('resize', resize);
}

function renderWorld(organisms) {
    if (!worldCtx) return;

    const width = worldCanvas.width;
    const height = worldCanvas.height;

    // Clear canvas
    worldCtx.fillStyle = '#0a0a1a';
    worldCtx.fillRect(0, 0, width, height);

    // Draw grid
    worldCtx.strokeStyle = '#1a1a2e';
    worldCtx.lineWidth = 1;
    const gridSize = 50;

    for (let x = 0; x < width; x += gridSize) {
        worldCtx.beginPath();
        worldCtx.moveTo(x, 0);
        worldCtx.lineTo(x, height);
        worldCtx.stroke();
    }

    for (let y = 0; y < height; y += gridSize) {
        worldCtx.beginPath();
        worldCtx.moveTo(0, y);
        worldCtx.lineTo(width, y);
        worldCtx.stroke();
    }

    // Draw organisms
    organisms.forEach((org, index) => {
        // Calculate position (use hash of ID if no position)
        let x, y;
        if (org.position && typeof org.position.x === 'number') {
            x = (org.position.x % width + width) % width;
            y = (org.position.y % height + height) % height;
        } else {
            // Distribute organisms in a pattern based on index
            const cols = Math.ceil(Math.sqrt(organisms.length));
            const row = Math.floor(index / cols);
            const col = index % cols;
            x = (col + 0.5) * (width / cols);
            y = (row + 0.5) * (height / Math.ceil(organisms.length / cols));
        }

        // Size based on age
        const baseSize = 8;
        const maxSize = 20;
        const sizeMultiplier = Math.min(org.age / 100, 1);
        const size = baseSize + (maxSize - baseSize) * sizeMultiplier;

        // Color based on energy/health
        const energy = org.energy || 0.5;
        const health = org.health || 0.5;

        // HSL color: hue based on phase, saturation on health, lightness on energy
        let hue;
        switch (org.phase) {
            case 'NASCENT': hue = 120; break;      // Green
            case 'DEVELOPING': hue = 180; break;   // Cyan
            case 'MATURE': hue = 220; break;       // Blue
            case 'DECLINING': hue = 40; break;     // Orange
            case 'TERMINAL': hue = 0; break;       // Red
            default: hue = 280;                     // Purple
        }

        const saturation = 50 + health * 50;
        const lightness = 30 + energy * 40;

        // Draw organism
        worldCtx.beginPath();
        worldCtx.arc(x, y, size, 0, Math.PI * 2);
        worldCtx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        worldCtx.fill();

        // Draw border for selected organism
        if (state.selectedOrganism === org.id) {
            worldCtx.strokeStyle = '#e94560';
            worldCtx.lineWidth = 3;
            worldCtx.stroke();
        }

        // Draw energy/health indicators
        const barWidth = size * 2;
        const barHeight = 3;
        const barY = y + size + 5;

        // Energy bar
        worldCtx.fillStyle = '#333';
        worldCtx.fillRect(x - barWidth / 2, barY, barWidth, barHeight);
        worldCtx.fillStyle = '#ffd93d';
        worldCtx.fillRect(x - barWidth / 2, barY, barWidth * energy, barHeight);

        // Health bar
        worldCtx.fillStyle = '#333';
        worldCtx.fillRect(x - barWidth / 2, barY + barHeight + 1, barWidth, barHeight);
        worldCtx.fillStyle = '#51cf66';
        worldCtx.fillRect(x - barWidth / 2, barY + barHeight + 1, barWidth * health, barHeight);
    });

    // Draw legend
    drawLegend();
}

function drawLegend() {
    const phases = [
        { name: 'Nascent', hue: 120 },
        { name: 'Developing', hue: 180 },
        { name: 'Mature', hue: 220 },
        { name: 'Declining', hue: 40 },
        { name: 'Terminal', hue: 0 }
    ];

    const startX = 10;
    const startY = 10;

    worldCtx.font = '10px sans-serif';

    phases.forEach((phase, i) => {
        const y = startY + i * 18;

        worldCtx.beginPath();
        worldCtx.arc(startX + 6, y + 6, 6, 0, Math.PI * 2);
        worldCtx.fillStyle = `hsl(${phase.hue}, 70%, 50%)`;
        worldCtx.fill();

        worldCtx.fillStyle = '#aaa';
        worldCtx.fillText(phase.name, startX + 16, y + 10);
    });
}

// =============================================================================
// Charts
// =============================================================================

let populationChart, energyChart, traitChart;

function initCharts() {
    // Population over time chart
    const popCtx = document.getElementById('populationChart').getContext('2d');
    populationChart = new Chart(popCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Population',
                data: [],
                borderColor: '#e94560',
                backgroundColor: 'rgba(233, 69, 96, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#1a1a2e' },
                    ticks: { color: '#aaa' }
                }
            }
        }
    });

    // Energy distribution chart
    const energyCtx = document.getElementById('energyChart').getContext('2d');
    energyChart = new Chart(energyCtx, {
        type: 'bar',
        data: {
            labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
            datasets: [{
                label: 'Energy',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#ff6b6b',
                    '#ffa94d',
                    '#ffd93d',
                    '#a9e34b',
                    '#51cf66'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#aaa', font: { size: 9 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#1a1a2e' },
                    ticks: { color: '#aaa' }
                }
            }
        }
    });

    // Trait distribution chart
    const traitCtx = document.getElementById('traitChart').getContext('2d');
    traitChart = new Chart(traitCtx, {
        type: 'radar',
        data: {
            labels: ['Vitality', 'Metabolism', 'Resilience', 'Recovery', 'Awareness'],
            datasets: [{
                label: 'Average Traits',
                data: [0.5, 0.5, 0.5, 0.5, 0.5],
                borderColor: '#e94560',
                backgroundColor: 'rgba(233, 69, 96, 0.2)',
                pointBackgroundColor: '#e94560'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1.5,
                    grid: { color: '#1a1a2e' },
                    angleLines: { color: '#1a1a2e' },
                    pointLabels: { color: '#aaa', font: { size: 9 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

function updateCharts(stateData) {
    // Update population chart
    if (populationChart && state.populationHistory.length > 0) {
        populationChart.data.labels = state.populationHistory.map(h => h.tick);
        populationChart.data.datasets[0].data = state.populationHistory.map(h => h.size);
        populationChart.update('none');
    }

    // Update energy distribution
    if (energyChart && stateData.organisms) {
        const buckets = [0, 0, 0, 0, 0];
        stateData.organisms.forEach(org => {
            const energy = org.energy || 0;
            const bucket = Math.min(4, Math.floor(energy * 5));
            buckets[bucket]++;
        });
        energyChart.data.datasets[0].data = buckets;
        energyChart.update('none');
    }

    // Update trait chart
    if (traitChart && stateData.organisms && stateData.organisms.length > 0) {
        const traitSums = {
            vitality: 0,
            metabolism: 0,
            resilience: 0,
            recovery: 0,
            awareness: 0
        };

        stateData.organisms.forEach(org => {
            if (org.traits) {
                Object.keys(traitSums).forEach(trait => {
                    traitSums[trait] += org.traits[trait] || 1.0;
                });
            }
        });

        const count = stateData.organisms.length;
        traitChart.data.datasets[0].data = [
            traitSums.vitality / count,
            traitSums.metabolism / count,
            traitSums.resilience / count,
            traitSums.recovery / count,
            traitSums.awareness / count
        ];
        traitChart.update('none');
    }
}

// =============================================================================
// Initialization
// =============================================================================

function init() {
    initWorldCanvas();
    initCharts();
    connect();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
