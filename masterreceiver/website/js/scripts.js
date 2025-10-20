// Grid configuration
const COLS = 18;
const ROWS = 32;
let grid;
const cells = [];

function initializeGrid() {
    grid = document.getElementById('gameGrid');
    if (!grid) return;

    // Clear existing content if any
    grid.innerHTML = '';

    // Create grid cells
    for (let row = 0; row < ROWS; row++) {
        cells[row] = [];
        for (let col = 0; col < COLS; col++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            grid.appendChild(cell);
            cells[row][col] = cell;
        }
    }
}

/**
 * Paint a cell at position (x, y) with the specified color
 * @param {number} x - Column index (0-17)
 * @param {number} y - Row index (0-31)
 * @param {string} color - CSS color value
 */
function paintCell(x, y, color) {
    if (x >= 0 && x < COLS && y >= 0 && y < ROWS) {
        cells[y][x].style.backgroundColor = color;
    } else {
        console.warn(`Invalid cell coordinates: (${x}, ${y})`);
    }
}

function manualPaint() {
    const x = parseInt(document.getElementById('paintX').value);
    const y = parseInt(document.getElementById('paintY').value);
    const color = document.getElementById('paintColor').value;
    paintCell(x, y, color);
    try {
        if (cells[y] && cells[y][x]) cells[y][x].title = `(${x},${y})`;
    } catch (e) {
        // ignore
    }
}

function clearGrid() {
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
            cells[row][col].style.backgroundColor = '#1a1a1a';
            try { cells[row][col].title = ''; } catch (e) { }
        }
    }
}

// Initialize on DOM ready
window.addEventListener('DOMContentLoaded', () => {
    initializeGrid();
    // Demo paints (safe only if grid initialized)
    try {
        paintCell(5, 10, 'red');
        paintCell(10, 10, 'blue');
        paintCell(15, 8, '#00ff00');
    } catch (e) {
        // ignore if not ready
    }
    // Start receiving live updates from server via SSE
    try {
        startStream();
    } catch (e) {
        console.warn('Failed to start EventSource stream:', e);
    }
});


function startStream() {
    if (typeof EventSource === 'undefined') {
        console.warn('This browser does not support EventSource (SSE). Live updates disabled.');
        return;
    }

    const es = new EventSource('/stream');
    es.onmessage = function (evt) {
        try {
            const data = JSON.parse(evt.data);
            
            // Always update all information continuously, regardless of win state
            
            // Update location grid
            if (data && data.locations) {
                clearGrid();
                data.locations.forEach(loc => {
                    const x = Number(loc.x);
                    const y = Number(loc.y);
                    let color = '#ffff66';
                    const label = (loc.label || '').toLowerCase();
                    if (label.includes('enemy')) color = '#ff6666';
                    else if (label.includes('ally')) color = '#66b3ff';
                    else if (label.includes('princess') || label.includes('tower')) color = '#ffd24d';
                    paintCell(x, y, color);
                    try { if (cells[y] && cells[y][x]) cells[y][x].title = `${loc.label || ''} : (${x},${y})`; } catch (e) {}
                });
            }
            
            // Update cards in hand (handle null/None)
            if (data && data.cards_in_hand !== undefined && data.cards_in_hand !== null) {
                renderCards(data.cards_in_hand);
            } else {
                renderCards([]);
            }
            
            // Update elixir count (handle null/None)
            if (data && data.elixir !== undefined && data.elixir !== null) {
                renderElixir(data.elixir);
            } else {
                renderElixir(null);
            }
            
            // Update status (handle null/None)
            if (data && data.status !== undefined) {
                renderStatus(data.status, data.win_detection);
            }
        } catch (e) {
            console.error('Failed to parse SSE data:', e, evt.data);
        }
    };

    es.onerror = function (err) {
        console.warn('EventSource error', err);
    };

    // store the EventSource reference so it can be closed later if needed
    window._gameStream = es;
}


function renderCards(cards) {
    const list = document.getElementById('cardsList');
    if (!list) return;
    list.innerHTML = '';
    
    if (!cards || cards.length === 0) {
        const el = document.createElement('div');
        el.style.padding = '6px 8px';
        el.style.color = '#888';
        el.style.fontSize = '13px';
        el.textContent = 'No cards data';
        list.appendChild(el);
        return;
    }
    
    cards.forEach(c => {
        const el = document.createElement('div');
        el.style.padding = '6px 8px';
        el.style.background = '#222';
        el.style.border = '1px solid #444';
        el.style.borderRadius = '6px';
        el.style.color = '#fff';
        el.style.fontSize = '13px';
        el.textContent = c.name || `slot ${c.slot || '?'}`;
        list.appendChild(el);
    });
}

function renderElixir(elixir) {
    const el = document.getElementById('elixirCount');
    if (!el) return;
    
    if (elixir === null || elixir === undefined) {
        el.textContent = '-';
        el.style.color = '#888';
    } else {
        el.textContent = elixir;
        el.style.color = '#9b59b6';
    }
}

function renderStatus(status, win_detection) {
    const el = document.getElementById('matchStatus');
    if (!el) return;
    
    // Handle null/None values
    if (status === null || status === undefined) {
        el.textContent = 'No status data';
        el.style.color = '#888';
        return;
    }
    
    // status from server: 'won'|'lost'|'ongoing'
    // Note: Updates continue flowing regardless of win/loss state
    if (status === 'won') {
        el.textContent = 'Won - Match Won';
        el.style.color = '#2ecc71';
    } else if (status === 'lost') {
        el.textContent = 'Lost - Match Lost';
        el.style.color = '#e74c3c';
    } else {
        el.textContent = 'Ongoing - Match in progress';
        el.style.color = '#ffffff';
    }
}
