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

function visualizeGameState() {
    // Read JSON from textarea, POST to /visualize, and paint returned locations
    (async () => {
        const raw = document.getElementById('gameState').value;
        let payload;
        try {
            payload = JSON.parse(raw);
        } catch (e) {
            alert('Invalid JSON in game state: ' + e.message);
            return;
        }

        try {
            const resp = await fetch('/visualize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(`Server returned ${resp.status}: ${text}`);
            }

            const data = await resp.json();
            console.log('Visualize response:', data);

            clearGrid();

            const locations = data.locations || [];
            if (locations.length === 0) {
                console.info('No locations returned to visualize.');
            }

            locations.forEach(loc => {
                const x = Number(loc.x);
                const y = Number(loc.y);
                let color = '#ffff66';
                const label = (loc.label || '').toLowerCase();
                if (label.includes('enemy')) color = '#ff6666';
                else if (label.includes('ally')) color = '#66b3ff';
                else if (label.includes('princess') || label.includes('tower')) color = '#ffd24d';

                paintCell(x, y, color);
                // annotate cell with label and coordinates for hover
                try {
                    if (cells[y] && cells[y][x]) cells[y][x].title = `${loc.label || ''} : (${x},${y})`;
                } catch (e) {
                    // ignore annotation errors
                }
            });

        } catch (err) {
            console.error('Failed to fetch /visualize:', err);
            alert('Visualization failed: ' + err.message);
        }
    })();
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
});
