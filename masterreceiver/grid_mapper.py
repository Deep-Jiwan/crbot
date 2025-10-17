import json

def map_json_to_grid(data, canvas_width=1080, canvas_height=1920, rows=32, cols=18):
    """
    Map pixel locations in a JSON object to grid coordinates.

    Parameters:
    - data: dict or JSON string representing a single game data line.
    - canvas_width: width of the pixel canvas (default 1080).
    - canvas_height: height of the pixel canvas (default 1920).
    - rows: number of grid rows (default 31).
    - cols: number of grid columns (default 18).

    Returns:
    - dict: same JSON structure with x and y values replaced by grid indices.
    """
    # Parse JSON if input is a raw string
    if isinstance(data, str):
        data = json.loads(data)
        
    width_bias = 0  # adjust if needed
    height_bias = 1920-1600

    cell_width = (canvas_width-(width_bias)) / cols
    cell_height = (canvas_height-(height_bias)) / rows

    # Map locations for items that have x and y
    for item in data.get('troops', []) + data.get('cards_in_hand', []):
        # Only map if both x and y present
        if 'x' in item and 'y' in item:
            px = item['x']
            py = item['y']
            grid_x = int(px / cell_width)
            grid_y = int(py / cell_height)
            item['x'] = grid_x
            item['y'] = grid_y

    return data


def map_grid_to_pixel(col, row, canvas_width=1080, canvas_height=1920, rows=32, cols=18):
    """
    Convert grid coordinates (col, row) where col in [0, cols-1], row in [0, rows-1]
    to pixel coordinates (x, y) in the canvas coordinate space.
    """
    # Validate inputs
    if not (0 <= col < cols) or not (0 <= row < rows):
        raise ValueError(f"Grid coordinates out of range: col={col}, row={row}")

    # Calibrated affine transform (selected and frozen):
    # px = a*col + b*row + c
    # py = d*col + e*row + f
    a = 61.847454
    b = -1.203915
    c = 30.395021
    d = -1.582325
    e = 49.169769
    f = 45.042014

    px = a * col + b * row + c
    py = d * col + e * row + f

    # Clip to valid pixel bounds
    px = max(0.0, min(px, canvas_width - 1))
    py = max(0.0, min(py, canvas_height - 1))

    return px, py

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <jsonl_file_path>")
        sys.exit(1)
    jsonl_path = sys.argv[1]
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    mapped = map_json_to_grid(line)
                    # Pretty-print each mapped item
                    for item in mapped.get('troops', []):
                        print(f"{item.get('type')} : ({item.get('x')},{item.get('y')})")
                    for card in mapped.get('cards_in_hand', []):
                        # Cards may have x, y if located; skip otherwise
                        if 'x' in card and 'y' in card:
                            print(f"{card.get('name')} : ({card.get('x')},{card.get('y')})")
                    # Blank line between records
                    print()
                except json.JSONDecodeError:
                    # skip invalid JSON lines
                    continue
    except FileNotFoundError:
        print(f"File not found: {jsonl_path}")
        sys.exit(1)

