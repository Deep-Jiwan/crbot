import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# import the grid mapper to convert pixel positions to grid coordinates
from grid_mapper import map_json_to_grid

app = FastAPI()

# Define the website directory
website_dir = os.path.join(os.path.dirname(__file__), 'website')
# Mount CSS and JS subdirectories
app.mount('/css', StaticFiles(directory=os.path.join(website_dir, 'css')), name='css')
app.mount('/js', StaticFiles(directory=os.path.join(website_dir, 'js')), name='js')

# Serve index.html at root
@app.get('/')
async def root():
    index_file = os.path.join(website_dir, 'index.html')
    return FileResponse(index_file)


# Visualize endpoint: accepts JSON payload from frontend, maps pixel positions
# to grid coordinates using map_json_to_grid, extracts labeled locations and
# returns them as JSON. Primary method is POST; a GET returns an example
# payload and instructions.
@app.post('/visualize')
async def visualize(data: dict):
    """Accept a JSON body, map pixel positions to grid coordinates,
    and return the list of location objects.

    Expected input: the same game data structure used elsewhere (troops,
    cards_in_hand, etc.).
    """
    try:
        mapped = map_json_to_grid(data)
    except Exception as e:
        # If mapping fails, return a 400 with the error
        raise HTTPException(status_code=400, detail=str(e))

    locations = []
    # Extract positions from troops
    for item in mapped.get('troops', []):
        if 'x' in item and 'y' in item:
            locations.append({
                'label': item.get('type') or item.get('name') or 'troop',
                'x': item['x'],
                'y': item['y'],
            })

    # Extract positions from cards_in_hand if they have x/y
    for card in mapped.get('cards_in_hand', []):
        if 'x' in card and 'y' in card:
            label = card.get('name') or f"card_slot_{card.get('slot', '?')}"
            locations.append({
                'label': label,
                'x': card['x'],
                'y': card['y'],
            })

    return {'locations': locations}


# Health check endpoint
@app.get('/healthz')
def health_check():
    return {'status': 'ok'}

if __name__ == '__main__':
    import uvicorn, sys
    # Use PORT environment variable or default to 8000
    port = int(os.environ.get('PORT', 8002))
    try:
        uvicorn.run('server:app', host='0.0.0.0', port=port, reload=True)
    except PermissionError:
        print(f"Permission denied binding to port {port}. Try running with a different PORT or elevated permissions.")
        sys.exit(1)
    except OSError as e:
        print(f"Failed to bind to port {port}: {e}")
        sys.exit(1)