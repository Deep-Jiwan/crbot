import os
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# import the grid mapper to convert pixel positions to grid coordinates
from grid_mapper import map_json_to_grid

app = FastAPI()

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the website directory
website_dir = os.path.join(os.path.dirname(__file__), 'website')
# Mount CSS and JS subdirectories
app.mount('/css', StaticFiles(directory=os.path.join(website_dir, 'css')), name='css')
app.mount('/js', StaticFiles(directory=os.path.join(website_dir, 'js')), name='js')

# Allow the website to call endpoints (helps when running on different ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve index.html at root
@app.get('/')
async def root():
    index_file = os.path.join(website_dir, 'index.html')
    return FileResponse(index_file)





def _extract_locations_from_mapped(mapped):
    locations = []
    for item in mapped.get('troops', []):
        if 'x' in item and 'y' in item:
            locations.append({
                'label': item.get('type') or item.get('name') or 'troop',
                'x': item['x'],
                'y': item['y'],
            })
    for card in mapped.get('cards_in_hand', []):
        if 'x' in card and 'y' in card:
            label = card.get('name') or f"card_slot_{card.get('slot', '?')}"
            locations.append({
                'label': label,
                'x': card['x'],
                'y': card['y'],
            })
    return locations


async def _sse_event_generator(file_path: str, poll_interval: float = 1.0):
    """Async generator that yields Server-Sent Events containing the latest
    mapped locations from the jsonl file whenever the latest line changes.
    """
    last_line = None
    # send an initial comment to encourage proxies to stream
    try:
        yield ": connected\n\n"
    except Exception:
        # ignore if client disconnected immediately
        return
    # keep running until client disconnects
    while True:
        try:
            if os.path.exists(file_path):
                # read last non-empty line
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [ln for ln in f if ln.strip()]
                    if lines:
                        line = lines[-1].strip()
                        if line != last_line:
                            last_line = line
                            try:
                                raw = json.loads(line)
                                mapped = map_json_to_grid(raw)
                                locations = _extract_locations_from_mapped(mapped)

                                # Extract cards, elixir, and win detection; derive status
                                cards = raw.get('cards_in_hand', None)
                                elixir = raw.get('elixir', None)
                                win_det = raw.get('win_detection', None)
                                if win_det is True:
                                    status = 'won'
                                elif win_det is False:
                                    status = 'lost'
                                else:
                                    status = 'ongoing'

                                payload = {
                                    'locations': locations,
                                    'cards_in_hand': cards,
                                    'elixir': elixir,
                                    'win_detection': win_det,
                                    'status': status,
                                    'raw': raw
                                }
                                # Use json.dumps with ensure_ascii=False to preserve utf-8
                                data_str = json.dumps(payload, ensure_ascii=False)
                            except Exception as e:
                                logger.exception("Failed to build SSE payload")
                                data_str = json.dumps({'error': str(e)})

                            # SSE format: data: <json>\n\n
                            try:
                                yield f"data: {data_str}\n\n"
                            except Exception:
                                # client likely disconnected
                                return
            else:
                # file not present yet; send a noop or just wait
                pass
        except Exception as e:
            # In case of unexpected error, send error event and continue
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except Exception:
                # ignore payload formation errors
                pass

        await asyncio.sleep(poll_interval)


@app.get('/stream')
def stream_game_log():
    """Server-Sent Events endpoint that streams the latest mapped game state
    (as produced from the last line of game_data_log.jsonl).
    """
    # try server dir first, fall back to parent
    server_dir = os.path.dirname(__file__)
    candidate = os.path.join(server_dir, 'game_data_log.jsonl')
    if not os.path.exists(candidate):
        candidate = os.path.join(server_dir, '..', 'game_data_log.jsonl')

    # Use StreamingResponse with text/event-stream for EventSource in browser
    return StreamingResponse(_sse_event_generator(candidate), media_type='text/event-stream')


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