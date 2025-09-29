<!-- PROJECT LOGO -->
<h3 align="center">Clash Royale Bot</h3>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#docker">Docker</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository provides a modular pipeline for a Clash Royale helper bot built around real‑time frame streaming and lightweight computer vision. Components communicate via ZeroMQ topics, enabling you to run only what you need:

- Publisher: captures webcam frames and publishes them as `frame|<jpg_bytes>`.
- Elixir Counter: subscribes to frames, samples pixel colors where elixir pips appear, optionally annotates the frame, and publishes `ecount|<int>`.
- Card Detection (optional): crops the four card slots and queries a Roboflow workflow; publishes JSON predictions.
- Troop Detection (optional): example Roboflow HTTP workflow invoker that logs detections.
- Demo Sub/Pub: simple ZeroMQ example for learning and testing.

*(Disclaimer: This project is not affiliated with Supercell. Use at your own risk—automated gameplay may violate Clash Royale's Terms of Service.)*

### Built With

* [![Docker][Docker.com]][Docker-url]
* [![Roboflow][Roboflow.com]][Roboflow-url]
* [![Python][Python.org]][Python-url]



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Windows 10/11
* Python 3.11+
* Bluestacks (or other frame source)
* Optional: Docker, Roboflow account

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Deep-Jiwan/crbot.git
   cd crbot
   ```
2. Create and activate a virtual environment
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```
3. Install dependencies per component you plan to run
   ```powershell
   pip install -r publisher/requirements.txt
   pip install -r elixircount/requirements.txt
   # optional
   pip install -r demosub/requirements.txt
   ```
4. Configure environment variables in a `.env` file next to each script you run.

Key variables and defaults:

- Shared: `FRAME_WIDTH=1080`, `FRAME_HEIGHT=1920`
- Publisher: binds `tcp://*:5550`, camera index set in code (default `9`)
- Elixir Counter: `ZMQ_ADDRESS=tcp://localhost:5550`, `PUB_PORT=5551`, `ANNOTATE=True`, position/color thresholds
- Card Detection: `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKFLOW_ID`, `ZMQ_PUB_ADDRESS=tcp://*:5552`
- Troop Detection: `ROB_FLOW_API_KEY`, `ROB_FLOW_WORKFLOW_URL`, `ZMQ_SUB_ADDRESS`


### Roboflow: Fork Workflows and Set API Keys

1. Create a Roboflow account and workspace, then obtain your private API key.
   
   ```bash
   # example .env entries
   ROBOFLOW_API_KEY=your_actual_api_key_here
   ROBOFLOW_WORKFLOW_ID=your_card_workflow_id_or_project_slug
   ROB_FLOW_API_KEY=your_actual_api_key_here
   ROB_FLOW_WORKFLOW_URL=https://api.roboflow.com/... # troop workflow endpoint
   ```

2. Fork the example workflows to your workspace (optional but recommended):
   - [Troop Detection Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTEx3TjlnOEduenBjWmVYSktKYzEiLCJ3b3Jrc3BhY2VJZCI6Ik5vVUlkM3gyYWRSU0tqaURrM0ZMTzlBSmE1bzEiLCJ1c2VySWQiOiJOb1VJZDN4MmFkUlNLamlEazNGTE85QUphNW8xIiwiaWF0IjoxNzUzODgxNTcyfQ.-ZO7pqc3mBX6W49-uThUSBLdUaCRzM9I8exfEu6-lo8)
   - [Card Detection Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMEFmeVpSQ3FSS1dhV1J5QTFGNkciLCJ3b3Jrc3BhY2VJZCI6InJtZHNiY2xlU292aEEwNm15UDFWIiwidXNlcklkIjoiTm9VSWQzeDJhZFJTS2ppRGszRkxPOUFKYTVvMSIsImlhdCI6MTc1Mzg4MjE4Mn0.ceYp4JZoNSIrDkrX2vuc9or3qVakNexseYEgacIrfLA)

3. Copy any required identifiers from your forked workflows:
   - For Card Detection: set `ROBOFLOW_WORKFLOW_ID` to the project/workflow identifier used by your Roboflow workspace.
   - For Troop Detection: set `ROB_FLOW_WORKFLOW_URL` to the HTTP workflow endpoint URL.

4. (Optional) Roboflow Inference Server
   - You can run the Roboflow Inference Server locally if your workflow supports it:
     ```bash
     pip install inference-cli
     inference server start
     ```
   - Then point your workflow URL to the local server if applicable.



## Usage

Run each component in its own terminal.

Publisher (produces `frame|`):
```powershell
cd publisher
python publisher.py
```

Elixir Counter (consumes frames, publishes `ecount|`):
```powershell
cd elixircount
python elixir_count.py
```

Outputs:
- Latest frame: `elixircount/images/latest_frame.jpg`
- Annotated (if enabled): `elixircount/frame/latest_annotated.jpg`
- Published topic: `ecount|<int>` on TCP port 5551

Optional – Card Detection:
```powershell
cd carddetection
$env:ROBOFLOW_API_KEY = "<your_api_key>"
$env:ROBOFLOW_WORKFLOW_ID = "<your_workflow_id>"
python card_detection.py
```

Optional – Troop Detection (HTTP workflow):
```powershell
cd troopdetection
$env:ROB_FLOW_API_KEY = "<your_api_key>"
$env:ROB_FLOW_WORKFLOW_URL = "https://api.roboflow.com/..."
python troop_detection.py
```

### BlueStacks Setup (Optional)

If you plan to capture gameplay from an Android emulator:

1. Install [BlueStacks](https://www.bluestacks.com/download.html) and open the Multi‑Instance Manager.
2. Create a fresh Pie 64‑bit instance and start it.
3. Install Clash Royale from Google Play Store in the instance.
4. (Optional) Disable ads: Settings → Preferences → Allow BlueStacks to show Ads during gameplay (disable).
5. Open Clash Royale and position/resize the window to fit your capture workflow.

Window positioning example:

![BlueStacks-window-tutorial](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3k2enMwY3E4cHJ0MDhnbmg1NnhsaDI3bGhmazJ4aXlxczFkamFxeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/y8yXKqwN40cdcr4yR5/giphy.gif)

ZeroMQ topics and ports:
- Publisher → Subscribers: `tcp://*:5550`, topic `frame|`
- Elixir Counter publishes: `tcp://*:5551`, topic `ecount|`
- Card Detection publishes: `ZMQ_PUB_ADDRESS` (default `tcp://*:5552`), JSON payload


## Docker

Elixir Counter:
```powershell
cd elixircount
docker build -t elixir_counter:latest .
docker run --name elixir_counter -p 5550:5550 -p 5551:5551 -e ZMQ_ADDRESS="tcp://host.docker.internal:5550" -d elixir_counter:latest
```

Publisher:
```powershell
cd publisher
docker build -t frame_publisher:latest .
docker run --network=host --name frame_publisher -d frame_publisher:latest
```

Custom network:
```powershell
docker network create cr_network
docker run --name frame_publisher --network cr_network -d frame_publisher:latest
docker run --name elixir_counter --network cr_network -e ZMQ_ADDRESS="tcp://frame_publisher:5550" -d elixir_counter:latest
```


## Contributing

Contributions are welcome! Please fork the repository and open a pull request. If you find a bug or want to request a feature, open an issue.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## License

Distributed under the project license. See `LICENSE` for more information.


## Contact

Open an issue for questions and support.


## Acknowledgments

* ZeroMQ community and examples
* Roboflow documentation and tooling
* Best README Template inspiration


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Docker.com]: https://img.shields.io/badge/Docker-yellow?style=for-the-badge&logo=Docker&link=https%3A%2F%2Fwww.docker.com%2F
[Docker-url]: https://www.docker.com/
[Roboflow.com]: https://img.shields.io/badge/Roboflow-gray?style=for-the-badge&logo=roboflow&link=https%3A%2F%2Fwww.roboflow.com%2F
[Roboflow-url]: https://www.roboflow.com/
[Python.org]: https://img.shields.io/badge/Python-white?style=for-the-badge&logo=python&link=https%3A%2F%2Fwww.python.org%2F
[Python-url]: https://www.python.org/
