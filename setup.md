# Setup Guide

This guide will help you get the Context Budget Optimizer up and running on your machine.

## Prerequisites

Before you start, make sure you have:
- Python 3.8 or higher installed
- A Mistral API key (get one from https://console.mistral.ai/)
- Git installed (if cloning from GitHub)

## Installation

### Option 1: Using Docker (Recommended)

If you have Docker installed, this is the easiest way to run the project.

1. Clone the repository:
```bash
git clone <repository-url>
cd contextllm
```

2. Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-small
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

To stop the application, press `Ctrl+C` or run:
```bash
docker-compose down
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd contextllm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-small
```

Replace `your_api_key_here` with your actual Mistral API key.

## Configuration

The project uses a `config.yaml` file for settings. You can modify it to adjust:
- Embedding model
- Chunk size and overlap
- Default token budget
- Mistral model selection

Most settings work fine with defaults, but you can tweak them if needed.

## Running the Application

### Step 1: Ingest Documents

Before you can query the system, you need to ingest some documents. Use the CLI:

```bash
python -m contextllm.main ingest path/to/document1.txt path/to/document2.pdf
```

You can ingest multiple files at once. Supported formats are `.txt` and `.pdf`.

Example:
```bash
python -m contextllm.main ingest examples/sample_document.txt
```

### Step 2: Start the Web Server

Run the server:
```bash
python -m contextllm.main server
```

Or if using Docker, it's already running.

### Step 3: Access the Web Interface

Open your browser and go to:
```
http://localhost:8000
```

You should see the web interface where you can:
- Enter queries
- Adjust token budget with the slider
- View answers and optimization statistics
- See which chunks were selected or excluded

## Using the CLI

You can also use the command line interface instead of the web UI:

### Query the system:
```bash
python -m contextllm.main query "Your question here" --budget 2000
```

### Query with explanation:
```bash
python -m contextllm.main query "Your question" --budget 2000 --explain
```

## Environment Variables

The `.env` file should contain:

```
MISTRAL_API_KEY=your_actual_api_key
MISTRAL_MODEL=mistral-small
```

Available Mistral models:
- `mistral-tiny` (fastest, cheapest)
- `mistral-small` (balanced, recommended)
- `mistral-medium` (better quality)
- `mistral-large-latest` (best quality)

## Project Structure

- `contextllm/` - Main Python package
- `frontend/` - Web interface files
- `data/` - Generated data (vector DB, cache, logs)
- `examples/` - Sample documents
- `config.yaml` - Configuration file
- `.env` - Environment variables (create this)

## Troubleshooting

**Error: API key not found**
- Make sure you created the `.env` file
- Check that `MISTRAL_API_KEY` is set correctly
- Restart the server after creating/modifying `.env`

**Error: No documents found**
- Make sure you've ingested at least one document
- Check that the ingestion command completed successfully

**Error: Port already in use**
- Change the port: `python -m contextllm.main server --port 8001`
- Or stop whatever is using port 8000

**Docker build fails**
- Make sure Docker is running
- Check that you have enough disk space
- Try: `docker-compose build --no-cache`

## Data Storage

The application stores data in the `data/` directory:
- `data/vector_db/` - ChromaDB vector database
- `data/cache/` - Cached embeddings and token counts
- `data/metadata.db` - SQLite database with query history
- `data/app.log` - Application logs

You can delete the `data/` folder to start fresh, but you'll need to re-ingest all documents.

## Next Steps

1. Ingest some documents using the CLI
2. Start the web server
3. Try asking questions in the web interface
4. Experiment with different token budgets to see how it affects chunk selection
5. Check the optimization statistics and charts to understand the system's decisions

That's it. You should be up and running now.
