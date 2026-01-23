# Examples

This directory contains example scripts and sample data for the Context Budget Optimizer.

## Files

- `example_usage.py`: Comprehensive usage examples showing how to use the system
- `sample_document.txt`: Sample document for testing ingestion and querying

## Running Examples

1. First, ingest the sample document:
```bash
python -m contextllm.main ingest examples/sample_document.txt
```

2. Then run the example script:
```bash
python examples/example_usage.py
```

Or use the CLI to query:
```bash
python -m contextllm.main query "What is the Context Budget Optimizer?" --budget 2000 --explain
```

## Example Workflows

### Basic Workflow

1. **Ingest documents**
   ```bash
   python -m contextllm.main ingest doc1.txt doc2.pdf
   ```

2. **Query with default budget**
   ```bash
   python -m contextllm.main query "Your question here"
   ```

3. **Query with custom budget and explanation**
   ```bash
   python -m contextllm.main query "Your question" --budget 3000 --explain
   ```

### Comparing Different Budgets

Use the Python API to compare how different budgets affect results:

```python
from contextllm.retrieval.searcher import search_chunks
from contextllm.optimization.optimizer import optimize_context

query = "Your question"
chunks = search_chunks(query, top_k=50)

for budget in [1000, 2000, 4000]:
    result = optimize_context(chunks, budget=budget)
    print(f"Budget {budget}: {len(result['selected_chunks'])} chunks selected")
```

### Web Interface

Start the server and use the interactive web interface:

```bash
python -m contextllm.main server
```

Then visit `http://localhost:8000` in your browser.
