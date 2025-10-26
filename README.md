# README.md

# Entropy-Based Reasoning Path Selection

Implementation of entropy-based path selection for chain-of-thought reasoning using LZW compression ratios.

## Project Structure
```
entropy-path-select/
├── README.md
├── requirements.txt
├── .env.example
├── Makefile
├── config.yaml
├── src/                    # All Python source files
│   ├── main.py            # Main entry point
│   ├── config.py          # Configuration loading
│   ├── io_utils.py        # I/O utilities
│   ├── templates.py       # Prompt templates
│   ├── loaders.py         # Dataset loaders
│   ├── ollama_client.py   # Ollama API client
│   ├── generation.py      # Text generation functions
│   ├── lzw.py             # LZW compression
│   ├── selection.py       # Path selection logic
│   ├── parse.py           # Answer parsing
│   └── metrics.py         # Evaluation metrics
└── tests/                 # Unit tests
```

## Setup

1. Install Ollama: https://ollama.com/
2. Pull a model:
```bash
   ollama pull llama3.1:8b-instruct
```
3. Install dependencies:
```bash
   make install
   # or manually:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
```
4. Configure environment:
```bash
   cp .env.example .env
```

## Usage

Run on GSM8K:
```bash
python -m src.main --config config.yaml --dataset gsm8k
```

Run on StrategyQA:
```bash
python -m src.main --config config.yaml --dataset strategyqa
```

Or use Makefile:
```bash
make run-gsm8k
make run-strategyqa
```

## Configuration

Edit `config.yaml` to adjust:
- `reasoning.K`: Number of reasoning paths (default: 8)
- `reasoning.sweet_spot`: Compression ratio range (default: 0.30-0.60)
- `model.temperature`: Sampling temperature (default: 0.8)
- `dataset.limit`: Number of examples to process

## Output

Results are saved to `outputs/<dataset>/`:
- `results.json`: Accuracy metrics
- `traces.jsonl`: Detailed traces of all reasoning paths

## How It Works

1. Generate K diverse reasoning paths using temperature sampling
2. Compute LZW compression ratio for each path
3. Select the path with compression ratio closest to the sweet spot
4. Compare against greedy decoding and self-consistency baselines

## Testing
```bash
python -m pytest tests/
```

## Notes

- Supports any Ollama model: llama3.1, mistral, qwen2.5, etc.
- Increase `max_tokens` (768-1024) for longer reasoning
- Adjust temperature for more/less diversity in paths
- The sweet spot range may need calibration per model

## License

MIT