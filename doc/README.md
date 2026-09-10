# Building documentation locally

## Prerequisites

- The west workspace must be initialized and the west manifest must point to
  `sdk-edge-ai`. See `.west/config`.
- Doxygen must be installed (`sudo apt install doxygen`).

## Using uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the Python version, virtual
environment, and dependencies automatically from `pyproject.toml`.

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Build the docs:

```bash
cd sdk-edge-ai
uv sync
uv run make -C doc html
```

## Without uv

Create a virtual environment and install the dependencies manually:

```bash
cd sdk-edge-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r doc/requirements.txt west python-dotenv
make -C doc html
```

## Output

The generated HTML pages are placed in `doc/_build/html/`. Open
`doc/_build/html/index.html` in a browser to view the documentation.
