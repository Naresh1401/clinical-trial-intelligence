.PHONY: install data test lint api dashboard clean all

# ─── Configuration ───────────────────────────────────────────────────
PYTHON   := python3
PIP      := pip3
API_HOST := 0.0.0.0
API_PORT := 8000

# ─── Targets ─────────────────────────────────────────────────────────

install:  ## Install all dependencies
	$(PIP) install -r requirements.txt

data:  ## Generate synthetic clinical trial data
	$(PYTHON) data/generate_synthetic_data.py

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src/ tests/ dashboard/

api:  ## Start FastAPI server
	$(PYTHON) -m uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

dashboard:  ## Start Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

clean:  ## Remove generated data and cache
	rm -f data/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

all: install data test  ## Full setup: install → generate data → test
