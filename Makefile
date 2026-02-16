.PHONY: setup setup-python setup-rust build test test-fast clean help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MATURIN := $(VENV)/bin/maturin

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: setup-python setup-rust ## Full setup: Python venv + Rust crates

setup-python: ## Create venv and install Python deps
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install maturin

setup-rust: ## Build and install Rust crates into the venv
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	command -v cargo >/dev/null 2>&1 || { echo "Rust toolchain not found. Install from https://rustup.rs"; exit 1; }; \
	echo "Building poker-engine (hand eval + EV)..."; \
	cd poker-engine && $(MATURIN) develop --release && cd ..; \
	echo "Building rust_cfr (CFR+ engine)..."; \
	cd rust_cfr && $(MATURIN) develop --release; \
	echo "Rust crates installed."

build: ## Rebuild Rust crates (release mode)
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	cd poker-engine && $(MATURIN) develop --release && cd ..; \
	cd rust_cfr && $(MATURIN) develop --release

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-fast: ## Run tests excluding slow ones
	$(PYTHON) -m pytest tests/ -v -m "not slow"

clean: ## Remove build artifacts
	rm -rf poker-engine/target rust_cfr/target
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
