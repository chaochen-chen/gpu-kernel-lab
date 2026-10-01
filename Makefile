.PHONY: install test bench lint format standalone clean

# Usage:
#   make test                 run all tests
#   make test K=triton_online run tests whose name matches K
#   make bench                run all benchmarks (CSV -> results/)

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v $(if $(K),-k $(K),)

bench:
	python -m benchmarks.bench_softmax
	python -m benchmarks.bench_rmsnorm

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

standalone:
	$(MAKE) -C cuda_standalone

clean:
	rm -rf build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
