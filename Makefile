# Makefile
install:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run-gsm8k:
	python -m src.main --config config.yaml --dataset gsm8k

run-strategyqa:
	python -m src.main --config config.yaml --dataset strategyqa