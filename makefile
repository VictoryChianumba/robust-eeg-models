.PHONY: venv install clean help

help:
	@echo "available commands:"
	@echo "  make venv     - Create a virtual environment"
	@echo "  make install   - Install project dependencies"
	@echo "  make clean    - Remove the virtual environment"

venv:
	python -m venv .venv



install: venv 
	./.venv/bin/pip install --upgrade pip 
	./.venv/bin/pip install -r requirements.txt

clean:
	rm -rf .venv