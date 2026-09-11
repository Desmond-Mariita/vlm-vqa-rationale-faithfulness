PYTHON ?= python3
.PHONY: check tables figures thesis-status
check:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/reproduce.py
tables:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/build_public_tables.py --check
figures:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/check_result_figures.py
thesis-status:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/validate_release.py --thesis-status
