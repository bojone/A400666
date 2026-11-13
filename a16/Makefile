PYTHON ?= python3
NVCC ?= nvcc
GPU_ARCH ?= native
GPU_ID ?= 0
SRC := src

.PHONY: all audit validate selftest run status

all: $(SRC)/search16

$(SRC)/search16: $(SRC)/search16.cu
	$(NVCC) -O3 -std=c++17 -arch=$(GPU_ARCH) $< -o $@

audit:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify16.py --audit

validate:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify16.py

selftest: $(SRC)/search16
	cd $(SRC) && ./search16 selftest $(GPU_ID)

run: $(SRC)/search16
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) run16.py --through 28

status:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) run16.py --status
