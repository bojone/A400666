PYTHON ?= python3
NVCC ?= nvcc
CXX ?= g++
GPU_ARCH ?= native
GPU_ID ?= 0
SRC := src
FILTER_DIGITS := 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
GPU_DIGITS := 21 22 23 24

.PHONY: all audit validate selftest run status

all: $(SRC)/search17 $(SRC)/filter17

$(SRC)/search17: $(SRC)/search17.cu
	$(NVCC) -O3 -std=c++17 -arch=$(GPU_ARCH) $< -o $@

$(SRC)/filter17: $(SRC)/filter17.cpp
	$(CXX) -O3 -march=native -fopenmp $< -o $@

audit:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify17.py --audit
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify17_matrix.py

validate:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify17.py
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) verify17_matrix.py

selftest: all
	cd $(SRC) && ./search17 selftest $(GPU_ID)
	cd $(SRC) && ./filter17 selftest

run: all
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) filter17.py --digits $(FILTER_DIGITS)
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) run17.py --digits $(GPU_DIGITS)
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) finalize17.py

status:
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) filter17.py --status
	cd $(SRC) && PYTHONDONTWRITEBYTECODE=1 $(PYTHON) run17.py --status
