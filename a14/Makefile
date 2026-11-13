PYTHON ?= python3
NVCC ?= nvcc
GPU_ARCH ?= native
GPU_ID ?= 0
SRC := src

.PHONY: all audit verify-first13 selftest run

all: $(SRC)/search $(SRC)/verify_oeis

$(SRC)/search: $(SRC)/search.cu
	$(NVCC) -O3 -std=c++17 -arch=$(GPU_ARCH) $< -o $@

$(SRC)/verify_oeis: $(SRC)/verify_oeis.cu
	$(NVCC) -O3 -std=c++17 -arch=$(GPU_ARCH) $< -o $@

audit:
	cd $(SRC) && $(PYTHON) audit_results.py

verify-first13:
	cd $(SRC) && $(PYTHON) verify_first13.py --audit

selftest: all
	cd $(SRC) && ./search selftest $(GPU_ID)
	cd $(SRC) && ./verify_oeis selftest $(GPU_ID)

run: $(SRC)/search
	cd $(SRC) && $(PYTHON) run_all.py
	cd $(SRC) && $(PYTHON) audit_results.py
