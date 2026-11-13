PYTHON ?= python3
NVCC ?= nvcc
CXX ?= g++
GPU_ARCH ?= native
GPU_ID ?= 0
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(GPU_ARCH)
CXXFLAGS ?= -O3 -march=native -fopenmp
OPT := -DBLOCK_SIZE=1024 -DNEGATIVE_LINEAR_STEPS -DLAZY_SMALL_FILTER
SRC := src
CUDA_BINS := $(SRC)/search15_d19_fast $(SRC)/search15_d21_fast \
	$(SRC)/search15_b1024_signed $(SRC)/search15_balanced $(SRC)/search15_direct_signed
CPU_BINS := $(SRC)/cpu15_avx $(SRC)/cpu15_shared $(SRC)/cpu15_shared_a5

.PHONY: all audit validate status selftest selftest-gpu selftest-cpu

all: $(CUDA_BINS) $(CPU_BINS)

$(SRC)/search15_d19_fast: $(SRC)/search15_period_fast.cu
	$(NVCC) $(NVCCFLAGS) -DDIGITS=19 -DSTEP=5 -DPERIOD=823045260674080876ULL $(OPT) $< -o $@

$(SRC)/search15_d21_fast: $(SRC)/search15_period_fast.cu
	$(NVCC) $(NVCCFLAGS) -DDIGITS=21 -DSTEP=5 -DPERIOD=823045260674080876ULL -DT_LIMIT=243 $(OPT) $< -o $@

$(SRC)/search15_b1024_signed: $(SRC)/search15_block.cu
	$(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

$(SRC)/search15_balanced: $(SRC)/search15_balanced.cu $(SRC)/search15_prune.cu
	$(NVCC) $(NVCCFLAGS) -DDIGITS=19 -DSTEP=5 -DPERIOD=823045260674080876ULL $(OPT) $< -o $@

$(SRC)/search15_direct_signed: $(SRC)/search15_direct_odd.cu
	$(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

$(SRC)/cpu15_avx: $(SRC)/cpu15_avx.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(SRC)/cpu15_shared: $(SRC)/cpu15_shared.cpp $(SRC)/cpu15_avx.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/cpu15_shared.cpp -o $@

$(SRC)/cpu15_shared_a5: $(SRC)/cpu15_shared.cpp $(SRC)/cpu15_avx.cpp
	$(CXX) $(CXXFLAGS) $(SRC)/cpu15_shared.cpp -o $@

audit:
	cd $(SRC) && $(PYTHON) audit15_final.py

validate:
	cd $(SRC) && $(PYTHON) validate15.py
	cd $(SRC) && $(PYTHON) validate15_coset4.py

status:
	cd $(SRC) && $(PYTHON) status15.py

selftest: selftest-gpu selftest-cpu

selftest-gpu: $(CUDA_BINS)
	cd $(SRC) && ./search15_d19_fast selftest $(GPU_ID)
	cd $(SRC) && ./search15_d21_fast selftest $(GPU_ID)
	cd $(SRC) && ./search15_b1024_signed selftest $(GPU_ID)
	cd $(SRC) && ./search15_balanced selftest $(GPU_ID)
	cd $(SRC) && ./search15_direct_signed selftest $(GPU_ID)

selftest-cpu: $(CPU_BINS)
	cd $(SRC) && ./cpu15_shared selftest

