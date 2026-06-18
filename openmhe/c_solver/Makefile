# Fast sliding-window driver linked against Acados-generated OpenMHE code.

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

CODEGEN_DIR ?= $(abspath ../../c_generated_code)
# Escape spaces so paths work in prerequisite lists (GNU make splits on whitespace).
empty :=
space := $(empty) $(empty)
esc = $(subst $(space),\ ,$1)
CODEGEN_HDR = $(call esc,$(CODEGEN_DIR))/acados_solver_openmhe_mhe.h
ACADOS_DIR ?= $(abspath ../../../acados)
OPT_CFLAGS ?= -O3 -march=native -ffast-math

INCLUDE_PATH = $(ACADOS_DIR)/include
LIB_PATH = $(ACADOS_DIR)/lib

# Match the BLASFEO target used to build libblasfeo in ACADOS_DIR (override via env).
BLASFEO_TARGET_H := $(INCLUDE_PATH)/blasfeo/include/blasfeo_target.h
BLASFEO_TARGET_DEFINE ?= $(shell awk '/^#define TARGET_(X64|X86|ARM|GENERIC)/ {print $$2; exit}' "$(BLASFEO_TARGET_H)" 2>/dev/null)

CPPFLAGS += -I"$(INCLUDE_DIR)"
CPPFLAGS += -I"$(CODEGEN_DIR)"
CPPFLAGS += -I"$(INCLUDE_PATH)"
CPPFLAGS += -I"$(INCLUDE_PATH)/acados"
CPPFLAGS += -I"$(INCLUDE_PATH)/blasfeo/include"
CPPFLAGS += -I"$(INCLUDE_PATH)/hpipm/include"

CFLAGS = -fPIC -std=c99 -Wall -Wextra $(OPT_CFLAGS)
ifneq ($(BLASFEO_TARGET_DEFINE),)
CFLAGS += -D$(BLASFEO_TARGET_DEFINE)
endif

ifdef OPENMHE_PROFILE
CFLAGS += -DOPENMHE_PROFILE
endif

LDFLAGS += -L"$(CODEGEN_DIR)" -L"$(LIB_PATH)"
LDFLAGS += -Wl,-rpath,"$(CODEGEN_DIR)" -Wl,-rpath,"$(LIB_PATH)"
LDLIBS += -lacados_ocp_solver_openmhe_mhe -lacados -lhpipm -lblasfeo -lm

RUN_LIB = $(BUILD_DIR)/libopenmhe_mhe_run.so
RUN_OBJ = $(BUILD_DIR)/run_loop.o $(BUILD_DIR)/lti_fast.o $(BUILD_DIR)/filter_arrival.o

TEST_CFLAGS = $(CPPFLAGS) $(CFLAGS)

.PHONY: all clean test codegen_lib

all: codegen_lib $(RUN_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

codegen_lib:
	@test -f "$(CODEGEN_DIR)/libacados_ocp_solver_openmhe_mhe.so" || \
		$(MAKE) -C "$(CODEGEN_DIR)" ocp_shared_lib CFLAGS="$(CFLAGS)"

$(BUILD_DIR)/run_loop.o: $(SRC_DIR)/run_loop.c $(INCLUDE_DIR)/run_loop.h $(INCLUDE_DIR)/lti_fast.h $(INCLUDE_DIR)/profile.h $(CODEGEN_HDR) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(SRC_DIR)/run_loop.c

$(BUILD_DIR)/lti_fast.o: $(SRC_DIR)/lti_fast.c $(INCLUDE_DIR)/lti_fast.h $(INCLUDE_DIR)/run_loop.h $(CODEGEN_HDR) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(SRC_DIR)/lti_fast.c

$(BUILD_DIR)/filter_arrival.o: $(SRC_DIR)/filter_arrival.c $(INCLUDE_DIR)/filter_arrival.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(SRC_DIR)/filter_arrival.c

$(RUN_LIB): $(RUN_OBJ) codegen_lib
	$(CC) -shared $(RUN_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

test: $(BUILD_DIR)/test_filter $(BUILD_DIR)/test_inv $(BUILD_DIR)/test_arrival_inv

$(BUILD_DIR)/test_filter: test/test_filter.c $(BUILD_DIR)/filter_arrival.o | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -o $@ test/test_filter.c $(BUILD_DIR)/filter_arrival.o -lm

$(BUILD_DIR)/test_inv: test/test_inv.c $(BUILD_DIR)/filter_arrival.o | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -o $@ test/test_inv.c $(BUILD_DIR)/filter_arrival.o -lm

$(BUILD_DIR)/test_arrival_inv: test/test_arrival_inv.c $(BUILD_DIR)/filter_arrival.o | $(BUILD_DIR)
	$(CC) $(TEST_CFLAGS) -o $@ test/test_arrival_inv.c $(BUILD_DIR)/filter_arrival.o -lm

clean:
	$(RM) $(RUN_OBJ) $(RUN_LIB)
	$(RM) $(BUILD_DIR)/test_filter $(BUILD_DIR)/test_inv $(BUILD_DIR)/test_arrival_inv
