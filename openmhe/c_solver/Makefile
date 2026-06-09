# Fast sliding-window driver linked against Acados-generated OpenMHE code.

CODEGEN_DIR ?= $(abspath ../../c_generated_code)
CODEGEN_HDR = $(CODEGEN_DIR)/acados_solver_openmhe_mhe.h
ACADOS_DIR ?= $(abspath ../../../acados)
OPT_CFLAGS ?= -O3 -march=native -ffast-math

INCLUDE_PATH = $(ACADOS_DIR)/include
LIB_PATH = $(ACADOS_DIR)/lib

# Match the BLASFEO target used to build libblasfeo in ACADOS_DIR (override via env).
BLASFEO_TARGET_H := $(INCLUDE_PATH)/blasfeo/include/blasfeo_target.h
BLASFEO_TARGET_DEFINE ?= $(shell awk '/^#define TARGET_(X64|X86|ARM|GENERIC)/ {print $$2; exit}' $(BLASFEO_TARGET_H) 2>/dev/null)

CPPFLAGS += -I$(CODEGEN_DIR)
CPPFLAGS += -I$(INCLUDE_PATH)
CPPFLAGS += -I$(INCLUDE_PATH)/acados
CPPFLAGS += -I$(INCLUDE_PATH)/blasfeo/include
CPPFLAGS += -I$(INCLUDE_PATH)/hpipm/include

CFLAGS = -fPIC -std=c99 -Wall -Wextra $(OPT_CFLAGS)
ifneq ($(BLASFEO_TARGET_DEFINE),)
CFLAGS += -D$(BLASFEO_TARGET_DEFINE)
endif

ifdef OPENMHE_PROFILE
CFLAGS += -DOPENMHE_PROFILE
endif

LDFLAGS += -L$(CODEGEN_DIR) -L$(LIB_PATH)
LDFLAGS += -Wl,-rpath,$(CODEGEN_DIR) -Wl,-rpath,$(LIB_PATH)
LDLIBS += -lacados_ocp_solver_openmhe_mhe -lacados -lhpipm -lblasfeo -lm

RUN_LIB = libopenmhe_mhe_run.so
RUN_OBJ = run_loop.o lti_fast.o

TEST_CFLAGS = $(CPPFLAGS) $(CFLAGS) -I.

.PHONY: all clean test codegen_lib

all: codegen_lib $(RUN_LIB)

codegen_lib:
	@test -f $(CODEGEN_DIR)/libacados_ocp_solver_openmhe_mhe.so || \
		$(MAKE) -C $(CODEGEN_DIR) ocp_shared_lib CFLAGS="$(CFLAGS)"

run_loop.o: run_loop.c run_loop.h lti_fast.h profile.h $(CODEGEN_HDR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ run_loop.c

lti_fast.o: lti_fast.c lti_fast.h run_loop.h $(CODEGEN_HDR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ lti_fast.c

filter_arrival.o: filter_arrival.c filter_arrival.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ filter_arrival.c

$(RUN_LIB): $(RUN_OBJ) codegen_lib
	$(CC) -shared $(RUN_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

test: test/test_filter test/test_inv

test/test_filter: test/test_filter.c filter_arrival.o
	$(CC) $(TEST_CFLAGS) -o $@ test/test_filter.c filter_arrival.o -lm

test/test_inv: test/test_inv.c filter_arrival.o
	$(CC) $(TEST_CFLAGS) -o $@ test/test_inv.c filter_arrival.o -lm

clean:
	$(RM) $(RUN_OBJ) $(RUN_LIB)
	$(RM) test/test_filter test/test_inv
