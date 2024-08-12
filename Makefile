################################################################################
#  Makefile for CGFDM3D package
################################################################################

# known issues:
#  .h dependency is not included, use make cleanall

#-------------------------------------------------------------------------------
# compiler
#-------------------------------------------------------------------------------
CXX    :=  $(MPIHOME)/bin/mpicxx
GC     :=  $(CUDAHOME)/bin/nvcc 

#- debug
#CFLAGS_CUDA   := -g $(CFLAGS_CUDA)
#CPPFLAGS := -g -std=c++11 $(CPPFLAGS)
#- O3
CPPFLAGS := -O3 -std=c++11 $(CPPFLAGS)

CFLAGS_CUDA   := -O3 -arch=$(SMCODE) -std=c++11 -w -rdc=true
CFLAGS_CUDA += -I$(CUDAHOME)/include -I$(MPIHOME)/include
CFLAGS_CUDA += -I$(NETCDF)/include -I./src/lib/ -I./src/forward/ -I./src/media/ 

#- dynamic
LDFLAGS := -L$(NETCDF)/lib -lnetcdf -L$(CUDAHOME)/lib64 -lcudart -L$(MPIHOME)/lib -lmpi
LDFLAGS += -lm -arch=$(SMCODE)

skeldirs := obj
DIR_OBJ    := ./obj
#-------------------------------------------------------------------------------
# target
#-------------------------------------------------------------------------------

# special vars:
# 	$@ The file name of the target
# 	$< The names of the first prerequisite
#   $^ The names of all the prerequisites 

OBJS := cJSON.o sacLib.o fdlib_mem.o fdlib_math.o  \
		media_utility.o \
		media_layer2model.o \
		media_grid2model.o \
		media_bin2model.o \
		media_geometry3d.o \
		media_read_file.o \
		fd_t.o par_t.o interp.o mympi_t.o alloc.o  \
		gd_t.o md_t.o wav_t.o \
		bdry_t.o src_t.o io_funcs.o \
		blk_t.o cuda_common.o \
		drv_rk_curv_col.o \
		sv_curv_col_el_gpu.o \
		sv_curv_col_el_iso_gpu.o \
		sv_curv_col_vis_iso_gpu.o \
		sv_curv_col_el_vti_gpu.o \
		sv_curv_col_el_aniso_gpu.o \
		sv_curv_col_ac_iso_gpu.o \
		main_curv_col_el_3d.o

OBJS := $(addprefix $(DIR_OBJ)/,$(OBJS))

vpath  %.cpp ./
vpath  %.cu ./

all: skel main
skel:
	@mkdir -p $(skeldirs)

main: $(OBJS)
	$(GC) $^ $(LDFLAGS) -o $@ 

$(DIR_OBJ)/%.o: src/media/%.cpp
	${CXX} $(CPPFLAGS) -c $^ -o $@ 
$(DIR_OBJ)/%.o: src/lib/%.cu
	${GC} $(CFLAGS_CUDA) -c $^ -o $@
$(DIR_OBJ)/%.o: src/forward/%.cu
	${GC} $(CFLAGS_CUDA) -c $^ -o $@

cleanexe:
	rm -f main
cleanobj:
	rm -f $(DIR_OBJ)
cleanall: cleanexe cleanobj
	echo "clean all"
distclean: cleanexe cleanobj
	echo "clean all"

