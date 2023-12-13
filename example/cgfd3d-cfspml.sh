#!/bin/bash

#set -x
set -e

date

#-- system related dir
MPIDIR=/data3/lihl/software/openmpi-gnu-4.1.2
#MPIDIR=/data/apps/openmpi/4.1.5-cuda-aware

#-- program related dir
EXEC_WAVE=`pwd`/../main_curv_col_el_3d
echo "EXEC_WAVE=${EXEC_WAVE}"

#-- input dir
INPUTDIR=`pwd`

#-- output and conf
PROJDIR=`pwd`/../project
PAR_FILE=${PROJDIR}/test.json
GRID_DIR=${PROJDIR}/output
MEDIA_DIR=${PROJDIR}/output
SOURCE_DIR=${PROJDIR}/output
OUTPUT_DIR=${PROJDIR}/output

rm -rf ${PROJDIR}

#-- create dir
mkdir -p ${PROJDIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${GRID_DIR}
mkdir -p ${MEDIA_DIR}

#----------------------------------------------------------------------
#-- grid and mpi configurations
#----------------------------------------------------------------------

#-- total x grid points
NX=200
#-- total y grid points
NY=220
#-- total z grid points
NZ=200
#-- total x mpi procs
NPROCS_X=2
#-- total y mpi procs
NPROCS_Y=2
#----------------------------------------------------------------------
#-- create main conf
#----------------------------------------------------------------------
cat << ieof > ${PAR_FILE}
{
  "number_of_total_grid_points_x" : ${NX},
  "number_of_total_grid_points_y" : ${NY},
  "number_of_total_grid_points_z" : ${NZ},

  "number_of_mpiprocs_x" : $NPROCS_X,
  "number_of_mpiprocs_y" : $NPROCS_Y,

  "size_of_time_step" : 0.01,
  "number_of_time_steps" : 500,
  "#time_window_length" : 8,
  "check_stability" : 1,

  "boundary_x_left" : {
      "cfspml" : {
          "number_of_layers" : 20,
          "alpha_max" : 3.0,
          "beta_max" : 2.5,
          "ref_vel"  : 5000.0
          }
      },
  "boundary_x_right" : {
      "cfspml" : {
          "number_of_layers" : 20,
          "alpha_max" : 3.0,
          "beta_max" : 2.5,
          "ref_vel"  : 5000.0
          }
      },
  "boundary_y_front" : {
      "cfspml" : {
          "number_of_layers" : 20,
          "alpha_max" : 3.0,
          "beta_max" : 2.5,
          "ref_vel"  : 5000.0
          }
      },
  "boundary_y_back" : {
      "cfspml" : {
          "number_of_layers" : 20,
          "alpha_max" : 3.0,
          "beta_max" : 2.5,
          "ref_vel"  : 5000.0
          }
      },
  "boundary_z_bottom" : {
      "cfspml" : {
          "number_of_layers" : 20,
          "alpha_max" : 3.0,
          "beta_max" : 2.5,
          "ref_vel"  : 5000.0
          }
      },
  "boundary_z_top" : {
      "free" : "timg"
      },

  "grid_generation_method" : {
      "#import" : "$INPUTDIR/grid_model1",
      "cartesian" : {
        "origin"  : [0.0, 0.0, -29900.0 ],
        "inteval" : [ 100.0, 100.0, 100.0 ]
      }
  },
  "is_export_grid" : 1,
  "grid_export_dir"   : "$GRID_DIR",

  "metric_calculation_method" : {
      "#import" : "$GRID_DIR",
      "calculate" : 1
  },
  "is_export_metric" : 1,

  "medium" : {
      "type" : "viscoelastic_iso",
      "#type" : "elastic_iso",
      "#input_way" : "infile_layer",
      "#input_way" : "binfile",
      "input_way" : "code",
      "#input_way" : "import",
      "#binfile" : {
        "size"    : [1001, 1447, 1252],
        "spacing" : [-10, 10, 10],
        "origin"  : [0.0,0.0,0.0],
        "dim1" : "z",
        "dim2" : "x",
        "dim3" : "y",
        "Vp" : "$INPUTDIR/prep_medium/seam_vp.bin",
        "Vs" : "$INPUTDIR/prep_medium/seam_vs.bin",
        "rho" : "$INPUTDIR/prep_medium/seam_rho.bin"
      },
      "code" : "func_name_here",
      "#import" : "$MEDIA_DIR",
      "#infile_layer" : "$INPUTDIR/prep_medium/basin_el_iso.md3lay",
      "#infile_grid" : "$INPUTDIR/prep_medium/topolay_el_iso.md3grd",
      "#equivalent_medium_method" : "loc",
      "#equivalent_medium_method" : "har"
  },

  "is_export_media" : 1,
  "media_export_dir"  : "$MEDIA_DIR",

  "visco_config" : {
      "visco_type" : "gmb",
      "Qs_freq" : 1.0,
      "number_of_maxwell" : 3,
      "max_freq" : 10.0,
      "min_freq" : 0.1,
      "refer_freq" : 1.0
  },

  "in_source_file" : "$INPUTDIR/prep_source/test_source.src",
  "is_export_source" : 1,
  "source_export_dir"  : "$SOURCE_DIR",

  "output_dir" : "$OUTPUT_DIR",

  "in_station_file" : "$INPUTDIR/prep_station/station.list",

  "#receiver_line" : [
    {
      "name" : "line_x_1",
      "grid_index_start"    : [  50, 149, 59 ],
      "grid_index_incre"    : [  5,  0,  0 ],
      "grid_index_count"    : 10
    },
    {
      "name" : "line_y_1",
      "grid_index_start"    : [ 200, 100, 59 ],
      "grid_index_incre"    : [  0,  5,  0 ],
      "grid_index_count"    : 10
    } 
  ],

  "slice" : {
      "x_index" : [ 100 ],
      "y_index" : [ 110 ],
      "z_index" : [ 100 ]
  },

  "snapshot" : [
    {
      "name" : "volume_vel",
      "grid_index_start" : [ 0, 100, 0 ],
      "grid_index_count" : [ $NX, 1, $NZ ],
      "grid_index_incre" : [  1, 1, 1 ],
      "time_index_start" : 0,
      "time_index_incre" : 1,
      "save_velocity" : 1,
      "save_stress"   : 0,
      "save_strain"   : 0
    }
  ],

  "check_nan_every_nummber_of_steps" : 0,
  "output_all" : 0 
}
ieof

echo "+ created $PAR_FILE"

#-------------------------------------------------------------------------------
#-- Performce simulation
#-------------------------------------------------------------------------------
#

#-- get np
NUMPROCS_X=`grep number_of_mpiprocs_x ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS_Y=`grep number_of_mpiprocs_y ${PAR_FILE} | sed 's/:/ /g' | sed 's/,/ /g' | awk '{print $2}'`
NUMPROCS=$(( NUMPROCS_X*NUMPROCS_Y ))
echo $NUMPROCS_X $NUMPROCS_Y $NUMPROCS

#-- gen run script
cat << ieof > ${PROJDIR}/cgfd_sim.sh
#!/bin/bash

set -e
printf "\nUse $NUMPROCS CPUs on following nodes:\n"

printf "\nStart simualtion ...\n";
time $MPIDIR/bin/mpiexec -np $NUMPROCS $EXEC_WAVE $PAR_FILE 100 0 2>&1 |tee log
if [ $? -ne 0 ]; then
    printf "\nSimulation fail! stop!\n"
    exit 1
fi

ieof

#-------------------------------------------------------------------------------
#-- start run
#-------------------------------------------------------------------------------

chmod 755 ${PROJDIR}/cgfd_sim.sh
${PROJDIR}/cgfd_sim.sh
if [ $? -ne 0 ]; then
    printf "\nSimulation fail! stop!\n"
    exit 1
fi

date

# vim:ts=4:sw=4:nu:et:ai:
