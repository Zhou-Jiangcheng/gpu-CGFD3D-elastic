#ifndef BLK_T_H
#define BLK_T_H

#include "constants.h"
#include "fd_t.h"
#include "mympi_t.h"
#include "gd_t.h"
#include "md_t.h"
#include "wav_t.h"
#include "src_t.h"
#include "bdry_t.h"
#include "io_funcs.h"

/*******************************************************************************
 * structure
 ******************************************************************************/

typedef struct
{
  // name for output file name
  char name[CONST_MAX_STRLEN];

  //// flag of medium
  //int medium_type;

  // fd
  fd_t    *fd;     // collocated grid fd

  // mpi
  mympi_t *mympi;
  
  // coordnate: x3d, y3d, z3d
  gd_t *gd;

  // grid metrics: jac, xi_x, etc
  gd_metric_t *gd_metric;

  // media: rho, lambda, mu etc
  md_t *md;

  // wavefield:
  wav_t *wav;
  
  // source term
  src_t *src;
  
  bdryfree_t *bdryfree;
  bdrypml_t  *bdrypml;
  bdryexp_t  *bdryexp;

  // io
  iorecv_t  *iorecv;
  ioline_t  *ioline;
  iosnap_t  *iosnap;
  ioslice_t *ioslice;

  // fname and dir
  char output_fname_part[CONST_MAX_STRLEN];
  // wavefield output
  char output_dir[CONST_MAX_STRLEN];
  // seperate grid output to save grid for repeat simulation
  char grid_export_dir[CONST_MAX_STRLEN];
  // seperate medium output to save medium for repeat simulation
  char media_export_dir[CONST_MAX_STRLEN];

  // exchange between blocks
  // point-to-point values
  int num_of_conn_points;
  int *conn_this_indx;
  int *conn_out_blk;
  int *conn_out_indx;
  // interp point
  int num_of_interp_points;
  int *interp_this_indx;
  int *interp_out_blk;
  int *interp_out_indx;
  float *interp_out_dxyz;

  // mem usage
  size_t number_of_float;
  size_t number_of_btye;
} blk_t;

/*******************************************************************************
 * function prototype
 ******************************************************************************/

int
blk_init(blk_t *blk,
         const int myid);

// set str
int
blk_set_output(blk_t *blk,
               mympi_t *mympi,
               char *output_dir,
               char *grid_export_dir,
               char *media_export_dir);

void
blk_macdrp_mesg_init(mympi_t *mympi,
                fd_t *fd,
                int ni,
                int nj,
                int nk,
                int num_of_vars);

int
blk_macdrp_pack_mesg_gpu(float *w_cur,
                         fd_t *fd,
                         gd_t *gd,
                         mympi_t *mpmpi,
                         int num_of_vars,
                         int ipair_mpi,
                         int istage_mpi,
                         int myid);


__global__ void
blk_macdrp_pack_mesg_x1(
           float* w_cur,float *sbuff_x1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj1, int nk1, int nx1_g, int nj, int nk);


__global__ void
blk_macdrp_pack_mesg_x2(
           float* w_cur,float *sbuff_x2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni2, int nj1, int nk1, int nx2_g, int nj, int nk);


__global__ void
blk_macdrp_pack_mesg_y1(
           float* w_cur,float *sbuff_y1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int ny1_g, int nk);

__global__ void
blk_macdrp_pack_mesg_y2(
           float* w_cur,float *sbuff_y2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj2, int nk1, int ni, int ny2_g, int nk);

int 
blk_macdrp_unpack_mesg_gpu(float *w_cur, 
                           fd_t *fd,
                           gd_t *gd,
                           mympi_t *mympi, 
                           int num_of_vars,
                           int ipair_mpi,
                           int istage_mpi,
                           int *neighid);

__global__ void
blk_macdrp_unpack_mesg_x1(
           float *w_cur, float *rbuff_x1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj1, int nk1, int nx2_g, int nj, int nk, int *neighid);


__global__ void
blk_macdrp_unpack_mesg_x2(
           float *w_cur, float *rbuff_x2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni2, int nj1, int nk1, int nx1_g, int nj, int nk, int *neighid);

__global__ void
blk_macdrp_unpack_mesg_y1(
           float *w_cur, float *rbuff_y1, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj1, int nk1, int ni, int ny2_g, int nk, int *neighid);

__global__ void
blk_macdrp_unpack_mesg_y2(
           float *w_cur, float *rbuff_y2, size_t siz_iy, size_t siz_iz, size_t siz_icmp,
           int num_of_vars, int ni1, int nj2, int nk1, int ni, int ny1_g, int nk, int *neighid);

int
blk_print(blk_t *blk);

int
blk_dt_esti_curv(gd_t *gd, md_t *md,
    float CFL, float *dtmax, float *dtmaxVp, float *dtmaxL,
    int *dtmaxi, int *dtmaxj, int *dtmaxk);

float
blk_keep_three_digi(float dt);

#endif
