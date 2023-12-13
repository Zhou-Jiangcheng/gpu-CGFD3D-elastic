#ifndef BDRY_H
#define BDRY_H

#include "constants.h"
#include "gd_t.h"
#include "wav_t.h"

/*************************************************
 * structure
 *************************************************/

#define BDRY_FREE 1 
#define BDRY_PML  2 
#define BDRY_EXP  3 

/*
 * structure for PML auxvar
 */

typedef struct {
  float *var;
  int nx, ny, nz, ncmp, nlevel;

  size_t siz_iy;
  size_t siz_iz;
  size_t siz_icmp;
  size_t siz_ilevel;

  size_t *cmp_pos;
  char  **cmp_name;
  size_t *level_pos;

  size_t Vx_pos;
  size_t Vy_pos;
  size_t Vz_pos;
  size_t Txx_pos;
  size_t Tyy_pos;
  size_t Tzz_pos;
  size_t Tyz_pos;
  size_t Txz_pos;
  size_t Txy_pos;

  // for rk scheme
  float *cur;
  float *pre;
  float *rhs;
  float *end;
  float *tmp;

} bdrypml_auxvar_t;

/*
 * structure for block index range
 */

typedef struct {
  int enable;
  int nx1,nx2,ny1,ny2,nz1,nz2,nx,ny,nz;
  int ni1,ni2,nj1,nj2,nk1,nk2,ni,nj,nk;
} bdry_block_t;

/*
 * main bdry structure to implement free, pml, exp etc
 */

typedef struct
{
  int is_sides_free[CONST_NDIM][2];
  int is_enable_free;

  float *matD;  // for atten

  // top
  float *matVx2Vz2; // [j,i, dzVi, dxVi]
  float *matVy2Vz2;

  // bottom
  float *matVx2Vz1;
  float *matVy2Vz1;

  // left
  float *matVy2Vx1;
  float *matVz2Vx1;

  // right
  float *matVy2Vx2;
  float *matVz2Vx2;

  // front
  float *matVx2Vy1;
  float *matVz2Vy1;

  // back
  float *matVx2Vy2;
  float *matVz2Vy2;

} bdryfree_t;

typedef struct
{
  // 0 or 1 to indicate corresponding boundary condition
  //   such implementation simplifies calling the funcs
  int is_sides_pml [CONST_NDIM][2];
  int is_sides_mpml[CONST_NDIM][2];

  int is_enable_pml;
  int is_enable_mpml;

  // used for PML or exp
  int num_of_layers[CONST_NDIM][2]; //

  int ni1[CONST_NDIM][2];
  int ni2[CONST_NDIM][2];
  int nj1[CONST_NDIM][2];
  int nj2[CONST_NDIM][2];
  int nk1[CONST_NDIM][2];
  int nk2[CONST_NDIM][2];

  float *A[CONST_NDIM][2]; // dim, side, length
  float *B[CONST_NDIM][2]; // dim, side, length
  float *D[CONST_NDIM][2]; // dim, side, length

  bdrypml_auxvar_t auxvar[CONST_NDIM][2];

} bdrypml_t;

typedef struct
{
  int is_sides_ablexp [CONST_NDIM][2];
  int is_enable_ablexp;

  // used for PML or exp
  int num_of_layers[CONST_NDIM][2]; //

  int ni1[CONST_NDIM][2];
  int ni2[CONST_NDIM][2];
  int nj1[CONST_NDIM][2];
  int nj2[CONST_NDIM][2];
  int nk1[CONST_NDIM][2];
  int nk2[CONST_NDIM][2];
  // use 6 blocks to partition the boundaries
  bdry_block_t bdry_blk[CONST_NDIM_2];

  float *ablexp_Ex;
  float *ablexp_Ey;
  float *ablexp_Ez;

} bdryexp_t;
/*************************************************
 * function prototype
 *************************************************/

float
bdry_pml_cal_R(float N);

float
bdry_pml_cal_dmax(float L, float Vp, float Rpp);

float
bdry_pml_cal_amax(float fc);

float
bdry_pml_cal_d(float x, float L, float dmax);

float
bdry_pml_cal_a(float x, float L, float amax);

float
bdry_pml_cal_b(float x, float L, float bmax);

void
bdry_pml_set(gd_t *gd,
             wav_t *wav,
             bdrypml_t *bdrypml,
             int   *neighid, 
             int   in_is_sides[][2],
             int   in_num_layers[][2],
             float in_alpha_max[][2], //
             float in_beta_max[][2], //
             float in_velocity[][2], //
             int verbose);

// alloc auxvar
void
bdry_pml_auxvar_init(int nx, int ny, int nz, 
                     wav_t *wav,
                     bdrypml_auxvar_t *auxvar,
                     const int verbose);

int
bdry_cal_abl_len_dh(gd_t *gd, 
                    int abs_ni1, int abs_ni2,
                    int abs_nj1, int abs_nj2,
                    int abs_nk1, int abs_nk2,
                    int idim,
                    float *avg_L, float *avg_dh);

int
bdry_free_set(gd_t        *gd,
              bdryfree_t  *bdryfree,
              int   *neighid, 
              int   in_is_sides[][2],
              const int verbose);

int
bdry_ablexp_set(gd_t *gd,
                wav_t *wav,
                bdryexp_t *bdryexp,
                int   *neighid, 
                int   in_is_sides[][2],
                int   in_num_layers[][2],
                float in_velocity[][2], //
                float dt,
                int  *topoid,
                int verbose);

float
bdry_ablexp_cal_mask(int i, float vel, float dt, int num_lay, float dh);

int
bdry_ablexp_apply(bdryexp_t bdryexp, gd_t *gd, float *w_end, int ncmp);

__global__ 
void
bdry_ablexp_apply_gpu(float *Ex, float *Ey, float *Ez, 
                      float *w_end, int ncmp, 
                      int ni1, int nj1, int nk1, int ni, int nj, int nk,
                      size_t siz_iy, size_t siz_iz, size_t siz_icmp);

#endif
