#ifndef MD_EL_H
#define MD_EL_H

#include "gd_t.h"

/*************************************************
 * structure
 *************************************************/

typedef struct {
  int nx, ny, nz, ncmp;
  float *v4d; // allocated var

  size_t siz_iy;
  size_t siz_iz;
  size_t siz_icmp;

  size_t *cmp_pos;
  char  **cmp_name;

  // flag to determine medium type
  int medium_type;

  // rho for all media
  float *rho;

  // for acustic
  float *kappa; // pointer to var

  // for isotropic media
  float *lambda; // pointer to var
  float *mu;

  // for visco attenuation
  int nmaxwell;
  float visco_GMB_freq;
  float visco_GMB_fmin;
  float visco_GMB_fmax;
  float *Qs;
  float *Qp;
  // use pointer array,
  // easy gpu code implement
  // default maxwell number <= 10 
  float *Ylam[10];
  float *Ymu[10];
  float *wl;

  // for anisotropic media
  float *c11;
  float *c12;
  float *c13;
  float *c14;
  float *c15;
  float *c16;
  float *c22;
  float *c23;
  float *c24;
  float *c25;
  float *c26;
  float *c33;
  float *c34;
  float *c35;
  float *c36;
  float *c44;
  float *c45;
  float *c46;
  float *c55;
  float *c56;
  float *c66;

  int visco_type;
  float visco_Qs_freq;

} md_t;

/*************************************************
 * function prototype
 *************************************************/

int
md_init(gd_t *gd, md_t *md, int media_type, int visco_type, int nmaxwell);

int
md_import(gd_t *gd, md_t *md, char *fname_coords, char *in_dir);

int
md_export(gd_t *gd,
          md_t *md,
          char *fname_coords,
          char *output_dir);

int
md_gen_test_ac_iso(md_t *md);

int
md_gen_test_el_iso(md_t *md);

int
md_gen_test_Qs(md_t *md, float Qs_freq);

int
md_gen_test_el_vti(md_t *md);

int
md_gen_test_el_aniso(md_t *md);

int
md_gen_test_vis_iso(md_t *md);

int
md_rho_to_slow(float *rho, size_t siz_icmp);

int
md_ac_Vp_to_kappa(float *rho, float *kappa, size_t siz_icmp);

int
md_vis_GMB_cal_Y(md_t *md, float freq, float fmin, float fmax);

int 
md_visco_LS(float ** input, float * output, float d, int m, int n);

int 
md_visco_LS_mat_inv(float matrix[][VISCO_LS_MAXSIZE], float inverse[][VISCO_LS_MAXSIZE], int n);

#endif
