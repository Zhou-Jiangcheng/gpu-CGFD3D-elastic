#ifndef IO_FUNCS_H
#define IO_FUNCS_H

#include "constants.h"
#include "gd_info.h"
#include "gd_t.h"
#include "io_funcs.h"
#include "md_t.h"
#include "wav_t.h"

/*************************************************
 * structure
 *************************************************/

// for stations output

// single station
typedef struct
{
  float x;
  float y;
  float z;
  float di;
  float dj;
  float dk;
  int   i;
  int   j;
  int   k;
  size_t   indx1d[CONST_2_NDIM];
  float *seismo;
  char  name[CONST_MAX_STRLEN];
} iorecv_one_t;

typedef struct
{
  int                 total_number;
  int                 max_nt;
  int                 ncmp;
  iorecv_one_t *recvone;
} iorecv_t;

// line output
typedef struct
{
  int     num_of_lines; 
  int     max_nt;
  int     ncmp;

  int    *line_nr; // number of receivers, for name from input file
  int    *line_seq; // line number, for name from input file
  //int    **recv_ir;
  //int    **recv_jr;
  //int    **recv_kr;
  int    **recv_seq; // recv seq in this line
  int    **recv_iptr;
  float  **recv_x; // for sac output
  float  **recv_y; // for sac output
  float  **recv_z; // for sac output
  float  **recv_seismo;
  char   **line_name;
} ioline_t;

// slice output
typedef struct
{
  // for esti size of working space var
  size_t siz_max_wrk;

  int num_of_slice_x;
  int *slice_x_indx;
  char **slice_x_fname;

  int num_of_slice_y;
  int *slice_y_indx;
  char **slice_y_fname;

  int num_of_slice_z;
  int *slice_z_indx;
  char **slice_z_fname;
} ioslice_t;

// snapshot output
typedef struct
{
  // for esti size of working space var
  size_t siz_max_wrk;

  int num_of_snap;

  int *i1;
  int *j1;
  int *k1;
  int *ni;
  int *nj;
  int *nk;
  int *di;
  int *dj;
  int *dk;
  int *it1;
  int *dit;
  int *out_vel;
  int *out_stress;
  int *out_strain;

  int *i1_to_glob;
  int *j1_to_glob;
  int *k1_to_glob;

  char **fname;
} iosnap_t;

// for nc output

typedef struct
{
  int num_of_slice_x;
  int num_of_slice_y;
  int num_of_slice_z;
  int num_of_vars;

  int *ncid_slx;
  int *timeid_slx;
  int *varid_slx;

  int *ncid_sly;
  int *timeid_sly;
  int *varid_sly;

  int *ncid_slz;
  int *timeid_slz;
  int *varid_slz;
}
ioslice_nc_t;

typedef struct
{
  int num_of_snap;
  int *ncid;
  int *timeid;
  int *varid_V;  // [num_of_snap*CONST_NDIM];
  int *varid_T;  // [num_of_snap*CONST_NDIM_2];
  int *varid_E;  // [num_of_snap*CONST_NDIM_2];
  int *cur_it ;  // [num_of_snap];
}
iosnap_nc_t;

/*************************************************
 * function prototype
 *************************************************/

void
io_build_fname(char *out_dir,
               char *prefix,
               char *subfix,
               int *myid2,
               char *ou_fname);

void
io_build_fname_time(char *out_dir,
                    char *prefix,
                    char *subfix,
                    int *myid2,
                    int  it,
                    char *ou_fname);

void
io_snapshot_export_binary(char *fname,
                   float *var,
                   int nx,
                   int ny,
                   int nz,
                   int *snap_indx,
                   int verbose);

void
io_var3d_export_nc(char   *ou_file,
                   float  *v3d,
                   size_t *v3d_pos,
                   char  **v3d_name,
                   int   number_of_vars,
                   char  **coord_name,
                   int  nx,
                   int  ny,
                   int  nz);

int
io_recv_read_locate(gdinfo_t *gdinfo,
                    gd_t *gd,
                    iorecv_t  *iorecv,
                    int       nt_total,
                    int       num_of_vars,
                    char      *in_filenm,
                    MPI_Comm  comm,
                    int       myid,
                    int       verbose);

int
io_line_locate(gdinfo_t *gdinfo,
               gd_t *gd,
               ioline_t *ioline,
               int    num_of_vars,
               int    nt_total,
               int    number_of_receiver_line,
               int   *receiver_line_index_start,
               int   *receiver_line_index_incre,
               int   *receiver_line_count,
               char **receiver_line_name);

int
io_slice_locate(gdinfo_t  *gdinfo,
                ioslice_t *ioslice,
                int  number_of_slice_x,
                int  number_of_slice_y,
                int  number_of_slice_z,
                int *slice_x_index,
                int *slice_y_index,
                int *slice_z_index,
                char *output_fname_part,
                char *output_dir);

void
io_snapshot_locate(gdinfo_t *gdinfo,
                   iosnap_t *iosnap,
                    int  number_of_snapshot,
                    char **snapshot_name,
                    int *snapshot_index_start,
                    int *snapshot_index_count,
                    int *snapshot_index_incre,
                    int *snapshot_time_start,
                    int *snapshot_time_incre,
                    int *snapshot_save_velocity,
                    int *snapshot_save_stress,
                    int *snapshot_save_strain,
                    char *output_fname_part,
                    char *output_dir);

int
io_slice_nc_create(ioslice_t *ioslice, 
                  int num_of_vars, char **w3d_name,
                  int ni, int nj, int nk,
                  int *topoid, ioslice_nc_t *ioslice_nc);

int
io_snap_nc_create(iosnap_t *iosnap, iosnap_nc_t *iosnap_nc, int *topoid);

int
io_slice_nc_put(ioslice_t    *ioslice,
                ioslice_nc_t *ioslice_nc,
                gdinfo_t     *gdinfo,
                float *w4d,
                float *buff,
                int   it,
                float time,
                int   i1_cmp,
                int   i2_cmp);

int
io_snap_nc_put(iosnap_t *iosnap,
               iosnap_nc_t *iosnap_nc,
               gdinfo_t    *gdinfo,
               md_t    *md,
               wav_t   *wav,
               float *w4d,
               float *buff,
               int   nt_total,
               int   it,
               float time,
               int is_run_out_vel,     // for stg, out vel and stress at sep call
               int is_run_out_stress,  // 
               int is_incr_cur_it);     // for stg, should output cur_it once

int
io_snap_nc_create_ac(iosnap_t *iosnap, iosnap_nc_t *iosnap_nc, int *topoid);

int
io_snap_nc_put_ac(iosnap_t *iosnap,
               iosnap_nc_t *iosnap_nc,
               gdinfo_t    *gdinfo,
               wav_t   *wav,
               float *w4d,
               float *buff,
               int   nt_total,
               int   it,
               float time,
               int is_run_out_vel,     // for stg, out vel and stress at sep call
               int is_run_out_stress,  // 
               int is_incr_cur_it);     // for stg, should output cur_it once

int
io_snap_stress_to_strain_eliso(float *lam3d,
                               float *mu3d,
                               float *Txx,
                               float *Tyy,
                               float *Tzz,
                               float *Tyz,
                               float *Txz,
                               float *Txy,
                               float *Exx,
                               float *Eyy,
                               float *Ezz,
                               float *Eyz,
                               float *Exz,
                               float *Exy,
                               size_t siz_line,
                               size_t siz_slice,
                               int starti,
                               int counti,
                               int increi,
                               int startj,
                               int countj,
                               int increj,
                               int startk,
                               int countk,
                               int increk);

__global__ void
io_slice_pack_buff_x(int i, int nj, int nk, size_t siz_line, size_t siz_slice, float *var, float *buff_d);

__global__ void
io_slice_pack_buff_y(int j, int ni, int nk, size_t siz_line, size_t siz_slice, float *var, float *buff_d);

__global__ void
io_slice_pack_buff_z(int k, int ni, int nj, size_t siz_line, size_t siz_slice, float *var, float *buff_d);

__global__ void
io_snap_pack_buff(float *var,
                  size_t siz_line,
                  size_t siz_slice,
                  int starti,
                  int counti,
                  int increi,
                  int startj,
                  int countj,
                  int increj,
                  int startk,
                  int countk,
                  int increk,
                  float *buff_d);

int
io_slice_nc_close(ioslice_nc_t *ioslice_nc);

int
io_snap_nc_close(iosnap_nc_t *iosnap_nc);

int
io_recv_keep(iorecv_t *iorecv, float *w_end_d,
             float* buff, int it, int ncmp, size_t siz_icmp);

int
io_line_keep(ioline_t *ioline, float *w_end_d,
             float *buff, int it, int ncmp, size_t siz_icmp);

__global__ void
recv_depth_to_axis(float *all_coords_d, int num_recv, gdinfo_t gdinfo_d, gd_t gd_d, 
                   int *flag_indx, int *flag_depth, 
                   MPI_Comm comm, int myid);

__global__ void 
recv_coords_to_glob_indx(float *all_coords_d, int *all_index_d, 
                         float *all_inc_d, int num_recv, gdinfo_t gdinfo_d, gd_t gd_d, 
                         int *flag_indx, MPI_Comm comm, int myid);

//use trilinear interpolation 
__global__ void
io_recv_line_interp_pack_buff(float *var, float *buff_d, int ncmp, size_t siz_icmp, size_t *indx1d_d);

__global__ void
io_recv_line_pack_buff(float *var, float *buff_d, int ncmp,
                  size_t siz_icmp, int iptr);

int
io_recv_output_sac(iorecv_t *iorecv,
                   float dt,
                   int num_of_vars,
                   char **cmp_name,
                   char *evtnm,
                   char *output_dir,
                   char *err_message);
int
io_recv_output_sac_el_iso_strain(iorecv_t *iorecv,
                   float * lam3d,
                   float * mu3d,
                   float dt,
                   char *evtnm,
                   char *output_dir,
                   char *err_message);

int
io_recv_output_sac_el_vti_strain(iorecv_t *iorecv,
                        float * c11, float * c13,
                        float * c33, float * c55,
                        float * c66,
                        float dt,
                        char *evtnm,
                        char *output_dir,
                        char *err_message);

int
io_recv_output_sac_el_aniso_strain(iorecv_t *iorecv,
                        float * c11d, float * c12d,
                        float * c13d, float * c14d,
                        float * c15d, float * c16d,
                        float * c22d, float * c23d,
                        float * c24d, float * c25d,
                        float * c26d, float * c33d,
                        float * c34d, float * c35d,
                        float * c36d, float * c44d,
                        float * c45d, float * c46d,
                        float * c55d, float * c56d,
                        float * c66d,
                        float dt,
                        char *evtnm,
                        char *output_dir,
                        char *err_message);

int
io_line_output_sac(ioline_t *ioline,
      float dt, char **cmp_name, char *evtnm, char *output_dir);

int
ioslice_print(ioslice_t *ioslice);

int
iosnap_print(iosnap_t *iosnap);

int
iorecv_print(iorecv_t *iorecv);

int
PG_slice_output(float *PG,  gdinfo_t *gdinfo, char *output_dir, char *frame_coords, int* topoid);

int
io_get_nextline(FILE *fp, char *str, int length);

#endif
