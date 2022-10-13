#ifndef FD_T_H
#define FD_T_H

/*******************************************************************************
 *macro for fd opterators
 *******************************************************************************/

// use siz_shift to find adjacent point of the stentil for 3d var
#define M_FD_SHIFT(deriv, var, iptr, fd_length, fd_shift, fd_coef, n) \
   deriv = fd_coef[0] * var[iptr + fd_shift[0]];                      \
   for (n=1; n<fd_length; n++) {                                      \
       deriv += fd_coef[n] * var[iptr + fd_shift[n]];                 \
   }

// use pointer for cur point for speedup
#define M_FD_SHIFT_PTR(deriv, var_ptr, fd_length, fd_shift, fd_coef, n) \
  deriv = fd_coef[0] * *(var_ptr + fd_shift[0]);                        \
  for (n = 1; n < fd_length; n++)                                       \
  {                                                                     \
    deriv += fd_coef[n] * *(var_ptr + fd_shift[n]);                     \
  }

// only valid for macdrp etc with len = 5, may be faster? 
#define M_FD_SHIFT_PTR_MACDRP(deriv, var_ptr, fd_shift, fd_coef) \
  deriv = fd_coef[0] * *(var_ptr + fd_shift[0])                  \
         +fd_coef[1] * *(var_ptr + fd_shift[1])                  \
         +fd_coef[2] * *(var_ptr + fd_shift[2])                  \
         +fd_coef[3] * *(var_ptr + fd_shift[3])                  \
         +fd_coef[4] * *(var_ptr + fd_shift[4])

// assume var has the same size as fd_coef, ordered one by one, thus no index needed
#define M_FD_NOINDX(deriv, var, fd_length, fd_coef, n) \
   deriv = fd_coef[0] * var[0];                        \
   for (n=1; n<fd_length; n++) {                       \
       deriv += fd_coef[n] * var[n];                   \
   }

// use indx relative to cur point as (-1,0,1), need to multiply siz_shift for 3d array
#define M_FD_INDX(deriv, var, iptr, fd_length, fd_indx, fd_coef, shift, n) \
   deriv = fd_coef[0] * var[iptr + fd_shift[0] * shift];                   \
   for (n=1; n<fd_length; n++) {                                           \
       deriv += fd_coef[n] * var[iptr + fd_shift[n] * shift];              \
   }

/*******************************************************************************
 * structure for different fd schemes
 ******************************************************************************/

/*
 * elementary operator
 */

typedef struct
{
  int total_len;
  int half_len;
  int left_len;
  int right_len;
  int   *indx;  // indx change to cur point as 0 for 1d
  int   *shift; // num of grid points skipped
  float *coef;
} fd_op_t; 

/*
 * collocated grid scheme
 */

typedef struct {

  float CFL; // 1d cfl value for the scheme

  //----------------------------------------------------------------------------
  // Runge-Kutta time scheme
  //----------------------------------------------------------------------------

  int num_rk_stages;
  float *rk_a;
  float *rk_b;
  float *rk_rhs_time; // relative time for rhs eval

  //----------------------------------------------------------------------------
  // central scheme
  //----------------------------------------------------------------------------

  int     fdc_len;
  int     fdc_half_len;
  int     fdc_nghosts;
  int    *fdc_indx;
  float  *fdc_coef;

  //----------------------------------------------------------------------------
  // para for different schemes at points to boundaries for different dim
  //----------------------------------------------------------------------------

  // ghost point required 
  int fdx_nghosts;
  int fdy_nghosts;
  int fdz_nghosts;

  // max total len of op
  int fdx_max_len;
  int fdy_max_len;
  int fdz_max_len;

  // max half len
  int fdx_max_half_len;
  int fdy_max_half_len;
  int fdz_max_half_len;

  //----------------------------------------------------------------------------
  // pairs for 3d space for MacCormack-type schemes
  //----------------------------------------------------------------------------

  int num_of_pairs;

  // number of layers that need to use biased op near boundary
  int num_of_fdx_op;
  int num_of_fdy_op;
  int num_of_fdz_op;

  fd_op_t ***pair_fdx_op; // [pair][stage][nlay]
  fd_op_t ***pair_fdy_op;
  fd_op_t ***pair_fdz_op;

} fd_t;

typedef struct
{
  int *fdz_len_d;
  float *fdx_coef_d;
  float *fdy_coef_d;
  float *fdz_coef_d;
  float *fdz_coef_all_d;

  int *fdx_indx_d;
  int *fdy_indx_d;
  int *fdz_indx_d;
  int *fdz_indx_all_d;
  
  size_t *fdx_shift_d;
  size_t *fdy_shift_d;
  size_t *fdz_shift_d;
  size_t *fdz_shift_all_d;
} fd_wav_t;

/*******************************************************************************
 * function prototype
 ******************************************************************************/

int 
fd_set_macdrp(fd_t *fd);

void
fd_print(fd_t *fd);

#endif
