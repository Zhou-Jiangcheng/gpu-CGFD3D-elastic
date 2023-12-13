#ifndef CONSTANTS_H
#define CONSTANTS_H

// consts
#define CONST_NDIM   3
#define CONST_NDIM_2 6  // 2 * ndim
#define CONST_NDIM_3 9  // 3 * ndim
#define CONST_NDIM_4 12 // 4 * ndim
#define CONST_NDIM_5 15 // 5 * ndim
#define CONST_2_NDIM 8  // 2 ^ ndim
#define CONST_MAX_STRLEN 1024

#ifndef M_PI
#define PI 3.14159265358979323846264338327950288419716939937510
#else
#define PI M_PI
#endif

// medium type
#define CONST_MEDIUM_ACOUSTIC_ISO  1
#define CONST_MEDIUM_ELASTIC_ISO   2
#define CONST_MEDIUM_ELASTIC_VTI   3
#define CONST_MEDIUM_ELASTIC_ANISO 4
#define CONST_MEDIUM_VISCOELASTIC_ISO 5

#define VISCO_LS_MAXSIZE 20

// visco type
#define CONST_VISCO_GRAVES_QS  1
#define CONST_VISCO_GMB 2

#define handle_nc_err(err)                       \
{                                                \
  if (err != NC_NOERR) {                         \
     fprintf(stderr,"nc error: %s\n", nc_strerror(err)); \
     exit(-1);                                   \
  }                                              \
}

#endif
