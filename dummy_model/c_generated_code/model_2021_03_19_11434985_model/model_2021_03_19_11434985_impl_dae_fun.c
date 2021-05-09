/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) model_2021_03_19_11434985_impl_dae_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};

/* model_2021_03_19_11434985_impl_dae_fun:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][1] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][2] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][3] : 0;
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][4] : 0;
  a1=arg[0]? arg[0][6] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][5] : 0;
  a2=arg[0]? arg[0][7] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][6] : 0;
  a3=1.2330447799599942e+00;
  a4=-2.7025639012821762e-01;
  a5=5.0000000000000000e-01;
  a6=arg[0]? arg[0][4] : 0;
  a7=sin(a6);
  a8=(a5*a7);
  a9=-5.0000000000000000e-01;
  a10=(a8+a9);
  a11=cos(a6);
  a12=(a5*a11);
  a10=(a10*a12);
  a11=(a5*a11);
  a7=(a5*a7);
  a7=(a11*a7);
  a10=(a10-a7);
  a7=(a8+a9);
  a7=casadi_sq(a7);
  a12=casadi_sq(a11);
  a7=(a7+a12);
  a7=sqrt(a7);
  a10=(a10/a7);
  a7=arg[0]? arg[0][0] : 0;
  a12=arg[2]? arg[2][0] : 0;
  a13=1.;
  a14=(a12<a13);
  a12=(a14?a12:0);
  a14=(!a14);
  a14=(a14?a13:0);
  a12=(a12+a14);
  a8=(a8+a9);
  a8=casadi_sq(a8);
  a11=casadi_sq(a11);
  a8=(a8+a11);
  a8=sqrt(a8);
  a8=(a8-a13);
  a11=9.9995000041666526e-01;
  a8=(a8/a11);
  a9=6.9999999999999996e-01;
  a14=(a8/a9);
  a14=(a14-a13);
  a14=casadi_sq(a14);
  a15=4.5000000000000001e-01;
  a14=(a14/a15);
  a14=(-a14);
  a14=exp(a14);
  a12=(a12*a14);
  a14=(a10*a1);
  a16=0.;
  a17=(a14<=a16);
  a18=fabs(a14);
  a19=10.;
  a18=(a18/a19);
  a18=(a13-a18);
  a20=fabs(a14);
  a20=(a20/a19);
  a20=(a13+a20);
  a18=(a18/a20);
  a18=(a17?a18:0);
  a17=(!a17);
  a20=1.3300000000000001e+00;
  a21=(a20*a14);
  a21=(a21/a19);
  a22=-8.2500000000000004e-02;
  a21=(a21/a22);
  a21=(a13-a21);
  a23=(a14/a19);
  a23=(a23/a22);
  a23=(a13-a23);
  a21=(a21/a23);
  a17=(a17?a21:0);
  a18=(a18+a17);
  a12=(a12*a18);
  a18=(a13<a8);
  a8=(a8/a9);
  a8=(a8-a13);
  a8=(a19*a8);
  a8=exp(a8);
  a8=(a8-a13);
  a17=1.4741315910257660e+02;
  a8=(a8/a17);
  a18=(a18?a8:0);
  a12=(a12+a18);
  a18=1.0000000000000001e-01;
  a8=7.;
  a14=(a14/a8);
  a14=(a18*a14);
  a12=(a12+a14);
  a7=(a7*a12);
  a10=(a10*a7);
  a7=sin(a6);
  a12=(a5*a7);
  a14=(a12+a5);
  a21=cos(a6);
  a23=(a5*a21);
  a14=(a14*a23);
  a21=(a5*a21);
  a7=(a5*a7);
  a7=(a21*a7);
  a14=(a14-a7);
  a7=(a12+a5);
  a7=casadi_sq(a7);
  a23=casadi_sq(a21);
  a7=(a7+a23);
  a7=sqrt(a7);
  a14=(a14/a7);
  a7=arg[0]? arg[0][1] : 0;
  a23=arg[2]? arg[2][1] : 0;
  a24=(a23<a13);
  a23=(a24?a23:0);
  a24=(!a24);
  a24=(a24?a13:0);
  a23=(a23+a24);
  a12=(a12+a5);
  a12=casadi_sq(a12);
  a21=casadi_sq(a21);
  a12=(a12+a21);
  a12=sqrt(a12);
  a12=(a12-a13);
  a12=(a12/a11);
  a21=(a12/a9);
  a21=(a21-a13);
  a21=casadi_sq(a21);
  a21=(a21/a15);
  a21=(-a21);
  a21=exp(a21);
  a23=(a23*a21);
  a21=(a14*a1);
  a24=(a21<=a16);
  a25=fabs(a21);
  a25=(a25/a19);
  a25=(a13-a25);
  a26=fabs(a21);
  a26=(a26/a19);
  a26=(a13+a26);
  a25=(a25/a26);
  a25=(a24?a25:0);
  a24=(!a24);
  a26=(a20*a21);
  a26=(a26/a19);
  a26=(a26/a22);
  a26=(a13-a26);
  a27=(a21/a19);
  a27=(a27/a22);
  a27=(a13-a27);
  a26=(a26/a27);
  a24=(a24?a26:0);
  a25=(a25+a24);
  a23=(a23*a25);
  a25=(a13<a12);
  a12=(a12/a9);
  a12=(a12-a13);
  a12=(a19*a12);
  a12=exp(a12);
  a12=(a12-a13);
  a12=(a12/a17);
  a25=(a25?a12:0);
  a23=(a23+a25);
  a21=(a21/a8);
  a21=(a18*a21);
  a23=(a23+a21);
  a7=(a7*a23);
  a14=(a14*a7);
  a10=(a10+a14);
  a14=arg[0]? arg[0][5] : 0;
  a7=sin(a14);
  a23=sin(a6);
  a21=(a7*a23);
  a25=cos(a14);
  a12=cos(a6);
  a24=(a25*a12);
  a21=(a21-a24);
  a24=(a5*a21);
  a26=1.2500000000000000e+00;
  a27=(a26*a23);
  a24=(a24-a27);
  a28=7.5000000000000000e-01;
  a29=(a28*a23);
  a30=(a24+a29);
  a31=(a28*a12);
  a32=(a26*a12);
  a33=(a25*a23);
  a34=(a7*a12);
  a33=(a33+a34);
  a34=(a5*a33);
  a34=(a32-a34);
  a31=(a31-a34);
  a31=(a30*a31);
  a35=(a5*a33);
  a35=(a32-a35);
  a36=(a28*a12);
  a37=(a35-a36);
  a38=(a5*a21);
  a38=(a38-a27);
  a28=(a28*a23);
  a28=(a38+a28);
  a28=(a37*a28);
  a31=(a31+a28);
  a28=(a24+a29);
  a28=casadi_sq(a28);
  a39=(a35-a36);
  a39=casadi_sq(a39);
  a28=(a28+a39);
  a28=sqrt(a28);
  a31=(a31/a28);
  a39=arg[0]? arg[0][2] : 0;
  a40=arg[2]? arg[2][2] : 0;
  a41=(a40<a13);
  a40=(a41?a40:0);
  a41=(!a41);
  a41=(a41?a13:0);
  a40=(a40+a41);
  a24=(a24+a29);
  a24=casadi_sq(a24);
  a35=(a35-a36);
  a35=casadi_sq(a35);
  a24=(a24+a35);
  a24=sqrt(a24);
  a24=(a24-a13);
  a24=(a24/a11);
  a35=(a24/a9);
  a35=(a35-a13);
  a35=casadi_sq(a35);
  a35=(a35/a15);
  a35=(-a35);
  a35=exp(a35);
  a40=(a40*a35);
  a35=(a31*a1);
  a36=(a7*a12);
  a29=(a25*a23);
  a36=(a36+a29);
  a29=(a21*a27);
  a41=(a33*a32);
  a29=(a29+a41);
  a41=(a36*a29);
  a36=(a36*a27);
  a25=(a25*a12);
  a7=(a7*a23);
  a25=(a25-a7);
  a32=(a25*a32);
  a36=(a36+a32);
  a21=(a21*a36);
  a41=(a41-a21);
  a41=(a41-a34);
  a30=(a30*a41);
  a33=(a33*a36);
  a25=(a25*a29);
  a33=(a33-a25);
  a33=(a33+a38);
  a37=(a37*a33);
  a30=(a30+a37);
  a30=(a30/a28);
  a28=(a30*a2);
  a35=(a35+a28);
  a28=(a35<=a16);
  a37=fabs(a35);
  a37=(a37/a19);
  a37=(a13-a37);
  a33=fabs(a35);
  a33=(a33/a19);
  a33=(a13+a33);
  a37=(a37/a33);
  a37=(a28?a37:0);
  a28=(!a28);
  a33=(a20*a35);
  a33=(a33/a19);
  a33=(a33/a22);
  a33=(a13-a33);
  a38=(a35/a19);
  a38=(a38/a22);
  a38=(a13-a38);
  a33=(a33/a38);
  a28=(a28?a33:0);
  a37=(a37+a28);
  a40=(a40*a37);
  a37=(a13<a24);
  a24=(a24/a9);
  a24=(a24-a13);
  a24=(a19*a24);
  a24=exp(a24);
  a24=(a24-a13);
  a24=(a24/a17);
  a37=(a37?a24:0);
  a40=(a40+a37);
  a35=(a35/a8);
  a35=(a18*a35);
  a40=(a40+a35);
  a39=(a39*a40);
  a31=(a31*a39);
  a10=(a10+a31);
  a31=sin(a14);
  a40=sin(a6);
  a35=(a31*a40);
  a37=cos(a14);
  a24=cos(a6);
  a28=(a37*a24);
  a35=(a35-a28);
  a28=(a5*a35);
  a33=(a26*a40);
  a28=(a28-a33);
  a38=1.7500000000000000e+00;
  a25=(a38*a40);
  a29=(a28+a25);
  a36=(a38*a24);
  a41=(a26*a24);
  a34=(a37*a40);
  a21=(a31*a24);
  a34=(a34+a21);
  a21=(a5*a34);
  a21=(a41-a21);
  a36=(a36-a21);
  a36=(a29*a36);
  a32=(a5*a34);
  a32=(a41-a32);
  a7=(a38*a24);
  a23=(a32-a7);
  a5=(a5*a35);
  a5=(a5-a33);
  a38=(a38*a40);
  a38=(a5+a38);
  a38=(a23*a38);
  a36=(a36+a38);
  a38=(a28+a25);
  a38=casadi_sq(a38);
  a12=(a32-a7);
  a12=casadi_sq(a12);
  a38=(a38+a12);
  a38=sqrt(a38);
  a36=(a36/a38);
  a12=arg[0]? arg[0][3] : 0;
  a27=arg[2]? arg[2][3] : 0;
  a42=(a27<a13);
  a27=(a42?a27:0);
  a42=(!a42);
  a42=(a42?a13:0);
  a27=(a27+a42);
  a28=(a28+a25);
  a28=casadi_sq(a28);
  a32=(a32-a7);
  a32=casadi_sq(a32);
  a28=(a28+a32);
  a28=sqrt(a28);
  a28=(a28-a13);
  a28=(a28/a11);
  a11=(a28/a9);
  a11=(a11-a13);
  a11=casadi_sq(a11);
  a11=(a11/a15);
  a11=(-a11);
  a11=exp(a11);
  a27=(a27*a11);
  a11=(a36*a1);
  a15=(a31*a24);
  a32=(a37*a40);
  a15=(a15+a32);
  a32=(a35*a33);
  a7=(a34*a41);
  a32=(a32+a7);
  a7=(a15*a32);
  a15=(a15*a33);
  a37=(a37*a24);
  a31=(a31*a40);
  a37=(a37-a31);
  a41=(a37*a41);
  a15=(a15+a41);
  a35=(a35*a15);
  a7=(a7-a35);
  a7=(a7-a21);
  a29=(a29*a7);
  a34=(a34*a15);
  a37=(a37*a32);
  a34=(a34-a37);
  a34=(a34+a5);
  a23=(a23*a34);
  a29=(a29+a23);
  a29=(a29/a38);
  a38=(a29*a2);
  a11=(a11+a38);
  a16=(a11<=a16);
  a38=fabs(a11);
  a38=(a38/a19);
  a38=(a13-a38);
  a23=fabs(a11);
  a23=(a23/a19);
  a23=(a13+a23);
  a38=(a38/a23);
  a38=(a16?a38:0);
  a16=(!a16);
  a20=(a20*a11);
  a20=(a20/a19);
  a20=(a20/a22);
  a20=(a13-a20);
  a23=(a11/a19);
  a23=(a23/a22);
  a23=(a13-a23);
  a20=(a20/a23);
  a16=(a16?a20:0);
  a38=(a38+a16);
  a27=(a27*a38);
  a38=(a13<a28);
  a28=(a28/a9);
  a28=(a28-a13);
  a19=(a19*a28);
  a19=exp(a19);
  a19=(a19-a13);
  a19=(a19/a17);
  a38=(a38?a19:0);
  a27=(a27+a38);
  a11=(a11/a8);
  a18=(a18*a11);
  a27=(a27+a18);
  a12=(a12*a27);
  a36=(a36*a12);
  a10=(a10+a36);
  a36=sin(a14);
  a14=cos(a14);
  a27=9.8100000000000005e+00;
  a18=cos(a6);
  a18=(a27*a18);
  a11=(a14*a18);
  a6=sin(a6);
  a27=(a27*a6);
  a6=(a36*a27);
  a11=(a11-a6);
  a6=(a26*a1);
  a8=(a14*a6);
  a38=(a8*a2);
  a11=(a11+a38);
  a1=(a1+a2);
  a8=(a1*a8);
  a11=(a11-a8);
  a11=(a36*a11);
  a6=(a36*a6);
  a1=(a1*a6);
  a27=(a14*a27);
  a36=(a36*a18);
  a27=(a27+a36);
  a6=(a6*a2);
  a27=(a27+a6);
  a1=(a1-a27);
  a14=(a14*a1);
  a11=(a11+a14);
  a26=(a26*a11);
  a10=(a10+a26);
  a4=(a4*a10);
  a26=9.6278838983177639e-01;
  a30=(a30*a39);
  a29=(a29*a12);
  a30=(a30+a29);
  a26=(a26*a30);
  a4=(a4+a26);
  a26=6.9253199970355839e-01;
  a4=(a4/a26);
  a3=(a3*a4);
  a26=9.6278838983177628e-01;
  a26=(a26*a10);
  a10=2.7025639012821789e-01;
  a10=(a10*a30);
  a26=(a26+a10);
  a3=(a3-a26);
  a26=3.7001900289039211e+00;
  a3=(a3/a26);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_11434985_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_11434985_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_11434985_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_11434985_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_11434985_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_11434985_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_11434985_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_11434985_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_19_11434985_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_19_11434985_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_19_11434985_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_19_11434985_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_19_11434985_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_19_11434985_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_19_11434985_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_11434985_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
