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
  #define CASADI_PREFIX(ID) model_2021_04_09_15201944_impl_dae_fun_ ## ID
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

/* model_2021_04_09_15201944_impl_dae_fun:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8]) */
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
  a5=arg[0]? arg[0][4] : 0;
  a6=sin(a5);
  a7=arg[0]? arg[0][0] : 0;
  a8=(a6*a7);
  a9=-5.0000000000000000e-01;
  a10=(a8+a9);
  a11=cos(a5);
  a12=(a11*a7);
  a10=(a10*a12);
  a11=(a11*a7);
  a6=(a6*a7);
  a6=(a11*a6);
  a10=(a10-a6);
  a6=(a8+a9);
  a6=casadi_sq(a6);
  a7=casadi_sq(a11);
  a6=(a6+a7);
  a6=sqrt(a6);
  a10=(a10/a6);
  a6=700.;
  a7=arg[2]? arg[2][0] : 0;
  a8=(a8+a9);
  a8=casadi_sq(a8);
  a11=casadi_sq(a11);
  a8=(a8+a11);
  a8=sqrt(a8);
  a11=4.0000000000000001e-02;
  a8=(a8-a11);
  a9=8.7758256189037276e-01;
  a8=(a8/a9);
  a12=6.9999999999999996e-01;
  a13=(a8/a12);
  a14=1.;
  a13=(a13-a14);
  a13=casadi_sq(a13);
  a15=4.5000000000000001e-01;
  a13=(a13/a15);
  a13=(-a13);
  a13=exp(a13);
  a7=(a7*a13);
  a13=(a10*a1);
  a16=0.;
  a17=(a13<=a16);
  a18=fabs(a13);
  a19=10.;
  a18=(a18/a19);
  a18=(a14-a18);
  a20=fabs(a13);
  a20=(a20/a19);
  a20=(a14+a20);
  a18=(a18/a20);
  a18=(a17?a18:0);
  a17=(!a17);
  a20=1.3300000000000001e+00;
  a21=(a20*a13);
  a21=(a21/a19);
  a22=-8.2500000000000004e-02;
  a21=(a21/a22);
  a21=(a14-a21);
  a23=(a13/a19);
  a23=(a23/a22);
  a23=(a14-a23);
  a21=(a21/a23);
  a17=(a17?a21:0);
  a18=(a18+a17);
  a7=(a7*a18);
  a18=(a11<a8);
  a8=(a8/a12);
  a8=(a8-a14);
  a8=(a19*a8);
  a8=exp(a8);
  a8=(a8-a14);
  a17=1.4741315910257660e+02;
  a8=(a8/a17);
  a18=(a18?a8:0);
  a7=(a7+a18);
  a18=1.0000000000000001e-01;
  a8=7.;
  a13=(a13/a8);
  a13=(a18*a13);
  a7=(a7+a13);
  a7=(a6*a7);
  a10=(a10*a7);
  a7=sin(a5);
  a13=arg[0]? arg[0][1] : 0;
  a21=(a7*a13);
  a23=5.0000000000000000e-01;
  a24=(a21+a23);
  a25=cos(a5);
  a26=(a25*a13);
  a24=(a24*a26);
  a25=(a25*a13);
  a7=(a7*a13);
  a7=(a25*a7);
  a24=(a24-a7);
  a7=(a21+a23);
  a7=casadi_sq(a7);
  a13=casadi_sq(a25);
  a7=(a7+a13);
  a7=sqrt(a7);
  a24=(a24/a7);
  a7=arg[2]? arg[2][1] : 0;
  a21=(a21+a23);
  a21=casadi_sq(a21);
  a25=casadi_sq(a25);
  a21=(a21+a25);
  a21=sqrt(a21);
  a21=(a21-a11);
  a21=(a21/a9);
  a25=(a21/a12);
  a25=(a25-a14);
  a25=casadi_sq(a25);
  a25=(a25/a15);
  a25=(-a25);
  a25=exp(a25);
  a7=(a7*a25);
  a25=(a24*a1);
  a23=(a25<=a16);
  a13=fabs(a25);
  a13=(a13/a19);
  a13=(a14-a13);
  a26=fabs(a25);
  a26=(a26/a19);
  a26=(a14+a26);
  a13=(a13/a26);
  a13=(a23?a13:0);
  a23=(!a23);
  a26=(a20*a25);
  a26=(a26/a19);
  a26=(a26/a22);
  a26=(a14-a26);
  a27=(a25/a19);
  a27=(a27/a22);
  a27=(a14-a27);
  a26=(a26/a27);
  a23=(a23?a26:0);
  a13=(a13+a23);
  a7=(a7*a13);
  a13=(a11<a21);
  a21=(a21/a12);
  a21=(a21-a14);
  a21=(a19*a21);
  a21=exp(a21);
  a21=(a21-a14);
  a21=(a21/a17);
  a13=(a13?a21:0);
  a7=(a7+a13);
  a25=(a25/a8);
  a25=(a18*a25);
  a7=(a7+a25);
  a7=(a6*a7);
  a24=(a24*a7);
  a10=(a10+a24);
  a24=arg[0]? arg[0][5] : 0;
  a7=sin(a24);
  a25=sin(a5);
  a13=(a7*a25);
  a21=cos(a24);
  a23=cos(a5);
  a26=(a21*a23);
  a13=(a13-a26);
  a26=arg[0]? arg[0][2] : 0;
  a27=(a13*a26);
  a28=1.2500000000000000e+00;
  a29=(a28*a25);
  a27=(a27-a29);
  a30=7.5000000000000000e-01;
  a31=(a30*a25);
  a32=(a27+a31);
  a33=(a30*a23);
  a34=(a28*a23);
  a35=(a21*a25);
  a36=(a7*a23);
  a35=(a35+a36);
  a36=(a35*a26);
  a36=(a34-a36);
  a33=(a33-a36);
  a33=(a32*a33);
  a37=(a35*a26);
  a37=(a34-a37);
  a38=(a30*a23);
  a39=(a37-a38);
  a26=(a13*a26);
  a26=(a26-a29);
  a30=(a30*a25);
  a30=(a26+a30);
  a30=(a39*a30);
  a33=(a33+a30);
  a30=(a27+a31);
  a30=casadi_sq(a30);
  a40=(a37-a38);
  a40=casadi_sq(a40);
  a30=(a30+a40);
  a30=sqrt(a30);
  a33=(a33/a30);
  a40=arg[2]? arg[2][2] : 0;
  a27=(a27+a31);
  a27=casadi_sq(a27);
  a37=(a37-a38);
  a37=casadi_sq(a37);
  a27=(a27+a37);
  a27=sqrt(a27);
  a27=(a27-a11);
  a27=(a27/a9);
  a37=(a27/a12);
  a37=(a37-a14);
  a37=casadi_sq(a37);
  a37=(a37/a15);
  a37=(-a37);
  a37=exp(a37);
  a40=(a40*a37);
  a37=(a33*a1);
  a38=(a7*a23);
  a31=(a21*a25);
  a38=(a38+a31);
  a31=(a13*a29);
  a41=(a35*a34);
  a31=(a31+a41);
  a41=(a38*a31);
  a38=(a38*a29);
  a21=(a21*a23);
  a7=(a7*a25);
  a21=(a21-a7);
  a34=(a21*a34);
  a38=(a38+a34);
  a13=(a13*a38);
  a41=(a41-a13);
  a41=(a41-a36);
  a32=(a32*a41);
  a35=(a35*a38);
  a21=(a21*a31);
  a35=(a35-a21);
  a35=(a35+a26);
  a39=(a39*a35);
  a32=(a32+a39);
  a32=(a32/a30);
  a30=(a32*a2);
  a37=(a37+a30);
  a30=(a37<=a16);
  a39=fabs(a37);
  a39=(a39/a19);
  a39=(a14-a39);
  a35=fabs(a37);
  a35=(a35/a19);
  a35=(a14+a35);
  a39=(a39/a35);
  a39=(a30?a39:0);
  a30=(!a30);
  a35=(a20*a37);
  a35=(a35/a19);
  a35=(a35/a22);
  a35=(a14-a35);
  a26=(a37/a19);
  a26=(a26/a22);
  a26=(a14-a26);
  a35=(a35/a26);
  a30=(a30?a35:0);
  a39=(a39+a30);
  a40=(a40*a39);
  a39=(a11<a27);
  a27=(a27/a12);
  a27=(a27-a14);
  a27=(a19*a27);
  a27=exp(a27);
  a27=(a27-a14);
  a27=(a27/a17);
  a39=(a39?a27:0);
  a40=(a40+a39);
  a37=(a37/a8);
  a37=(a18*a37);
  a40=(a40+a37);
  a40=(a6*a40);
  a33=(a33*a40);
  a10=(a10+a33);
  a33=sin(a24);
  a37=sin(a5);
  a39=(a33*a37);
  a27=cos(a24);
  a30=cos(a5);
  a35=(a27*a30);
  a39=(a39-a35);
  a35=arg[0]? arg[0][3] : 0;
  a26=(a39*a35);
  a21=(a28*a37);
  a26=(a26-a21);
  a31=1.7500000000000000e+00;
  a38=(a31*a37);
  a41=(a26+a38);
  a36=(a31*a30);
  a13=(a28*a30);
  a34=(a27*a37);
  a7=(a33*a30);
  a34=(a34+a7);
  a7=(a34*a35);
  a7=(a13-a7);
  a36=(a36-a7);
  a36=(a41*a36);
  a25=(a34*a35);
  a25=(a13-a25);
  a23=(a31*a30);
  a29=(a25-a23);
  a35=(a39*a35);
  a35=(a35-a21);
  a31=(a31*a37);
  a31=(a35+a31);
  a31=(a29*a31);
  a36=(a36+a31);
  a31=(a26+a38);
  a31=casadi_sq(a31);
  a42=(a25-a23);
  a42=casadi_sq(a42);
  a31=(a31+a42);
  a31=sqrt(a31);
  a36=(a36/a31);
  a42=arg[2]? arg[2][3] : 0;
  a26=(a26+a38);
  a26=casadi_sq(a26);
  a25=(a25-a23);
  a25=casadi_sq(a25);
  a26=(a26+a25);
  a26=sqrt(a26);
  a26=(a26-a11);
  a26=(a26/a9);
  a9=(a26/a12);
  a9=(a9-a14);
  a9=casadi_sq(a9);
  a9=(a9/a15);
  a9=(-a9);
  a9=exp(a9);
  a42=(a42*a9);
  a9=(a36*a1);
  a15=(a33*a30);
  a25=(a27*a37);
  a15=(a15+a25);
  a25=(a39*a21);
  a23=(a34*a13);
  a25=(a25+a23);
  a23=(a15*a25);
  a15=(a15*a21);
  a27=(a27*a30);
  a33=(a33*a37);
  a27=(a27-a33);
  a13=(a27*a13);
  a15=(a15+a13);
  a39=(a39*a15);
  a23=(a23-a39);
  a23=(a23-a7);
  a41=(a41*a23);
  a34=(a34*a15);
  a27=(a27*a25);
  a34=(a34-a27);
  a34=(a34+a35);
  a29=(a29*a34);
  a41=(a41+a29);
  a41=(a41/a31);
  a31=(a41*a2);
  a9=(a9+a31);
  a16=(a9<=a16);
  a31=fabs(a9);
  a31=(a31/a19);
  a31=(a14-a31);
  a29=fabs(a9);
  a29=(a29/a19);
  a29=(a14+a29);
  a31=(a31/a29);
  a31=(a16?a31:0);
  a16=(!a16);
  a20=(a20*a9);
  a20=(a20/a19);
  a20=(a20/a22);
  a20=(a14-a20);
  a29=(a9/a19);
  a29=(a29/a22);
  a29=(a14-a29);
  a20=(a20/a29);
  a16=(a16?a20:0);
  a31=(a31+a16);
  a42=(a42*a31);
  a11=(a11<a26);
  a26=(a26/a12);
  a26=(a26-a14);
  a19=(a19*a26);
  a19=exp(a19);
  a19=(a19-a14);
  a19=(a19/a17);
  a11=(a11?a19:0);
  a42=(a42+a11);
  a9=(a9/a8);
  a18=(a18*a9);
  a42=(a42+a18);
  a6=(a6*a42);
  a36=(a36*a6);
  a10=(a10+a36);
  a36=sin(a24);
  a24=cos(a24);
  a42=9.8100000000000005e+00;
  a18=cos(a5);
  a18=(a42*a18);
  a9=(a24*a18);
  a5=sin(a5);
  a42=(a42*a5);
  a5=(a36*a42);
  a9=(a9-a5);
  a5=(a28*a1);
  a8=(a24*a5);
  a11=(a8*a2);
  a9=(a9+a11);
  a1=(a1+a2);
  a8=(a1*a8);
  a9=(a9-a8);
  a9=(a36*a9);
  a5=(a36*a5);
  a1=(a1*a5);
  a42=(a24*a42);
  a36=(a36*a18);
  a42=(a42+a36);
  a5=(a5*a2);
  a42=(a42+a5);
  a1=(a1-a42);
  a24=(a24*a1);
  a9=(a9+a24);
  a28=(a28*a9);
  a10=(a10+a28);
  a4=(a4*a10);
  a28=9.6278838983177639e-01;
  a32=(a32*a40);
  a41=(a41*a6);
  a32=(a32+a41);
  a28=(a28*a32);
  a4=(a4+a28);
  a28=6.9253199970355839e-01;
  a4=(a4/a28);
  a3=(a3*a4);
  a28=9.6278838983177628e-01;
  a28=(a28*a10);
  a10=2.7025639012821789e-01;
  a10=(a10*a32);
  a28=(a28+a10);
  a3=(a3-a28);
  a28=3.7001900289039211e+00;
  a3=(a3/a28);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201944_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201944_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201944_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201944_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201944_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201944_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201944_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201944_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201944_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201944_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15201944_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201944_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201944_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201944_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201944_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201944_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
