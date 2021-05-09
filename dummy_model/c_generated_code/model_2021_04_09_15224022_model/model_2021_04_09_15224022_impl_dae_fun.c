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
  #define CASADI_PREFIX(ID) model_2021_04_09_15224022_impl_dae_fun_ ## ID
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

/* model_2021_04_09_15224022_impl_dae_fun:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8]) */
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
  a6=3.0039062417372782e+02;
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
  a6=(a6*a7);
  a10=(a10*a6);
  a6=sin(a5);
  a7=arg[0]? arg[0][1] : 0;
  a13=(a6*a7);
  a21=5.0000000000000000e-01;
  a23=(a13+a21);
  a24=cos(a5);
  a25=(a24*a7);
  a23=(a23*a25);
  a24=(a24*a7);
  a6=(a6*a7);
  a6=(a24*a6);
  a23=(a23-a6);
  a6=(a13+a21);
  a6=casadi_sq(a6);
  a7=casadi_sq(a24);
  a6=(a6+a7);
  a6=sqrt(a6);
  a23=(a23/a6);
  a6=1.1995117197415834e+03;
  a7=arg[2]? arg[2][1] : 0;
  a13=(a13+a21);
  a13=casadi_sq(a13);
  a24=casadi_sq(a24);
  a13=(a13+a24);
  a13=sqrt(a13);
  a13=(a13-a11);
  a13=(a13/a9);
  a24=(a13/a12);
  a24=(a24-a14);
  a24=casadi_sq(a24);
  a24=(a24/a15);
  a24=(-a24);
  a24=exp(a24);
  a7=(a7*a24);
  a24=(a23*a1);
  a21=(a24<=a16);
  a25=fabs(a24);
  a25=(a25/a19);
  a25=(a14-a25);
  a26=fabs(a24);
  a26=(a26/a19);
  a26=(a14+a26);
  a25=(a25/a26);
  a25=(a21?a25:0);
  a21=(!a21);
  a26=(a20*a24);
  a26=(a26/a19);
  a26=(a26/a22);
  a26=(a14-a26);
  a27=(a24/a19);
  a27=(a27/a22);
  a27=(a14-a27);
  a26=(a26/a27);
  a21=(a21?a26:0);
  a25=(a25+a21);
  a7=(a7*a25);
  a25=(a11<a13);
  a13=(a13/a12);
  a13=(a13-a14);
  a13=(a19*a13);
  a13=exp(a13);
  a13=(a13-a14);
  a13=(a13/a17);
  a25=(a25?a13:0);
  a7=(a7+a25);
  a24=(a24/a8);
  a24=(a18*a24);
  a7=(a7+a24);
  a6=(a6*a7);
  a23=(a23*a6);
  a10=(a10+a23);
  a23=arg[0]? arg[0][5] : 0;
  a6=sin(a23);
  a7=sin(a5);
  a24=(a6*a7);
  a25=cos(a23);
  a13=cos(a5);
  a21=(a25*a13);
  a24=(a24-a21);
  a21=arg[0]? arg[0][2] : 0;
  a26=(a24*a21);
  a27=1.2500000000000000e+00;
  a28=(a27*a7);
  a26=(a26-a28);
  a29=7.5000000000000000e-01;
  a30=(a29*a7);
  a31=(a26+a30);
  a32=(a29*a13);
  a33=(a27*a13);
  a34=(a25*a7);
  a35=(a6*a13);
  a34=(a34+a35);
  a35=(a34*a21);
  a35=(a33-a35);
  a32=(a32-a35);
  a32=(a31*a32);
  a36=(a34*a21);
  a36=(a33-a36);
  a37=(a29*a13);
  a38=(a36-a37);
  a21=(a24*a21);
  a21=(a21-a28);
  a29=(a29*a7);
  a29=(a21+a29);
  a29=(a38*a29);
  a32=(a32+a29);
  a29=(a26+a30);
  a29=casadi_sq(a29);
  a39=(a36-a37);
  a39=casadi_sq(a39);
  a29=(a29+a39);
  a29=sqrt(a29);
  a32=(a32/a29);
  a39=1.1995117190319347e+03;
  a40=arg[2]? arg[2][2] : 0;
  a26=(a26+a30);
  a26=casadi_sq(a26);
  a36=(a36-a37);
  a36=casadi_sq(a36);
  a26=(a26+a36);
  a26=sqrt(a26);
  a26=(a26-a11);
  a26=(a26/a9);
  a36=(a26/a12);
  a36=(a36-a14);
  a36=casadi_sq(a36);
  a36=(a36/a15);
  a36=(-a36);
  a36=exp(a36);
  a40=(a40*a36);
  a36=(a32*a1);
  a37=(a6*a13);
  a30=(a25*a7);
  a37=(a37+a30);
  a30=(a24*a28);
  a41=(a34*a33);
  a30=(a30+a41);
  a41=(a37*a30);
  a37=(a37*a28);
  a25=(a25*a13);
  a6=(a6*a7);
  a25=(a25-a6);
  a33=(a25*a33);
  a37=(a37+a33);
  a24=(a24*a37);
  a41=(a41-a24);
  a41=(a41-a35);
  a31=(a31*a41);
  a34=(a34*a37);
  a25=(a25*a30);
  a34=(a34-a25);
  a34=(a34+a21);
  a38=(a38*a34);
  a31=(a31+a38);
  a31=(a31/a29);
  a29=(a31*a2);
  a36=(a36+a29);
  a29=(a36<=a16);
  a38=fabs(a36);
  a38=(a38/a19);
  a38=(a14-a38);
  a34=fabs(a36);
  a34=(a34/a19);
  a34=(a14+a34);
  a38=(a38/a34);
  a38=(a29?a38:0);
  a29=(!a29);
  a34=(a20*a36);
  a34=(a34/a19);
  a34=(a34/a22);
  a34=(a14-a34);
  a21=(a36/a19);
  a21=(a21/a22);
  a21=(a14-a21);
  a34=(a34/a21);
  a29=(a29?a34:0);
  a38=(a38+a29);
  a40=(a40*a38);
  a38=(a11<a26);
  a26=(a26/a12);
  a26=(a26-a14);
  a26=(a19*a26);
  a26=exp(a26);
  a26=(a26-a14);
  a26=(a26/a17);
  a38=(a38?a26:0);
  a40=(a40+a38);
  a36=(a36/a8);
  a36=(a18*a36);
  a40=(a40+a36);
  a39=(a39*a40);
  a32=(a32*a39);
  a10=(a10+a32);
  a32=sin(a23);
  a40=sin(a5);
  a36=(a32*a40);
  a38=cos(a23);
  a26=cos(a5);
  a29=(a38*a26);
  a36=(a36-a29);
  a29=arg[0]? arg[0][3] : 0;
  a34=(a36*a29);
  a21=(a27*a40);
  a34=(a34-a21);
  a25=1.7500000000000000e+00;
  a30=(a25*a40);
  a37=(a34+a30);
  a41=(a25*a26);
  a35=(a27*a26);
  a24=(a38*a40);
  a33=(a32*a26);
  a24=(a24+a33);
  a33=(a24*a29);
  a33=(a35-a33);
  a41=(a41-a33);
  a41=(a37*a41);
  a6=(a24*a29);
  a6=(a35-a6);
  a7=(a25*a26);
  a13=(a6-a7);
  a29=(a36*a29);
  a29=(a29-a21);
  a25=(a25*a40);
  a25=(a29+a25);
  a25=(a13*a25);
  a41=(a41+a25);
  a25=(a34+a30);
  a25=casadi_sq(a25);
  a28=(a6-a7);
  a28=casadi_sq(a28);
  a25=(a25+a28);
  a25=sqrt(a25);
  a41=(a41/a25);
  a28=3.0039062454858475e+02;
  a42=arg[2]? arg[2][3] : 0;
  a34=(a34+a30);
  a34=casadi_sq(a34);
  a6=(a6-a7);
  a6=casadi_sq(a6);
  a34=(a34+a6);
  a34=sqrt(a34);
  a34=(a34-a11);
  a34=(a34/a9);
  a9=(a34/a12);
  a9=(a9-a14);
  a9=casadi_sq(a9);
  a9=(a9/a15);
  a9=(-a9);
  a9=exp(a9);
  a42=(a42*a9);
  a9=(a41*a1);
  a15=(a32*a26);
  a6=(a38*a40);
  a15=(a15+a6);
  a6=(a36*a21);
  a7=(a24*a35);
  a6=(a6+a7);
  a7=(a15*a6);
  a15=(a15*a21);
  a38=(a38*a26);
  a32=(a32*a40);
  a38=(a38-a32);
  a35=(a38*a35);
  a15=(a15+a35);
  a36=(a36*a15);
  a7=(a7-a36);
  a7=(a7-a33);
  a37=(a37*a7);
  a24=(a24*a15);
  a38=(a38*a6);
  a24=(a24-a38);
  a24=(a24+a29);
  a13=(a13*a24);
  a37=(a37+a13);
  a37=(a37/a25);
  a25=(a37*a2);
  a9=(a9+a25);
  a16=(a9<=a16);
  a25=fabs(a9);
  a25=(a25/a19);
  a25=(a14-a25);
  a13=fabs(a9);
  a13=(a13/a19);
  a13=(a14+a13);
  a25=(a25/a13);
  a25=(a16?a25:0);
  a16=(!a16);
  a20=(a20*a9);
  a20=(a20/a19);
  a20=(a20/a22);
  a20=(a14-a20);
  a13=(a9/a19);
  a13=(a13/a22);
  a13=(a14-a13);
  a20=(a20/a13);
  a16=(a16?a20:0);
  a25=(a25+a16);
  a42=(a42*a25);
  a11=(a11<a34);
  a34=(a34/a12);
  a34=(a34-a14);
  a19=(a19*a34);
  a19=exp(a19);
  a19=(a19-a14);
  a19=(a19/a17);
  a11=(a11?a19:0);
  a42=(a42+a11);
  a9=(a9/a8);
  a18=(a18*a9);
  a42=(a42+a18);
  a28=(a28*a42);
  a41=(a41*a28);
  a10=(a10+a41);
  a41=sin(a23);
  a23=cos(a23);
  a42=9.8100000000000005e+00;
  a18=cos(a5);
  a18=(a42*a18);
  a9=(a23*a18);
  a5=sin(a5);
  a42=(a42*a5);
  a5=(a41*a42);
  a9=(a9-a5);
  a5=(a27*a1);
  a8=(a23*a5);
  a11=(a8*a2);
  a9=(a9+a11);
  a1=(a1+a2);
  a8=(a1*a8);
  a9=(a9-a8);
  a9=(a41*a9);
  a5=(a41*a5);
  a1=(a1*a5);
  a42=(a23*a42);
  a41=(a41*a18);
  a42=(a42+a41);
  a5=(a5*a2);
  a42=(a42+a5);
  a1=(a1-a42);
  a23=(a23*a1);
  a9=(a9+a23);
  a27=(a27*a9);
  a10=(a10+a27);
  a4=(a4*a10);
  a27=9.6278838983177639e-01;
  a31=(a31*a39);
  a37=(a37*a28);
  a31=(a31+a37);
  a27=(a27*a31);
  a4=(a4+a27);
  a27=6.9253199970355839e-01;
  a4=(a4/a27);
  a3=(a3*a4);
  a27=9.6278838983177628e-01;
  a27=(a27*a10);
  a10=2.7025639012821789e-01;
  a10=(a10*a31);
  a27=(a27+a10);
  a3=(a3-a27);
  a27=3.7001900289039211e+00;
  a3=(a3/a27);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15224022_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15224022_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15224022_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15224022_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15224022_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15224022_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15224022_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15224022_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15224022_impl_dae_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15224022_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15224022_impl_dae_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15224022_impl_dae_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15224022_impl_dae_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15224022_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15224022_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15224022_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
