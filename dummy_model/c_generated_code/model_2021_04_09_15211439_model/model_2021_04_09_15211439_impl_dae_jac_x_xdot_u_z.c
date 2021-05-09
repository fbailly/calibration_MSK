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
  #define CASADI_PREFIX(ID) model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_ ## ID
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
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_sign CASADI_PREFIX(sign)
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

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[29] = {8, 8, 0, 2, 4, 6, 8, 10, 12, 15, 18, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 4, 6, 7, 5, 6, 7};
static const casadi_int casadi_s4[19] = {8, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s5[15] = {8, 4, 0, 2, 4, 6, 8, 6, 7, 6, 7, 6, 7, 6, 7};
static const casadi_int casadi_s6[3] = {8, 0, 0};

/* model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8x8,18nz],o1[8x8,8nz],o2[8x4,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[0]? arg[0][4] : 0;
  a1=cos(a0);
  a2=arg[0]? arg[0][0] : 0;
  a3=(a1*a2);
  a4=(a3+a3);
  a5=1.1394939273245490e+00;
  a6=4.0000000000000001e-02;
  a7=sin(a0);
  a8=(a7*a2);
  a9=-5.0000000000000000e-01;
  a10=(a8+a9);
  a11=casadi_sq(a10);
  a12=casadi_sq(a3);
  a11=(a11+a12);
  a11=sqrt(a11);
  a12=(a11-a6);
  a13=8.7758256189037276e-01;
  a12=(a12/a13);
  a14=(a6<a12);
  a15=1.4285714285714286e+00;
  a16=10.;
  a17=6.7836549063042314e-03;
  a18=5.3430360530311259e+02;
  a19=3.9024390243902418e-01;
  a20=(a8+a9);
  a21=(a1*a2);
  a22=(a20*a21);
  a23=(a7*a2);
  a24=(a3*a23);
  a22=(a22-a24);
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a24=casadi_sq(a3);
  a9=(a9+a24);
  a9=sqrt(a9);
  a22=(a22/a9);
  a24=(a19*a22);
  a24=(a18*a24);
  a25=(a17*a24);
  a26=6.9999999999999996e-01;
  a27=(a12/a26);
  a28=1.;
  a27=(a27-a28);
  a27=(a16*a27);
  a27=exp(a27);
  a25=(a25*a27);
  a25=(a16*a25);
  a25=(a15*a25);
  a25=(a14?a25:0);
  a12=(a12/a26);
  a12=(a12-a28);
  a29=(a12+a12);
  a30=2.2222222222222223e+00;
  a12=casadi_sq(a12);
  a31=4.5000000000000001e-01;
  a12=(a12/a31);
  a12=(-a12);
  a12=exp(a12);
  a32=arg[2]? arg[2][0] : 0;
  a33=arg[0]? arg[0][6] : 0;
  a34=(a22*a33);
  a35=0.;
  a36=(a34<=a35);
  a37=fabs(a34);
  a37=(a37/a16);
  a37=(a28-a37);
  a38=fabs(a34);
  a38=(a38/a16);
  a38=(a28+a38);
  a37=(a37/a38);
  a39=(a36?a37:0);
  a40=(!a36);
  a41=1.3300000000000001e+00;
  a42=(a41*a34);
  a42=(a42/a16);
  a43=-8.2500000000000004e-02;
  a42=(a42/a43);
  a42=(a28-a42);
  a44=(a34/a16);
  a44=(a44/a43);
  a44=(a28-a44);
  a42=(a42/a44);
  a45=(a40?a42:0);
  a39=(a39+a45);
  a45=(a39*a24);
  a45=(a32*a45);
  a45=(a12*a45);
  a45=(a30*a45);
  a45=(a29*a45);
  a45=(a15*a45);
  a25=(a25-a45);
  a25=(a5*a25);
  a11=(a11+a11);
  a25=(a25/a11);
  a45=(a4*a25);
  a46=(a3+a3);
  a47=(a22/a9);
  a48=(a32*a12);
  a49=(a48*a39);
  a50=(a27-a28);
  a51=1.4741315910257660e+02;
  a50=(a50/a51);
  a50=(a14?a50:0);
  a49=(a49+a50);
  a50=1.0000000000000001e-01;
  a52=7.;
  a53=(a34/a52);
  a53=(a50*a53);
  a49=(a49+a53);
  a49=(a18*a49);
  a53=(a19*a49);
  a54=1.4285714285714285e-01;
  a55=(a50*a24);
  a55=(a54*a55);
  a56=-1.2121212121212121e+01;
  a24=(a48*a24);
  a42=(a42/a44);
  a57=(a24*a42);
  a57=(a56*a57);
  a57=(a50*a57);
  a57=(a40?a57:0);
  a55=(a55+a57);
  a57=(a24/a44);
  a57=(a56*a57);
  a57=(a50*a57);
  a57=(a41*a57);
  a57=(-a57);
  a57=(a40?a57:0);
  a55=(a55+a57);
  a37=(a37/a38);
  a57=(a24*a37);
  a57=(a50*a57);
  a58=casadi_sign(a34);
  a57=(a57*a58);
  a57=(-a57);
  a57=(a36?a57:0);
  a55=(a55+a57);
  a24=(a24/a38);
  a24=(a50*a24);
  a34=casadi_sign(a34);
  a24=(a24*a34);
  a24=(-a24);
  a24=(a36?a24:0);
  a55=(a55+a24);
  a24=(a33*a55);
  a53=(a53+a24);
  a24=(a47*a53);
  a57=(a9+a9);
  a24=(a24/a57);
  a59=(a46*a24);
  a45=(a45-a59);
  a53=(a53/a9);
  a59=(a23*a53);
  a45=(a45-a59);
  a59=(a1*a45);
  a60=(a3*a53);
  a61=(a7*a60);
  a59=(a59-a61);
  a61=(a20*a53);
  a62=(a1*a61);
  a59=(a59+a62);
  a10=(a10+a10);
  a25=(a10*a25);
  a8=(a8+a8);
  a24=(a8*a24);
  a25=(a25-a24);
  a53=(a21*a53);
  a25=(a25+a53);
  a53=(a7*a25);
  a59=(a59+a53);
  if (res[0]!=0) res[0][0]=a59;
  a59=-3.9024390243902396e-01;
  a53=(a59*a22);
  a53=(a18*a53);
  a24=(a17*a53);
  a24=(a24*a27);
  a24=(a16*a24);
  a24=(a15*a24);
  a14=(a14?a24:0);
  a24=(a39*a53);
  a32=(a32*a24);
  a32=(a12*a32);
  a32=(a30*a32);
  a29=(a29*a32);
  a29=(a15*a29);
  a14=(a14-a29);
  a14=(a5*a14);
  a14=(a14/a11);
  a4=(a4*a14);
  a49=(a59*a49);
  a11=(a50*a53);
  a11=(a54*a11);
  a48=(a48*a53);
  a42=(a48*a42);
  a42=(a56*a42);
  a42=(a50*a42);
  a42=(a40?a42:0);
  a11=(a11+a42);
  a44=(a48/a44);
  a44=(a56*a44);
  a44=(a50*a44);
  a44=(a41*a44);
  a44=(-a44);
  a40=(a40?a44:0);
  a11=(a11+a40);
  a37=(a48*a37);
  a37=(a50*a37);
  a37=(a37*a58);
  a37=(-a37);
  a37=(a36?a37:0);
  a11=(a11+a37);
  a48=(a48/a38);
  a48=(a50*a48);
  a48=(a48*a34);
  a48=(-a48);
  a36=(a36?a48:0);
  a11=(a11+a36);
  a36=(a33*a11);
  a49=(a49+a36);
  a47=(a47*a49);
  a47=(a47/a57);
  a46=(a46*a47);
  a4=(a4-a46);
  a49=(a49/a9);
  a23=(a23*a49);
  a4=(a4-a23);
  a23=(a1*a4);
  a3=(a3*a49);
  a9=(a7*a3);
  a23=(a23-a9);
  a20=(a20*a49);
  a1=(a1*a20);
  a23=(a23+a1);
  a10=(a10*a14);
  a8=(a8*a47);
  a10=(a10-a8);
  a21=(a21*a49);
  a10=(a10+a21);
  a7=(a7*a10);
  a23=(a23+a7);
  if (res[0]!=0) res[0][1]=a23;
  a23=cos(a0);
  a7=arg[0]? arg[0][1] : 0;
  a21=(a23*a7);
  a49=(a21+a21);
  a8=sin(a0);
  a47=(a8*a7);
  a14=5.0000000000000000e-01;
  a1=(a47+a14);
  a9=casadi_sq(a1);
  a46=casadi_sq(a21);
  a9=(a9+a46);
  a9=sqrt(a9);
  a46=(a9-a6);
  a46=(a46/a13);
  a57=(a6<a46);
  a36=6.1683429662329331e+02;
  a48=(a47+a14);
  a34=(a23*a7);
  a38=(a48*a34);
  a37=(a8*a7);
  a58=(a21*a37);
  a38=(a38-a58);
  a47=(a47+a14);
  a14=casadi_sq(a47);
  a58=casadi_sq(a21);
  a14=(a14+a58);
  a14=sqrt(a14);
  a38=(a38/a14);
  a58=(a19*a38);
  a58=(a36*a58);
  a40=(a17*a58);
  a44=(a46/a26);
  a44=(a44-a28);
  a44=(a16*a44);
  a44=exp(a44);
  a40=(a40*a44);
  a40=(a16*a40);
  a40=(a15*a40);
  a40=(a57?a40:0);
  a46=(a46/a26);
  a46=(a46-a28);
  a42=(a46+a46);
  a46=casadi_sq(a46);
  a46=(a46/a31);
  a46=(-a46);
  a46=exp(a46);
  a53=arg[2]? arg[2][1] : 0;
  a29=(a38*a33);
  a32=(a29<=a35);
  a24=fabs(a29);
  a24=(a24/a16);
  a24=(a28-a24);
  a27=fabs(a29);
  a27=(a27/a16);
  a27=(a28+a27);
  a24=(a24/a27);
  a62=(a32?a24:0);
  a63=(!a32);
  a64=(a41*a29);
  a64=(a64/a16);
  a64=(a64/a43);
  a64=(a28-a64);
  a65=(a29/a16);
  a65=(a65/a43);
  a65=(a28-a65);
  a64=(a64/a65);
  a66=(a63?a64:0);
  a62=(a62+a66);
  a66=(a62*a58);
  a66=(a53*a66);
  a66=(a46*a66);
  a66=(a30*a66);
  a66=(a42*a66);
  a66=(a15*a66);
  a40=(a40-a66);
  a40=(a5*a40);
  a9=(a9+a9);
  a40=(a40/a9);
  a66=(a49*a40);
  a67=(a21+a21);
  a68=(a38/a14);
  a69=(a53*a46);
  a70=(a69*a62);
  a71=(a44-a28);
  a71=(a71/a51);
  a71=(a57?a71:0);
  a70=(a70+a71);
  a71=(a29/a52);
  a71=(a50*a71);
  a70=(a70+a71);
  a70=(a36*a70);
  a71=(a19*a70);
  a72=(a50*a58);
  a72=(a54*a72);
  a58=(a69*a58);
  a64=(a64/a65);
  a73=(a58*a64);
  a73=(a56*a73);
  a73=(a50*a73);
  a73=(a63?a73:0);
  a72=(a72+a73);
  a73=(a58/a65);
  a73=(a56*a73);
  a73=(a50*a73);
  a73=(a41*a73);
  a73=(-a73);
  a73=(a63?a73:0);
  a72=(a72+a73);
  a24=(a24/a27);
  a73=(a58*a24);
  a73=(a50*a73);
  a74=casadi_sign(a29);
  a73=(a73*a74);
  a73=(-a73);
  a73=(a32?a73:0);
  a72=(a72+a73);
  a58=(a58/a27);
  a58=(a50*a58);
  a29=casadi_sign(a29);
  a58=(a58*a29);
  a58=(-a58);
  a58=(a32?a58:0);
  a72=(a72+a58);
  a58=(a33*a72);
  a71=(a71+a58);
  a58=(a68*a71);
  a73=(a14+a14);
  a58=(a58/a73);
  a75=(a67*a58);
  a66=(a66-a75);
  a71=(a71/a14);
  a75=(a37*a71);
  a66=(a66-a75);
  a75=(a23*a66);
  a76=(a21*a71);
  a77=(a8*a76);
  a75=(a75-a77);
  a77=(a48*a71);
  a78=(a23*a77);
  a75=(a75+a78);
  a1=(a1+a1);
  a40=(a1*a40);
  a47=(a47+a47);
  a58=(a47*a58);
  a40=(a40-a58);
  a71=(a34*a71);
  a40=(a40+a71);
  a71=(a8*a40);
  a75=(a75+a71);
  if (res[0]!=0) res[0][2]=a75;
  a75=(a59*a38);
  a75=(a36*a75);
  a71=(a17*a75);
  a71=(a71*a44);
  a71=(a16*a71);
  a71=(a15*a71);
  a57=(a57?a71:0);
  a71=(a62*a75);
  a53=(a53*a71);
  a53=(a46*a53);
  a53=(a30*a53);
  a42=(a42*a53);
  a42=(a15*a42);
  a57=(a57-a42);
  a57=(a5*a57);
  a57=(a57/a9);
  a49=(a49*a57);
  a70=(a59*a70);
  a9=(a50*a75);
  a9=(a54*a9);
  a69=(a69*a75);
  a64=(a69*a64);
  a64=(a56*a64);
  a64=(a50*a64);
  a64=(a63?a64:0);
  a9=(a9+a64);
  a65=(a69/a65);
  a65=(a56*a65);
  a65=(a50*a65);
  a65=(a41*a65);
  a65=(-a65);
  a63=(a63?a65:0);
  a9=(a9+a63);
  a24=(a69*a24);
  a24=(a50*a24);
  a24=(a24*a74);
  a24=(-a24);
  a24=(a32?a24:0);
  a9=(a9+a24);
  a69=(a69/a27);
  a69=(a50*a69);
  a69=(a69*a29);
  a69=(-a69);
  a32=(a32?a69:0);
  a9=(a9+a32);
  a32=(a33*a9);
  a70=(a70+a32);
  a68=(a68*a70);
  a68=(a68/a73);
  a67=(a67*a68);
  a49=(a49-a67);
  a70=(a70/a14);
  a37=(a37*a70);
  a49=(a49-a37);
  a37=(a23*a49);
  a21=(a21*a70);
  a14=(a8*a21);
  a37=(a37-a14);
  a48=(a48*a70);
  a23=(a23*a48);
  a37=(a37+a23);
  a1=(a1*a57);
  a47=(a47*a68);
  a1=(a1-a47);
  a34=(a34*a70);
  a1=(a1+a34);
  a8=(a8*a1);
  a37=(a37+a8);
  if (res[0]!=0) res[0][3]=a37;
  a37=arg[0]? arg[0][5] : 0;
  a8=sin(a37);
  a34=sin(a0);
  a70=(a8*a34);
  a47=cos(a37);
  a68=cos(a0);
  a57=(a47*a68);
  a70=(a70-a57);
  a57=1.2500000000000000e+00;
  a23=(a57*a68);
  a14=(a47*a34);
  a67=(a8*a68);
  a14=(a14+a67);
  a67=arg[0]? arg[0][2] : 0;
  a73=(a14*a67);
  a73=(a23-a73);
  a32=7.5000000000000000e-01;
  a69=(a32*a68);
  a29=(a73-a69);
  a27=-3.9024390243902440e-01;
  a24=9.6426035565568645e+02;
  a74=arg[2]? arg[2][2] : 0;
  a63=(a70*a67);
  a65=(a57*a34);
  a63=(a63-a65);
  a64=(a32*a34);
  a75=(a63+a64);
  a42=casadi_sq(a75);
  a53=(a73-a69);
  a71=casadi_sq(a53);
  a42=(a42+a71);
  a42=sqrt(a42);
  a71=(a42-a6);
  a71=(a71/a13);
  a44=(a71/a26);
  a44=(a44-a28);
  a58=casadi_sq(a44);
  a58=(a58/a31);
  a58=(-a58);
  a58=exp(a58);
  a78=(a74*a58);
  a79=(a63+a64);
  a80=(a32*a68);
  a81=(a14*a67);
  a81=(a23-a81);
  a80=(a80-a81);
  a82=(a79*a80);
  a83=(a70*a67);
  a83=(a83-a65);
  a84=(a32*a34);
  a84=(a83+a84);
  a85=(a29*a84);
  a82=(a82+a85);
  a63=(a63+a64);
  a64=casadi_sq(a63);
  a73=(a73-a69);
  a69=casadi_sq(a73);
  a64=(a64+a69);
  a64=sqrt(a64);
  a82=(a82/a64);
  a69=(a82*a33);
  a85=(a8*a68);
  a86=(a47*a34);
  a85=(a85+a86);
  a86=(a70*a65);
  a87=(a14*a23);
  a86=(a86+a87);
  a87=(a85*a86);
  a88=(a85*a65);
  a89=(a47*a68);
  a90=(a8*a34);
  a89=(a89-a90);
  a90=(a89*a23);
  a88=(a88+a90);
  a90=(a70*a88);
  a87=(a87-a90);
  a87=(a87-a81);
  a81=(a79*a87);
  a90=(a14*a88);
  a91=(a89*a86);
  a90=(a90-a91);
  a90=(a90+a83);
  a83=(a29*a90);
  a81=(a81+a83);
  a81=(a81/a64);
  a83=arg[0]? arg[0][7] : 0;
  a91=(a81*a83);
  a69=(a69+a91);
  a91=(a69<=a35);
  a92=fabs(a69);
  a92=(a92/a16);
  a92=(a28-a92);
  a93=fabs(a69);
  a93=(a93/a16);
  a93=(a28+a93);
  a92=(a92/a93);
  a94=(a91?a92:0);
  a95=(!a91);
  a96=(a41*a69);
  a96=(a96/a16);
  a96=(a96/a43);
  a96=(a28-a96);
  a97=(a69/a16);
  a97=(a97/a43);
  a97=(a28-a97);
  a96=(a96/a97);
  a98=(a95?a96:0);
  a94=(a94+a98);
  a98=(a78*a94);
  a99=(a6<a71);
  a71=(a71/a26);
  a71=(a71-a28);
  a71=(a16*a71);
  a71=exp(a71);
  a100=(a71-a28);
  a100=(a100/a51);
  a100=(a99?a100:0);
  a98=(a98+a100);
  a100=(a69/a52);
  a100=(a50*a100);
  a98=(a98+a100);
  a98=(a24*a98);
  a100=(a27*a98);
  a101=(a27*a81);
  a102=(a19*a82);
  a101=(a101+a102);
  a101=(a24*a101);
  a102=(a50*a101);
  a102=(a54*a102);
  a103=(a78*a101);
  a96=(a96/a97);
  a104=(a103*a96);
  a104=(a56*a104);
  a104=(a50*a104);
  a104=(a95?a104:0);
  a102=(a102+a104);
  a104=(a103/a97);
  a104=(a56*a104);
  a104=(a50*a104);
  a104=(a41*a104);
  a104=(-a104);
  a104=(a95?a104:0);
  a102=(a102+a104);
  a92=(a92/a93);
  a104=(a103*a92);
  a104=(a50*a104);
  a105=casadi_sign(a69);
  a104=(a104*a105);
  a104=(-a104);
  a104=(a91?a104:0);
  a102=(a102+a104);
  a103=(a103/a93);
  a103=(a50*a103);
  a69=casadi_sign(a69);
  a103=(a103*a69);
  a103=(-a103);
  a103=(a91?a103:0);
  a102=(a102+a103);
  a103=(a83*a102);
  a100=(a100+a103);
  a103=(a100/a64);
  a104=(a29*a103);
  a106=(a19*a98);
  a107=(a33*a102);
  a106=(a106+a107);
  a107=(a106/a64);
  a108=(a29*a107);
  a109=(a104+a108);
  a110=(a70*a109);
  a53=(a53+a53);
  a111=(a17*a101);
  a111=(a111*a71);
  a111=(a16*a111);
  a111=(a15*a111);
  a111=(a99?a111:0);
  a44=(a44+a44);
  a101=(a94*a101);
  a101=(a74*a101);
  a101=(a58*a101);
  a101=(a30*a101);
  a101=(a44*a101);
  a101=(a15*a101);
  a111=(a111-a101);
  a111=(a5*a111);
  a42=(a42+a42);
  a111=(a111/a42);
  a101=(a53*a111);
  a73=(a73+a73);
  a112=(a81/a64);
  a100=(a112*a100);
  a113=(a82/a64);
  a106=(a113*a106);
  a100=(a100+a106);
  a106=(a64+a64);
  a100=(a100/a106);
  a114=(a73*a100);
  a115=(a101-a114);
  a116=(a90*a103);
  a117=(a84*a107);
  a116=(a116+a117);
  a115=(a115+a116);
  a117=(a14*a115);
  a110=(a110-a117);
  a117=(a79*a103);
  a118=(a79*a107);
  a119=(a117+a118);
  a120=(a14*a119);
  a110=(a110+a120);
  a75=(a75+a75);
  a111=(a75*a111);
  a63=(a63+a63);
  a100=(a63*a100);
  a120=(a111-a100);
  a103=(a87*a103);
  a107=(a80*a107);
  a103=(a103+a107);
  a120=(a120+a103);
  a107=(a70*a120);
  a110=(a110+a107);
  if (res[0]!=0) res[0][4]=a110;
  a110=1.3902439024390245e+00;
  a107=(a110*a98);
  a121=(a110*a81);
  a122=(a59*a82);
  a121=(a121+a122);
  a121=(a24*a121);
  a122=(a50*a121);
  a122=(a54*a122);
  a78=(a78*a121);
  a96=(a78*a96);
  a96=(a56*a96);
  a96=(a50*a96);
  a96=(a95?a96:0);
  a122=(a122+a96);
  a97=(a78/a97);
  a97=(a56*a97);
  a97=(a50*a97);
  a97=(a41*a97);
  a97=(-a97);
  a95=(a95?a97:0);
  a122=(a122+a95);
  a92=(a78*a92);
  a92=(a50*a92);
  a92=(a92*a105);
  a92=(-a92);
  a92=(a91?a92:0);
  a122=(a122+a92);
  a78=(a78/a93);
  a78=(a50*a78);
  a78=(a78*a69);
  a78=(-a78);
  a91=(a91?a78:0);
  a122=(a122+a91);
  a91=(a83*a122);
  a107=(a107+a91);
  a91=(a107/a64);
  a78=(a29*a91);
  a98=(a59*a98);
  a69=(a33*a122);
  a98=(a98+a69);
  a64=(a98/a64);
  a29=(a29*a64);
  a69=(a78+a29);
  a93=(a70*a69);
  a92=(a17*a121);
  a92=(a92*a71);
  a92=(a16*a92);
  a92=(a15*a92);
  a99=(a99?a92:0);
  a121=(a94*a121);
  a74=(a74*a121);
  a74=(a58*a74);
  a74=(a30*a74);
  a44=(a44*a74);
  a44=(a15*a44);
  a99=(a99-a44);
  a99=(a5*a99);
  a99=(a99/a42);
  a53=(a53*a99);
  a112=(a112*a107);
  a113=(a113*a98);
  a112=(a112+a113);
  a112=(a112/a106);
  a73=(a73*a112);
  a106=(a53-a73);
  a90=(a90*a91);
  a84=(a84*a64);
  a90=(a90+a84);
  a106=(a106+a90);
  a84=(a14*a106);
  a93=(a93-a84);
  a84=(a79*a91);
  a79=(a79*a64);
  a113=(a84+a79);
  a98=(a14*a113);
  a93=(a93+a98);
  a75=(a75*a99);
  a63=(a63*a112);
  a112=(a75-a63);
  a87=(a87*a91);
  a80=(a80*a64);
  a87=(a87+a80);
  a112=(a112+a87);
  a80=(a70*a112);
  a93=(a93+a80);
  if (res[0]!=0) res[0][5]=a93;
  a93=sin(a37);
  a80=sin(a0);
  a64=(a93*a80);
  a91=cos(a37);
  a99=cos(a0);
  a98=(a91*a99);
  a64=(a64-a98);
  a98=(a57*a99);
  a107=(a91*a80);
  a42=(a93*a99);
  a107=(a107+a42);
  a42=arg[0]? arg[0][3] : 0;
  a44=(a107*a42);
  a44=(a98-a44);
  a74=1.7500000000000000e+00;
  a121=(a74*a99);
  a92=(a44-a121);
  a71=8.7594279440248943e+02;
  a105=arg[2]? arg[2][3] : 0;
  a95=(a64*a42);
  a97=(a57*a80);
  a95=(a95-a97);
  a96=(a74*a80);
  a123=(a95+a96);
  a124=casadi_sq(a123);
  a125=(a44-a121);
  a126=casadi_sq(a125);
  a124=(a124+a126);
  a124=sqrt(a124);
  a126=(a124-a6);
  a126=(a126/a13);
  a13=(a126/a26);
  a13=(a13-a28);
  a127=casadi_sq(a13);
  a127=(a127/a31);
  a127=(-a127);
  a127=exp(a127);
  a31=(a105*a127);
  a128=(a95+a96);
  a129=(a74*a99);
  a130=(a107*a42);
  a130=(a98-a130);
  a129=(a129-a130);
  a131=(a128*a129);
  a132=(a64*a42);
  a132=(a132-a97);
  a133=(a74*a80);
  a133=(a132+a133);
  a134=(a92*a133);
  a131=(a131+a134);
  a95=(a95+a96);
  a96=casadi_sq(a95);
  a44=(a44-a121);
  a121=casadi_sq(a44);
  a96=(a96+a121);
  a96=sqrt(a96);
  a131=(a131/a96);
  a121=(a131*a33);
  a134=(a93*a99);
  a135=(a91*a80);
  a134=(a134+a135);
  a135=(a64*a97);
  a136=(a107*a98);
  a135=(a135+a136);
  a136=(a134*a135);
  a137=(a134*a97);
  a138=(a91*a99);
  a139=(a93*a80);
  a138=(a138-a139);
  a139=(a138*a98);
  a137=(a137+a139);
  a139=(a64*a137);
  a136=(a136-a139);
  a136=(a136-a130);
  a130=(a128*a136);
  a139=(a107*a137);
  a140=(a138*a135);
  a139=(a139-a140);
  a139=(a139+a132);
  a132=(a92*a139);
  a130=(a130+a132);
  a130=(a130/a96);
  a132=(a130*a83);
  a121=(a121+a132);
  a35=(a121<=a35);
  a132=fabs(a121);
  a132=(a132/a16);
  a132=(a28-a132);
  a140=fabs(a121);
  a140=(a140/a16);
  a140=(a28+a140);
  a132=(a132/a140);
  a141=(a35?a132:0);
  a142=(!a35);
  a143=(a41*a121);
  a143=(a143/a16);
  a143=(a143/a43);
  a143=(a28-a143);
  a144=(a121/a16);
  a144=(a144/a43);
  a144=(a28-a144);
  a143=(a143/a144);
  a43=(a142?a143:0);
  a141=(a141+a43);
  a43=(a31*a141);
  a6=(a6<a126);
  a126=(a126/a26);
  a126=(a126-a28);
  a126=(a16*a126);
  a126=exp(a126);
  a26=(a126-a28);
  a26=(a26/a51);
  a26=(a6?a26:0);
  a43=(a43+a26);
  a52=(a121/a52);
  a52=(a50*a52);
  a43=(a43+a52);
  a43=(a71*a43);
  a52=(a27*a43);
  a27=(a27*a130);
  a26=(a19*a131);
  a27=(a27+a26);
  a27=(a71*a27);
  a26=(a50*a27);
  a26=(a54*a26);
  a51=(a31*a27);
  a143=(a143/a144);
  a145=(a51*a143);
  a145=(a56*a145);
  a145=(a50*a145);
  a145=(a142?a145:0);
  a26=(a26+a145);
  a145=(a51/a144);
  a145=(a56*a145);
  a145=(a50*a145);
  a145=(a41*a145);
  a145=(-a145);
  a145=(a142?a145:0);
  a26=(a26+a145);
  a132=(a132/a140);
  a145=(a51*a132);
  a145=(a50*a145);
  a146=casadi_sign(a121);
  a145=(a145*a146);
  a145=(-a145);
  a145=(a35?a145:0);
  a26=(a26+a145);
  a51=(a51/a140);
  a51=(a50*a51);
  a121=casadi_sign(a121);
  a51=(a51*a121);
  a51=(-a51);
  a51=(a35?a51:0);
  a26=(a26+a51);
  a51=(a83*a26);
  a52=(a52+a51);
  a51=(a52/a96);
  a145=(a92*a51);
  a19=(a19*a43);
  a147=(a33*a26);
  a19=(a19+a147);
  a147=(a19/a96);
  a148=(a92*a147);
  a149=(a145+a148);
  a150=(a64*a149);
  a125=(a125+a125);
  a151=(a17*a27);
  a151=(a151*a126);
  a151=(a16*a151);
  a151=(a15*a151);
  a151=(a6?a151:0);
  a13=(a13+a13);
  a27=(a141*a27);
  a27=(a105*a27);
  a27=(a127*a27);
  a27=(a30*a27);
  a27=(a13*a27);
  a27=(a15*a27);
  a151=(a151-a27);
  a151=(a5*a151);
  a124=(a124+a124);
  a151=(a151/a124);
  a27=(a125*a151);
  a44=(a44+a44);
  a152=(a130/a96);
  a52=(a152*a52);
  a153=(a131/a96);
  a19=(a153*a19);
  a52=(a52+a19);
  a19=(a96+a96);
  a52=(a52/a19);
  a154=(a44*a52);
  a155=(a27-a154);
  a156=(a139*a51);
  a157=(a133*a147);
  a156=(a156+a157);
  a155=(a155+a156);
  a157=(a107*a155);
  a150=(a150-a157);
  a157=(a128*a51);
  a158=(a128*a147);
  a159=(a157+a158);
  a160=(a107*a159);
  a150=(a150+a160);
  a123=(a123+a123);
  a151=(a123*a151);
  a95=(a95+a95);
  a52=(a95*a52);
  a160=(a151-a52);
  a51=(a136*a51);
  a147=(a129*a147);
  a51=(a51+a147);
  a160=(a160+a51);
  a147=(a64*a160);
  a150=(a150+a147);
  if (res[0]!=0) res[0][6]=a150;
  a150=(a110*a43);
  a110=(a110*a130);
  a147=(a59*a131);
  a110=(a110+a147);
  a110=(a71*a110);
  a147=(a50*a110);
  a54=(a54*a147);
  a31=(a31*a110);
  a143=(a31*a143);
  a143=(a56*a143);
  a143=(a50*a143);
  a143=(a142?a143:0);
  a54=(a54+a143);
  a144=(a31/a144);
  a56=(a56*a144);
  a56=(a50*a56);
  a41=(a41*a56);
  a41=(-a41);
  a142=(a142?a41:0);
  a54=(a54+a142);
  a132=(a31*a132);
  a132=(a50*a132);
  a132=(a132*a146);
  a132=(-a132);
  a132=(a35?a132:0);
  a54=(a54+a132);
  a31=(a31/a140);
  a50=(a50*a31);
  a50=(a50*a121);
  a50=(-a50);
  a35=(a35?a50:0);
  a54=(a54+a35);
  a35=(a83*a54);
  a150=(a150+a35);
  a35=(a150/a96);
  a50=(a92*a35);
  a59=(a59*a43);
  a43=(a33*a54);
  a59=(a59+a43);
  a96=(a59/a96);
  a92=(a92*a96);
  a43=(a50+a92);
  a121=(a64*a43);
  a17=(a17*a110);
  a17=(a17*a126);
  a16=(a16*a17);
  a16=(a15*a16);
  a6=(a6?a16:0);
  a110=(a141*a110);
  a105=(a105*a110);
  a105=(a127*a105);
  a30=(a30*a105);
  a13=(a13*a30);
  a15=(a15*a13);
  a6=(a6-a15);
  a5=(a5*a6);
  a5=(a5/a124);
  a125=(a125*a5);
  a152=(a152*a150);
  a153=(a153*a59);
  a152=(a152+a153);
  a152=(a152/a19);
  a44=(a44*a152);
  a19=(a125-a44);
  a139=(a139*a35);
  a133=(a133*a96);
  a139=(a139+a133);
  a19=(a19+a139);
  a133=(a107*a19);
  a121=(a121-a133);
  a133=(a128*a35);
  a128=(a128*a96);
  a153=(a133+a128);
  a59=(a107*a153);
  a121=(a121+a59);
  a123=(a123*a5);
  a95=(a95*a152);
  a152=(a123-a95);
  a136=(a136*a35);
  a129=(a129*a96);
  a136=(a136+a129);
  a152=(a152+a136);
  a129=(a64*a152);
  a121=(a121+a129);
  if (res[0]!=0) res[0][7]=a121;
  a121=cos(a0);
  a129=(a107*a145);
  a96=(a64*a157);
  a129=(a129-a96);
  a96=(a97*a129);
  a35=(a135*a157);
  a96=(a96+a35);
  a35=(a91*a96);
  a5=(a98*a129);
  a59=(a135*a145);
  a5=(a5-a59);
  a59=(a93*a5);
  a35=(a35-a59);
  a148=(a74*a148);
  a35=(a35+a148);
  a148=(a137*a145);
  a59=(a134*a157);
  a145=(a138*a145);
  a59=(a59-a145);
  a145=(a98*a59);
  a148=(a148+a145);
  a145=(a42*a155);
  a148=(a148-a145);
  a145=(a42*a159);
  a148=(a148+a145);
  a145=(a91*a148);
  a35=(a35+a145);
  a151=(a151-a52);
  a151=(a151+a51);
  a151=(a74*a151);
  a35=(a35+a151);
  a151=(a134*a129);
  a51=(a64*a59);
  a151=(a151+a51);
  a151=(a151-a149);
  a151=(a151-a160);
  a151=(a57*a151);
  a35=(a35+a151);
  a151=(a97*a59);
  a157=(a137*a157);
  a151=(a151-a157);
  a149=(a42*a149);
  a151=(a151+a149);
  a160=(a42*a160);
  a151=(a151+a160);
  a160=(a93*a151);
  a35=(a35+a160);
  a35=(a121*a35);
  a160=cos(a0);
  a149=9.8100000000000005e+00;
  a157=cos(a37);
  a51=4.8780487804878025e-01;
  a52=(a51*a157);
  a145=(a157*a52);
  a150=sin(a37);
  a124=(a51*a150);
  a6=(a150*a124);
  a145=(a145+a6);
  a145=(a149*a145);
  a145=(a160*a145);
  a6=sin(a0);
  a15=(a157*a124);
  a13=(a150*a52);
  a15=(a15-a13);
  a15=(a149*a15);
  a15=(a6*a15);
  a145=(a145+a15);
  a15=sin(a0);
  a13=(a91*a5);
  a30=(a93*a96);
  a13=(a13+a30);
  a154=(a154-a27);
  a154=(a154-a156);
  a154=(a74*a154);
  a13=(a13+a154);
  a154=(a93*a148);
  a13=(a13+a154);
  a129=(a138*a129);
  a59=(a107*a59);
  a129=(a129+a59);
  a129=(a129+a155);
  a129=(a129-a159);
  a129=(a57*a129);
  a13=(a13+a129);
  a158=(a74*a158);
  a13=(a13+a158);
  a158=(a91*a151);
  a13=(a13-a158);
  a13=(a15*a13);
  a145=(a145+a13);
  a35=(a35-a145);
  a145=sin(a0);
  a13=(a14*a104);
  a158=(a70*a117);
  a13=(a13-a158);
  a158=(a23*a13);
  a129=(a86*a104);
  a158=(a158-a129);
  a129=(a47*a158);
  a159=(a65*a13);
  a155=(a86*a117);
  a159=(a159+a155);
  a155=(a8*a159);
  a129=(a129+a155);
  a114=(a114-a101);
  a114=(a114-a116);
  a114=(a32*a114);
  a129=(a129+a114);
  a114=(a88*a104);
  a116=(a85*a117);
  a104=(a89*a104);
  a116=(a116-a104);
  a104=(a23*a116);
  a114=(a114+a104);
  a104=(a67*a115);
  a114=(a114-a104);
  a104=(a67*a119);
  a114=(a114+a104);
  a104=(a8*a114);
  a129=(a129+a104);
  a104=(a89*a13);
  a101=(a14*a116);
  a104=(a104+a101);
  a104=(a104+a115);
  a104=(a104-a119);
  a104=(a57*a104);
  a129=(a129+a104);
  a118=(a32*a118);
  a129=(a129+a118);
  a118=(a65*a116);
  a117=(a88*a117);
  a118=(a118-a117);
  a117=(a67*a109);
  a118=(a118+a117);
  a117=(a67*a120);
  a118=(a118+a117);
  a117=(a47*a118);
  a129=(a129-a117);
  a129=(a145*a129);
  a35=(a35-a129);
  a129=cos(a0);
  a117=(a47*a159);
  a104=(a8*a158);
  a117=(a117-a104);
  a108=(a32*a108);
  a117=(a117+a108);
  a108=(a47*a114);
  a117=(a117+a108);
  a111=(a111-a100);
  a111=(a111+a103);
  a111=(a32*a111);
  a117=(a117+a111);
  a13=(a85*a13);
  a116=(a70*a116);
  a13=(a13+a116);
  a13=(a13-a109);
  a13=(a13-a120);
  a13=(a57*a13);
  a117=(a117+a13);
  a13=(a8*a118);
  a117=(a117+a13);
  a117=(a129*a117);
  a35=(a35+a117);
  a117=sin(a0);
  a66=(a7*a66);
  a77=(a7*a77);
  a66=(a66+a77);
  a66=(a117*a66);
  a35=(a35-a66);
  a66=cos(a0);
  a40=(a7*a40);
  a76=(a7*a76);
  a40=(a40-a76);
  a40=(a66*a40);
  a35=(a35+a40);
  a40=sin(a0);
  a45=(a2*a45);
  a61=(a2*a61);
  a45=(a45+a61);
  a45=(a40*a45);
  a35=(a35-a45);
  a45=cos(a0);
  a25=(a2*a25);
  a60=(a2*a60);
  a25=(a25-a60);
  a25=(a45*a25);
  a35=(a35+a25);
  if (res[0]!=0) res[0][8]=a35;
  a35=(a107*a50);
  a25=(a64*a133);
  a35=(a35-a25);
  a25=(a97*a35);
  a60=(a135*a133);
  a25=(a25+a60);
  a60=(a91*a25);
  a61=(a98*a35);
  a135=(a135*a50);
  a61=(a61-a135);
  a135=(a93*a61);
  a60=(a60-a135);
  a92=(a74*a92);
  a60=(a60+a92);
  a92=(a137*a50);
  a135=(a134*a133);
  a50=(a138*a50);
  a135=(a135-a50);
  a98=(a98*a135);
  a92=(a92+a98);
  a98=(a42*a19);
  a92=(a92-a98);
  a98=(a42*a153);
  a92=(a92+a98);
  a98=(a91*a92);
  a60=(a60+a98);
  a123=(a123-a95);
  a123=(a123+a136);
  a123=(a74*a123);
  a60=(a60+a123);
  a134=(a134*a35);
  a64=(a64*a135);
  a134=(a134+a64);
  a134=(a134-a43);
  a134=(a134-a152);
  a134=(a57*a134);
  a60=(a60+a134);
  a97=(a97*a135);
  a137=(a137*a133);
  a97=(a97-a137);
  a43=(a42*a43);
  a97=(a97+a43);
  a42=(a42*a152);
  a97=(a97+a42);
  a42=(a93*a97);
  a60=(a60+a42);
  a121=(a121*a60);
  a60=-4.8780487804877992e-01;
  a42=(a60*a157);
  a152=(a157*a42);
  a43=(a60*a150);
  a137=(a150*a43);
  a152=(a152+a137);
  a152=(a149*a152);
  a160=(a160*a152);
  a152=(a157*a43);
  a137=(a150*a42);
  a152=(a152-a137);
  a152=(a149*a152);
  a6=(a6*a152);
  a160=(a160+a6);
  a6=(a91*a61);
  a152=(a93*a25);
  a6=(a6+a152);
  a44=(a44-a125);
  a44=(a44-a139);
  a44=(a74*a44);
  a6=(a6+a44);
  a93=(a93*a92);
  a6=(a6+a93);
  a138=(a138*a35);
  a107=(a107*a135);
  a138=(a138+a107);
  a138=(a138+a19);
  a138=(a138-a153);
  a138=(a57*a138);
  a6=(a6+a138);
  a74=(a74*a128);
  a6=(a6+a74);
  a91=(a91*a97);
  a6=(a6-a91);
  a15=(a15*a6);
  a160=(a160+a15);
  a121=(a121-a160);
  a160=(a14*a78);
  a15=(a70*a84);
  a160=(a160-a15);
  a15=(a23*a160);
  a6=(a86*a78);
  a15=(a15-a6);
  a6=(a47*a15);
  a91=(a65*a160);
  a86=(a86*a84);
  a91=(a91+a86);
  a86=(a8*a91);
  a6=(a6+a86);
  a73=(a73-a53);
  a73=(a73-a90);
  a73=(a32*a73);
  a6=(a6+a73);
  a73=(a88*a78);
  a90=(a85*a84);
  a78=(a89*a78);
  a90=(a90-a78);
  a23=(a23*a90);
  a73=(a73+a23);
  a23=(a67*a106);
  a73=(a73-a23);
  a23=(a67*a113);
  a73=(a73+a23);
  a23=(a8*a73);
  a6=(a6+a23);
  a89=(a89*a160);
  a14=(a14*a90);
  a89=(a89+a14);
  a89=(a89+a106);
  a89=(a89-a113);
  a89=(a57*a89);
  a6=(a6+a89);
  a79=(a32*a79);
  a6=(a6+a79);
  a65=(a65*a90);
  a88=(a88*a84);
  a65=(a65-a88);
  a88=(a67*a69);
  a65=(a65+a88);
  a67=(a67*a112);
  a65=(a65+a67);
  a67=(a47*a65);
  a6=(a6-a67);
  a145=(a145*a6);
  a121=(a121-a145);
  a145=(a47*a91);
  a6=(a8*a15);
  a145=(a145-a6);
  a29=(a32*a29);
  a145=(a145+a29);
  a47=(a47*a73);
  a145=(a145+a47);
  a75=(a75-a63);
  a75=(a75+a87);
  a32=(a32*a75);
  a145=(a145+a32);
  a85=(a85*a160);
  a70=(a70*a90);
  a85=(a85+a70);
  a85=(a85-a69);
  a85=(a85-a112);
  a85=(a57*a85);
  a145=(a145+a85);
  a8=(a8*a65);
  a145=(a145+a8);
  a129=(a129*a145);
  a121=(a121+a129);
  a49=(a7*a49);
  a48=(a7*a48);
  a49=(a49+a48);
  a117=(a117*a49);
  a121=(a121-a117);
  a1=(a7*a1);
  a7=(a7*a21);
  a1=(a1-a7);
  a66=(a66*a1);
  a121=(a121+a66);
  a4=(a2*a4);
  a20=(a2*a20);
  a4=(a4+a20);
  a40=(a40*a4);
  a121=(a121-a40);
  a10=(a2*a10);
  a2=(a2*a3);
  a10=(a10-a2);
  a45=(a45*a10);
  a121=(a121+a45);
  if (res[0]!=0) res[0][9]=a121;
  a121=cos(a37);
  a45=(a57*a33);
  a33=(a33+a83);
  a10=(a33*a52);
  a2=(a83*a52);
  a10=(a10-a2);
  a2=(a45*a10);
  a3=cos(a0);
  a3=(a149*a3);
  a40=(a3*a52);
  a2=(a2-a40);
  a40=(a157*a3);
  a0=sin(a0);
  a149=(a149*a0);
  a0=(a150*a149);
  a40=(a40-a0);
  a0=(a157*a45);
  a4=(a0*a83);
  a40=(a40+a4);
  a4=(a33*a0);
  a40=(a40-a4);
  a4=(a51*a40);
  a2=(a2+a4);
  a4=(a149*a124);
  a2=(a2-a4);
  a2=(a121*a2);
  a4=sin(a37);
  a20=(a150*a45);
  a66=(a33*a20);
  a1=(a157*a149);
  a7=(a150*a3);
  a1=(a1+a7);
  a7=(a20*a83);
  a1=(a1+a7);
  a66=(a66-a1);
  a51=(a51*a66);
  a1=(a149*a52);
  a51=(a51-a1);
  a1=(a83*a124);
  a7=(a33*a124);
  a1=(a1-a7);
  a7=(a45*a1);
  a51=(a51+a7);
  a7=(a3*a124);
  a51=(a51+a7);
  a51=(a4*a51);
  a2=(a2-a51);
  a51=sin(a37);
  a7=(a99*a5);
  a21=(a80*a96);
  a7=(a7+a21);
  a21=(a80*a148);
  a7=(a7+a21);
  a21=(a99*a151);
  a7=(a7-a21);
  a7=(a51*a7);
  a2=(a2-a7);
  a7=cos(a37);
  a96=(a99*a96);
  a5=(a80*a5);
  a96=(a96-a5);
  a148=(a99*a148);
  a96=(a96+a148);
  a151=(a80*a151);
  a96=(a96+a151);
  a96=(a7*a96);
  a2=(a2+a96);
  a96=sin(a37);
  a151=(a68*a158);
  a148=(a34*a159);
  a151=(a151+a148);
  a148=(a34*a114);
  a151=(a151+a148);
  a148=(a68*a118);
  a151=(a151-a148);
  a151=(a96*a151);
  a2=(a2-a151);
  a37=cos(a37);
  a159=(a68*a159);
  a158=(a34*a158);
  a159=(a159-a158);
  a114=(a68*a114);
  a159=(a159+a114);
  a118=(a34*a118);
  a159=(a159+a118);
  a159=(a37*a159);
  a2=(a2+a159);
  if (res[0]!=0) res[0][10]=a2;
  a2=(a33*a42);
  a159=(a83*a42);
  a2=(a2-a159);
  a159=(a45*a2);
  a118=(a3*a42);
  a159=(a159-a118);
  a40=(a60*a40);
  a159=(a159+a40);
  a40=(a149*a43);
  a159=(a159-a40);
  a121=(a121*a159);
  a60=(a60*a66);
  a149=(a149*a42);
  a60=(a60-a149);
  a83=(a83*a43);
  a33=(a33*a43);
  a83=(a83-a33);
  a45=(a45*a83);
  a60=(a60+a45);
  a3=(a3*a43);
  a60=(a60+a3);
  a4=(a4*a60);
  a121=(a121-a4);
  a4=(a99*a61);
  a60=(a80*a25);
  a4=(a4+a60);
  a60=(a80*a92);
  a4=(a4+a60);
  a60=(a99*a97);
  a4=(a4-a60);
  a51=(a51*a4);
  a121=(a121-a51);
  a25=(a99*a25);
  a61=(a80*a61);
  a25=(a25-a61);
  a99=(a99*a92);
  a25=(a25+a99);
  a80=(a80*a97);
  a25=(a25+a80);
  a7=(a7*a25);
  a121=(a121+a7);
  a7=(a68*a15);
  a25=(a34*a91);
  a7=(a7+a25);
  a25=(a34*a73);
  a7=(a7+a25);
  a25=(a68*a65);
  a7=(a7-a25);
  a96=(a96*a7);
  a121=(a121-a96);
  a91=(a68*a91);
  a15=(a34*a15);
  a91=(a91-a15);
  a68=(a68*a73);
  a91=(a91+a68);
  a34=(a34*a65);
  a91=(a91+a34);
  a37=(a37*a91);
  a121=(a121+a37);
  if (res[0]!=0) res[0][11]=a121;
  a121=-1.;
  if (res[0]!=0) res[0][12]=a121;
  a37=(a20*a52);
  a91=(a0*a124);
  a37=(a37-a91);
  a10=(a150*a10);
  a1=(a157*a1);
  a10=(a10+a1);
  a10=(a57*a10);
  a10=(a37+a10);
  a1=(a131*a26);
  a10=(a10+a1);
  a1=(a82*a102);
  a10=(a10+a1);
  a72=(a38*a72);
  a10=(a10+a72);
  a55=(a22*a55);
  a10=(a10+a55);
  if (res[0]!=0) res[0][13]=a10;
  a10=(a20*a42);
  a55=(a0*a43);
  a10=(a10-a55);
  a150=(a150*a2);
  a157=(a157*a83);
  a150=(a150+a157);
  a57=(a57*a150);
  a57=(a10+a57);
  a150=(a131*a54);
  a57=(a57+a150);
  a150=(a82*a122);
  a57=(a57+a150);
  a9=(a38*a9);
  a57=(a57+a9);
  a11=(a22*a11);
  a57=(a57+a11);
  if (res[0]!=0) res[0][14]=a57;
  if (res[0]!=0) res[0][15]=a121;
  a52=(a20*a52);
  a37=(a37-a52);
  a124=(a0*a124);
  a37=(a37+a124);
  a26=(a130*a26);
  a37=(a37+a26);
  a102=(a81*a102);
  a37=(a37+a102);
  if (res[0]!=0) res[0][16]=a37;
  a20=(a20*a42);
  a10=(a10-a20);
  a0=(a0*a43);
  a10=(a10+a0);
  a54=(a130*a54);
  a10=(a10+a54);
  a122=(a81*a122);
  a10=(a10+a122);
  if (res[0]!=0) res[0][17]=a10;
  if (res[1]!=0) res[1][0]=a28;
  if (res[1]!=0) res[1][1]=a28;
  if (res[1]!=0) res[1][2]=a28;
  if (res[1]!=0) res[1][3]=a28;
  if (res[1]!=0) res[1][4]=a28;
  if (res[1]!=0) res[1][5]=a28;
  if (res[1]!=0) res[1][6]=a28;
  if (res[1]!=0) res[1][7]=a28;
  a28=2.7025639012821789e-01;
  a10=1.2330447799599942e+00;
  a122=1.4439765966454325e+00;
  a54=-2.7025639012821762e-01;
  a39=(a39*a12);
  a18=(a18*a39);
  a22=(a22*a18);
  a18=(a54*a22);
  a18=(a122*a18);
  a39=(a10*a18);
  a12=9.6278838983177628e-01;
  a22=(a12*a22);
  a39=(a39-a22);
  a39=(a28*a39);
  a39=(-a39);
  if (res[2]!=0) res[2][0]=a39;
  if (res[2]!=0) res[2][1]=a18;
  a62=(a62*a46);
  a36=(a36*a62);
  a38=(a38*a36);
  a36=(a54*a38);
  a36=(a122*a36);
  a62=(a10*a36);
  a38=(a12*a38);
  a62=(a62-a38);
  a62=(a28*a62);
  a62=(-a62);
  if (res[2]!=0) res[2][2]=a62;
  if (res[2]!=0) res[2][3]=a36;
  a94=(a94*a58);
  a24=(a24*a94);
  a82=(a82*a24);
  a94=(a54*a82);
  a58=9.6278838983177639e-01;
  a81=(a81*a24);
  a24=(a58*a81);
  a94=(a94+a24);
  a94=(a122*a94);
  a24=(a10*a94);
  a82=(a12*a82);
  a81=(a28*a81);
  a82=(a82+a81);
  a24=(a24-a82);
  a24=(a28*a24);
  a24=(-a24);
  if (res[2]!=0) res[2][4]=a24;
  if (res[2]!=0) res[2][5]=a94;
  a141=(a141*a127);
  a71=(a71*a141);
  a131=(a131*a71);
  a54=(a54*a131);
  a130=(a130*a71);
  a58=(a58*a130);
  a54=(a54+a58);
  a122=(a122*a54);
  a10=(a10*a122);
  a12=(a12*a131);
  a130=(a28*a130);
  a12=(a12+a130);
  a10=(a10-a12);
  a28=(a28*a10);
  a28=(-a28);
  if (res[2]!=0) res[2][6]=a28;
  if (res[2]!=0) res[2][7]=a122;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211439_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
