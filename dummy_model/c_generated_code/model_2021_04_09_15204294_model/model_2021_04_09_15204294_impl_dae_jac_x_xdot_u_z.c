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
  #define CASADI_PREFIX(ID) model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_ ## ID
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

/* model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8x8,18nz],o1[8x8,8nz],o2[8x4,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a18=700.;
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
  a36=(a47+a14);
  a48=(a23*a7);
  a34=(a36*a48);
  a38=(a8*a7);
  a37=(a21*a38);
  a34=(a34-a37);
  a47=(a47+a14);
  a14=casadi_sq(a47);
  a37=casadi_sq(a21);
  a14=(a14+a37);
  a14=sqrt(a14);
  a34=(a34/a14);
  a37=(a19*a34);
  a37=(a18*a37);
  a58=(a17*a37);
  a40=(a46/a26);
  a40=(a40-a28);
  a40=(a16*a40);
  a40=exp(a40);
  a58=(a58*a40);
  a58=(a16*a58);
  a58=(a15*a58);
  a58=(a57?a58:0);
  a46=(a46/a26);
  a46=(a46-a28);
  a44=(a46+a46);
  a46=casadi_sq(a46);
  a46=(a46/a31);
  a46=(-a46);
  a46=exp(a46);
  a42=arg[2]? arg[2][1] : 0;
  a53=(a34*a33);
  a29=(a53<=a35);
  a32=fabs(a53);
  a32=(a32/a16);
  a32=(a28-a32);
  a24=fabs(a53);
  a24=(a24/a16);
  a24=(a28+a24);
  a32=(a32/a24);
  a27=(a29?a32:0);
  a62=(!a29);
  a63=(a41*a53);
  a63=(a63/a16);
  a63=(a63/a43);
  a63=(a28-a63);
  a64=(a53/a16);
  a64=(a64/a43);
  a64=(a28-a64);
  a63=(a63/a64);
  a65=(a62?a63:0);
  a27=(a27+a65);
  a65=(a27*a37);
  a65=(a42*a65);
  a65=(a46*a65);
  a65=(a30*a65);
  a65=(a44*a65);
  a65=(a15*a65);
  a58=(a58-a65);
  a58=(a5*a58);
  a9=(a9+a9);
  a58=(a58/a9);
  a65=(a49*a58);
  a66=(a21+a21);
  a67=(a34/a14);
  a68=(a42*a46);
  a69=(a68*a27);
  a70=(a40-a28);
  a70=(a70/a51);
  a70=(a57?a70:0);
  a69=(a69+a70);
  a70=(a53/a52);
  a70=(a50*a70);
  a69=(a69+a70);
  a69=(a18*a69);
  a70=(a19*a69);
  a71=(a50*a37);
  a71=(a54*a71);
  a37=(a68*a37);
  a63=(a63/a64);
  a72=(a37*a63);
  a72=(a56*a72);
  a72=(a50*a72);
  a72=(a62?a72:0);
  a71=(a71+a72);
  a72=(a37/a64);
  a72=(a56*a72);
  a72=(a50*a72);
  a72=(a41*a72);
  a72=(-a72);
  a72=(a62?a72:0);
  a71=(a71+a72);
  a32=(a32/a24);
  a72=(a37*a32);
  a72=(a50*a72);
  a73=casadi_sign(a53);
  a72=(a72*a73);
  a72=(-a72);
  a72=(a29?a72:0);
  a71=(a71+a72);
  a37=(a37/a24);
  a37=(a50*a37);
  a53=casadi_sign(a53);
  a37=(a37*a53);
  a37=(-a37);
  a37=(a29?a37:0);
  a71=(a71+a37);
  a37=(a33*a71);
  a70=(a70+a37);
  a37=(a67*a70);
  a72=(a14+a14);
  a37=(a37/a72);
  a74=(a66*a37);
  a65=(a65-a74);
  a70=(a70/a14);
  a74=(a38*a70);
  a65=(a65-a74);
  a74=(a23*a65);
  a75=(a21*a70);
  a76=(a8*a75);
  a74=(a74-a76);
  a76=(a36*a70);
  a77=(a23*a76);
  a74=(a74+a77);
  a1=(a1+a1);
  a58=(a1*a58);
  a47=(a47+a47);
  a37=(a47*a37);
  a58=(a58-a37);
  a70=(a48*a70);
  a58=(a58+a70);
  a70=(a8*a58);
  a74=(a74+a70);
  if (res[0]!=0) res[0][2]=a74;
  a74=(a59*a34);
  a74=(a18*a74);
  a70=(a17*a74);
  a70=(a70*a40);
  a70=(a16*a70);
  a70=(a15*a70);
  a57=(a57?a70:0);
  a70=(a27*a74);
  a42=(a42*a70);
  a42=(a46*a42);
  a42=(a30*a42);
  a44=(a44*a42);
  a44=(a15*a44);
  a57=(a57-a44);
  a57=(a5*a57);
  a57=(a57/a9);
  a49=(a49*a57);
  a69=(a59*a69);
  a9=(a50*a74);
  a9=(a54*a9);
  a68=(a68*a74);
  a63=(a68*a63);
  a63=(a56*a63);
  a63=(a50*a63);
  a63=(a62?a63:0);
  a9=(a9+a63);
  a64=(a68/a64);
  a64=(a56*a64);
  a64=(a50*a64);
  a64=(a41*a64);
  a64=(-a64);
  a62=(a62?a64:0);
  a9=(a9+a62);
  a32=(a68*a32);
  a32=(a50*a32);
  a32=(a32*a73);
  a32=(-a32);
  a32=(a29?a32:0);
  a9=(a9+a32);
  a68=(a68/a24);
  a68=(a50*a68);
  a68=(a68*a53);
  a68=(-a68);
  a29=(a29?a68:0);
  a9=(a9+a29);
  a29=(a33*a9);
  a69=(a69+a29);
  a67=(a67*a69);
  a67=(a67/a72);
  a66=(a66*a67);
  a49=(a49-a66);
  a69=(a69/a14);
  a38=(a38*a69);
  a49=(a49-a38);
  a38=(a23*a49);
  a21=(a21*a69);
  a14=(a8*a21);
  a38=(a38-a14);
  a36=(a36*a69);
  a23=(a23*a36);
  a38=(a38+a23);
  a1=(a1*a57);
  a47=(a47*a67);
  a1=(a1-a47);
  a48=(a48*a69);
  a1=(a1+a48);
  a8=(a8*a1);
  a38=(a38+a8);
  if (res[0]!=0) res[0][3]=a38;
  a38=arg[0]? arg[0][5] : 0;
  a8=sin(a38);
  a48=sin(a0);
  a69=(a8*a48);
  a47=cos(a38);
  a67=cos(a0);
  a57=(a47*a67);
  a69=(a69-a57);
  a57=1.2500000000000000e+00;
  a23=(a57*a67);
  a14=(a47*a48);
  a66=(a8*a67);
  a14=(a14+a66);
  a66=arg[0]? arg[0][2] : 0;
  a72=(a14*a66);
  a72=(a23-a72);
  a29=7.5000000000000000e-01;
  a68=(a29*a67);
  a53=(a72-a68);
  a24=-3.9024390243902440e-01;
  a32=arg[2]? arg[2][2] : 0;
  a73=(a69*a66);
  a62=(a57*a48);
  a73=(a73-a62);
  a64=(a29*a48);
  a63=(a73+a64);
  a74=casadi_sq(a63);
  a44=(a72-a68);
  a42=casadi_sq(a44);
  a74=(a74+a42);
  a74=sqrt(a74);
  a42=(a74-a6);
  a42=(a42/a13);
  a70=(a42/a26);
  a70=(a70-a28);
  a40=casadi_sq(a70);
  a40=(a40/a31);
  a40=(-a40);
  a40=exp(a40);
  a37=(a32*a40);
  a77=(a73+a64);
  a78=(a29*a67);
  a79=(a14*a66);
  a79=(a23-a79);
  a78=(a78-a79);
  a80=(a77*a78);
  a81=(a69*a66);
  a81=(a81-a62);
  a82=(a29*a48);
  a82=(a81+a82);
  a83=(a53*a82);
  a80=(a80+a83);
  a73=(a73+a64);
  a64=casadi_sq(a73);
  a72=(a72-a68);
  a68=casadi_sq(a72);
  a64=(a64+a68);
  a64=sqrt(a64);
  a80=(a80/a64);
  a68=(a80*a33);
  a83=(a8*a67);
  a84=(a47*a48);
  a83=(a83+a84);
  a84=(a69*a62);
  a85=(a14*a23);
  a84=(a84+a85);
  a85=(a83*a84);
  a86=(a83*a62);
  a87=(a47*a67);
  a88=(a8*a48);
  a87=(a87-a88);
  a88=(a87*a23);
  a86=(a86+a88);
  a88=(a69*a86);
  a85=(a85-a88);
  a85=(a85-a79);
  a79=(a77*a85);
  a88=(a14*a86);
  a89=(a87*a84);
  a88=(a88-a89);
  a88=(a88+a81);
  a81=(a53*a88);
  a79=(a79+a81);
  a79=(a79/a64);
  a81=arg[0]? arg[0][7] : 0;
  a89=(a79*a81);
  a68=(a68+a89);
  a89=(a68<=a35);
  a90=fabs(a68);
  a90=(a90/a16);
  a90=(a28-a90);
  a91=fabs(a68);
  a91=(a91/a16);
  a91=(a28+a91);
  a90=(a90/a91);
  a92=(a89?a90:0);
  a93=(!a89);
  a94=(a41*a68);
  a94=(a94/a16);
  a94=(a94/a43);
  a94=(a28-a94);
  a95=(a68/a16);
  a95=(a95/a43);
  a95=(a28-a95);
  a94=(a94/a95);
  a96=(a93?a94:0);
  a92=(a92+a96);
  a96=(a37*a92);
  a97=(a6<a42);
  a42=(a42/a26);
  a42=(a42-a28);
  a42=(a16*a42);
  a42=exp(a42);
  a98=(a42-a28);
  a98=(a98/a51);
  a98=(a97?a98:0);
  a96=(a96+a98);
  a98=(a68/a52);
  a98=(a50*a98);
  a96=(a96+a98);
  a96=(a18*a96);
  a98=(a24*a96);
  a99=(a24*a79);
  a100=(a19*a80);
  a99=(a99+a100);
  a99=(a18*a99);
  a100=(a50*a99);
  a100=(a54*a100);
  a101=(a37*a99);
  a94=(a94/a95);
  a102=(a101*a94);
  a102=(a56*a102);
  a102=(a50*a102);
  a102=(a93?a102:0);
  a100=(a100+a102);
  a102=(a101/a95);
  a102=(a56*a102);
  a102=(a50*a102);
  a102=(a41*a102);
  a102=(-a102);
  a102=(a93?a102:0);
  a100=(a100+a102);
  a90=(a90/a91);
  a102=(a101*a90);
  a102=(a50*a102);
  a103=casadi_sign(a68);
  a102=(a102*a103);
  a102=(-a102);
  a102=(a89?a102:0);
  a100=(a100+a102);
  a101=(a101/a91);
  a101=(a50*a101);
  a68=casadi_sign(a68);
  a101=(a101*a68);
  a101=(-a101);
  a101=(a89?a101:0);
  a100=(a100+a101);
  a101=(a81*a100);
  a98=(a98+a101);
  a101=(a98/a64);
  a102=(a53*a101);
  a104=(a19*a96);
  a105=(a33*a100);
  a104=(a104+a105);
  a105=(a104/a64);
  a106=(a53*a105);
  a107=(a102+a106);
  a108=(a69*a107);
  a44=(a44+a44);
  a109=(a17*a99);
  a109=(a109*a42);
  a109=(a16*a109);
  a109=(a15*a109);
  a109=(a97?a109:0);
  a70=(a70+a70);
  a99=(a92*a99);
  a99=(a32*a99);
  a99=(a40*a99);
  a99=(a30*a99);
  a99=(a70*a99);
  a99=(a15*a99);
  a109=(a109-a99);
  a109=(a5*a109);
  a74=(a74+a74);
  a109=(a109/a74);
  a99=(a44*a109);
  a72=(a72+a72);
  a110=(a79/a64);
  a98=(a110*a98);
  a111=(a80/a64);
  a104=(a111*a104);
  a98=(a98+a104);
  a104=(a64+a64);
  a98=(a98/a104);
  a112=(a72*a98);
  a113=(a99-a112);
  a114=(a88*a101);
  a115=(a82*a105);
  a114=(a114+a115);
  a113=(a113+a114);
  a115=(a14*a113);
  a108=(a108-a115);
  a115=(a77*a101);
  a116=(a77*a105);
  a117=(a115+a116);
  a118=(a14*a117);
  a108=(a108+a118);
  a63=(a63+a63);
  a109=(a63*a109);
  a73=(a73+a73);
  a98=(a73*a98);
  a118=(a109-a98);
  a101=(a85*a101);
  a105=(a78*a105);
  a101=(a101+a105);
  a118=(a118+a101);
  a105=(a69*a118);
  a108=(a108+a105);
  if (res[0]!=0) res[0][4]=a108;
  a108=1.3902439024390245e+00;
  a105=(a108*a96);
  a119=(a108*a79);
  a120=(a59*a80);
  a119=(a119+a120);
  a119=(a18*a119);
  a120=(a50*a119);
  a120=(a54*a120);
  a37=(a37*a119);
  a94=(a37*a94);
  a94=(a56*a94);
  a94=(a50*a94);
  a94=(a93?a94:0);
  a120=(a120+a94);
  a95=(a37/a95);
  a95=(a56*a95);
  a95=(a50*a95);
  a95=(a41*a95);
  a95=(-a95);
  a93=(a93?a95:0);
  a120=(a120+a93);
  a90=(a37*a90);
  a90=(a50*a90);
  a90=(a90*a103);
  a90=(-a90);
  a90=(a89?a90:0);
  a120=(a120+a90);
  a37=(a37/a91);
  a37=(a50*a37);
  a37=(a37*a68);
  a37=(-a37);
  a89=(a89?a37:0);
  a120=(a120+a89);
  a89=(a81*a120);
  a105=(a105+a89);
  a89=(a105/a64);
  a37=(a53*a89);
  a96=(a59*a96);
  a68=(a33*a120);
  a96=(a96+a68);
  a64=(a96/a64);
  a53=(a53*a64);
  a68=(a37+a53);
  a91=(a69*a68);
  a90=(a17*a119);
  a90=(a90*a42);
  a90=(a16*a90);
  a90=(a15*a90);
  a97=(a97?a90:0);
  a119=(a92*a119);
  a32=(a32*a119);
  a32=(a40*a32);
  a32=(a30*a32);
  a70=(a70*a32);
  a70=(a15*a70);
  a97=(a97-a70);
  a97=(a5*a97);
  a97=(a97/a74);
  a44=(a44*a97);
  a110=(a110*a105);
  a111=(a111*a96);
  a110=(a110+a111);
  a110=(a110/a104);
  a72=(a72*a110);
  a104=(a44-a72);
  a88=(a88*a89);
  a82=(a82*a64);
  a88=(a88+a82);
  a104=(a104+a88);
  a82=(a14*a104);
  a91=(a91-a82);
  a82=(a77*a89);
  a77=(a77*a64);
  a111=(a82+a77);
  a96=(a14*a111);
  a91=(a91+a96);
  a63=(a63*a97);
  a73=(a73*a110);
  a110=(a63-a73);
  a85=(a85*a89);
  a78=(a78*a64);
  a85=(a85+a78);
  a110=(a110+a85);
  a78=(a69*a110);
  a91=(a91+a78);
  if (res[0]!=0) res[0][5]=a91;
  a91=sin(a38);
  a78=sin(a0);
  a64=(a91*a78);
  a89=cos(a38);
  a97=cos(a0);
  a96=(a89*a97);
  a64=(a64-a96);
  a96=(a57*a97);
  a105=(a89*a78);
  a74=(a91*a97);
  a105=(a105+a74);
  a74=arg[0]? arg[0][3] : 0;
  a70=(a105*a74);
  a70=(a96-a70);
  a32=1.7500000000000000e+00;
  a119=(a32*a97);
  a90=(a70-a119);
  a42=arg[2]? arg[2][3] : 0;
  a103=(a64*a74);
  a93=(a57*a78);
  a103=(a103-a93);
  a95=(a32*a78);
  a94=(a103+a95);
  a121=casadi_sq(a94);
  a122=(a70-a119);
  a123=casadi_sq(a122);
  a121=(a121+a123);
  a121=sqrt(a121);
  a123=(a121-a6);
  a123=(a123/a13);
  a13=(a123/a26);
  a13=(a13-a28);
  a124=casadi_sq(a13);
  a124=(a124/a31);
  a124=(-a124);
  a124=exp(a124);
  a31=(a42*a124);
  a125=(a103+a95);
  a126=(a32*a97);
  a127=(a105*a74);
  a127=(a96-a127);
  a126=(a126-a127);
  a128=(a125*a126);
  a129=(a64*a74);
  a129=(a129-a93);
  a130=(a32*a78);
  a130=(a129+a130);
  a131=(a90*a130);
  a128=(a128+a131);
  a103=(a103+a95);
  a95=casadi_sq(a103);
  a70=(a70-a119);
  a119=casadi_sq(a70);
  a95=(a95+a119);
  a95=sqrt(a95);
  a128=(a128/a95);
  a119=(a128*a33);
  a131=(a91*a97);
  a132=(a89*a78);
  a131=(a131+a132);
  a132=(a64*a93);
  a133=(a105*a96);
  a132=(a132+a133);
  a133=(a131*a132);
  a134=(a131*a93);
  a135=(a89*a97);
  a136=(a91*a78);
  a135=(a135-a136);
  a136=(a135*a96);
  a134=(a134+a136);
  a136=(a64*a134);
  a133=(a133-a136);
  a133=(a133-a127);
  a127=(a125*a133);
  a136=(a105*a134);
  a137=(a135*a132);
  a136=(a136-a137);
  a136=(a136+a129);
  a129=(a90*a136);
  a127=(a127+a129);
  a127=(a127/a95);
  a129=(a127*a81);
  a119=(a119+a129);
  a35=(a119<=a35);
  a129=fabs(a119);
  a129=(a129/a16);
  a129=(a28-a129);
  a137=fabs(a119);
  a137=(a137/a16);
  a137=(a28+a137);
  a129=(a129/a137);
  a138=(a35?a129:0);
  a139=(!a35);
  a140=(a41*a119);
  a140=(a140/a16);
  a140=(a140/a43);
  a140=(a28-a140);
  a141=(a119/a16);
  a141=(a141/a43);
  a141=(a28-a141);
  a140=(a140/a141);
  a43=(a139?a140:0);
  a138=(a138+a43);
  a43=(a31*a138);
  a6=(a6<a123);
  a123=(a123/a26);
  a123=(a123-a28);
  a123=(a16*a123);
  a123=exp(a123);
  a26=(a123-a28);
  a26=(a26/a51);
  a26=(a6?a26:0);
  a43=(a43+a26);
  a52=(a119/a52);
  a52=(a50*a52);
  a43=(a43+a52);
  a43=(a18*a43);
  a52=(a24*a43);
  a24=(a24*a127);
  a26=(a19*a128);
  a24=(a24+a26);
  a24=(a18*a24);
  a26=(a50*a24);
  a26=(a54*a26);
  a51=(a31*a24);
  a140=(a140/a141);
  a142=(a51*a140);
  a142=(a56*a142);
  a142=(a50*a142);
  a142=(a139?a142:0);
  a26=(a26+a142);
  a142=(a51/a141);
  a142=(a56*a142);
  a142=(a50*a142);
  a142=(a41*a142);
  a142=(-a142);
  a142=(a139?a142:0);
  a26=(a26+a142);
  a129=(a129/a137);
  a142=(a51*a129);
  a142=(a50*a142);
  a143=casadi_sign(a119);
  a142=(a142*a143);
  a142=(-a142);
  a142=(a35?a142:0);
  a26=(a26+a142);
  a51=(a51/a137);
  a51=(a50*a51);
  a119=casadi_sign(a119);
  a51=(a51*a119);
  a51=(-a51);
  a51=(a35?a51:0);
  a26=(a26+a51);
  a51=(a81*a26);
  a52=(a52+a51);
  a51=(a52/a95);
  a142=(a90*a51);
  a19=(a19*a43);
  a144=(a33*a26);
  a19=(a19+a144);
  a144=(a19/a95);
  a145=(a90*a144);
  a146=(a142+a145);
  a147=(a64*a146);
  a122=(a122+a122);
  a148=(a17*a24);
  a148=(a148*a123);
  a148=(a16*a148);
  a148=(a15*a148);
  a148=(a6?a148:0);
  a13=(a13+a13);
  a24=(a138*a24);
  a24=(a42*a24);
  a24=(a124*a24);
  a24=(a30*a24);
  a24=(a13*a24);
  a24=(a15*a24);
  a148=(a148-a24);
  a148=(a5*a148);
  a121=(a121+a121);
  a148=(a148/a121);
  a24=(a122*a148);
  a70=(a70+a70);
  a149=(a127/a95);
  a52=(a149*a52);
  a150=(a128/a95);
  a19=(a150*a19);
  a52=(a52+a19);
  a19=(a95+a95);
  a52=(a52/a19);
  a151=(a70*a52);
  a152=(a24-a151);
  a153=(a136*a51);
  a154=(a130*a144);
  a153=(a153+a154);
  a152=(a152+a153);
  a154=(a105*a152);
  a147=(a147-a154);
  a154=(a125*a51);
  a155=(a125*a144);
  a156=(a154+a155);
  a157=(a105*a156);
  a147=(a147+a157);
  a94=(a94+a94);
  a148=(a94*a148);
  a103=(a103+a103);
  a52=(a103*a52);
  a157=(a148-a52);
  a51=(a133*a51);
  a144=(a126*a144);
  a51=(a51+a144);
  a157=(a157+a51);
  a144=(a64*a157);
  a147=(a147+a144);
  if (res[0]!=0) res[0][6]=a147;
  a147=(a108*a43);
  a108=(a108*a127);
  a144=(a59*a128);
  a108=(a108+a144);
  a108=(a18*a108);
  a144=(a50*a108);
  a54=(a54*a144);
  a31=(a31*a108);
  a140=(a31*a140);
  a140=(a56*a140);
  a140=(a50*a140);
  a140=(a139?a140:0);
  a54=(a54+a140);
  a141=(a31/a141);
  a56=(a56*a141);
  a56=(a50*a56);
  a41=(a41*a56);
  a41=(-a41);
  a139=(a139?a41:0);
  a54=(a54+a139);
  a129=(a31*a129);
  a129=(a50*a129);
  a129=(a129*a143);
  a129=(-a129);
  a129=(a35?a129:0);
  a54=(a54+a129);
  a31=(a31/a137);
  a50=(a50*a31);
  a50=(a50*a119);
  a50=(-a50);
  a35=(a35?a50:0);
  a54=(a54+a35);
  a35=(a81*a54);
  a147=(a147+a35);
  a35=(a147/a95);
  a50=(a90*a35);
  a59=(a59*a43);
  a43=(a33*a54);
  a59=(a59+a43);
  a95=(a59/a95);
  a90=(a90*a95);
  a43=(a50+a90);
  a119=(a64*a43);
  a17=(a17*a108);
  a17=(a17*a123);
  a16=(a16*a17);
  a16=(a15*a16);
  a6=(a6?a16:0);
  a108=(a138*a108);
  a42=(a42*a108);
  a42=(a124*a42);
  a30=(a30*a42);
  a13=(a13*a30);
  a15=(a15*a13);
  a6=(a6-a15);
  a5=(a5*a6);
  a5=(a5/a121);
  a122=(a122*a5);
  a149=(a149*a147);
  a150=(a150*a59);
  a149=(a149+a150);
  a149=(a149/a19);
  a70=(a70*a149);
  a19=(a122-a70);
  a136=(a136*a35);
  a130=(a130*a95);
  a136=(a136+a130);
  a19=(a19+a136);
  a130=(a105*a19);
  a119=(a119-a130);
  a130=(a125*a35);
  a125=(a125*a95);
  a150=(a130+a125);
  a59=(a105*a150);
  a119=(a119+a59);
  a94=(a94*a5);
  a103=(a103*a149);
  a149=(a94-a103);
  a133=(a133*a35);
  a126=(a126*a95);
  a133=(a133+a126);
  a149=(a149+a133);
  a126=(a64*a149);
  a119=(a119+a126);
  if (res[0]!=0) res[0][7]=a119;
  a119=cos(a0);
  a126=(a105*a142);
  a95=(a64*a154);
  a126=(a126-a95);
  a95=(a93*a126);
  a35=(a132*a154);
  a95=(a95+a35);
  a35=(a89*a95);
  a5=(a96*a126);
  a59=(a132*a142);
  a5=(a5-a59);
  a59=(a91*a5);
  a35=(a35-a59);
  a145=(a32*a145);
  a35=(a35+a145);
  a145=(a134*a142);
  a59=(a131*a154);
  a142=(a135*a142);
  a59=(a59-a142);
  a142=(a96*a59);
  a145=(a145+a142);
  a142=(a74*a152);
  a145=(a145-a142);
  a142=(a74*a156);
  a145=(a145+a142);
  a142=(a89*a145);
  a35=(a35+a142);
  a148=(a148-a52);
  a148=(a148+a51);
  a148=(a32*a148);
  a35=(a35+a148);
  a148=(a131*a126);
  a51=(a64*a59);
  a148=(a148+a51);
  a148=(a148-a146);
  a148=(a148-a157);
  a148=(a57*a148);
  a35=(a35+a148);
  a148=(a93*a59);
  a154=(a134*a154);
  a148=(a148-a154);
  a146=(a74*a146);
  a148=(a148+a146);
  a157=(a74*a157);
  a148=(a148+a157);
  a157=(a91*a148);
  a35=(a35+a157);
  a35=(a119*a35);
  a157=cos(a0);
  a146=9.8100000000000005e+00;
  a154=cos(a38);
  a51=4.8780487804878025e-01;
  a52=(a51*a154);
  a142=(a154*a52);
  a147=sin(a38);
  a121=(a51*a147);
  a6=(a147*a121);
  a142=(a142+a6);
  a142=(a146*a142);
  a142=(a157*a142);
  a6=sin(a0);
  a15=(a154*a121);
  a13=(a147*a52);
  a15=(a15-a13);
  a15=(a146*a15);
  a15=(a6*a15);
  a142=(a142+a15);
  a15=sin(a0);
  a13=(a89*a5);
  a30=(a91*a95);
  a13=(a13+a30);
  a151=(a151-a24);
  a151=(a151-a153);
  a151=(a32*a151);
  a13=(a13+a151);
  a151=(a91*a145);
  a13=(a13+a151);
  a126=(a135*a126);
  a59=(a105*a59);
  a126=(a126+a59);
  a126=(a126+a152);
  a126=(a126-a156);
  a126=(a57*a126);
  a13=(a13+a126);
  a155=(a32*a155);
  a13=(a13+a155);
  a155=(a89*a148);
  a13=(a13-a155);
  a13=(a15*a13);
  a142=(a142+a13);
  a35=(a35-a142);
  a142=sin(a0);
  a13=(a14*a102);
  a155=(a69*a115);
  a13=(a13-a155);
  a155=(a23*a13);
  a126=(a84*a102);
  a155=(a155-a126);
  a126=(a47*a155);
  a156=(a62*a13);
  a152=(a84*a115);
  a156=(a156+a152);
  a152=(a8*a156);
  a126=(a126+a152);
  a112=(a112-a99);
  a112=(a112-a114);
  a112=(a29*a112);
  a126=(a126+a112);
  a112=(a86*a102);
  a114=(a83*a115);
  a102=(a87*a102);
  a114=(a114-a102);
  a102=(a23*a114);
  a112=(a112+a102);
  a102=(a66*a113);
  a112=(a112-a102);
  a102=(a66*a117);
  a112=(a112+a102);
  a102=(a8*a112);
  a126=(a126+a102);
  a102=(a87*a13);
  a99=(a14*a114);
  a102=(a102+a99);
  a102=(a102+a113);
  a102=(a102-a117);
  a102=(a57*a102);
  a126=(a126+a102);
  a116=(a29*a116);
  a126=(a126+a116);
  a116=(a62*a114);
  a115=(a86*a115);
  a116=(a116-a115);
  a115=(a66*a107);
  a116=(a116+a115);
  a115=(a66*a118);
  a116=(a116+a115);
  a115=(a47*a116);
  a126=(a126-a115);
  a126=(a142*a126);
  a35=(a35-a126);
  a126=cos(a0);
  a115=(a47*a156);
  a102=(a8*a155);
  a115=(a115-a102);
  a106=(a29*a106);
  a115=(a115+a106);
  a106=(a47*a112);
  a115=(a115+a106);
  a109=(a109-a98);
  a109=(a109+a101);
  a109=(a29*a109);
  a115=(a115+a109);
  a13=(a83*a13);
  a114=(a69*a114);
  a13=(a13+a114);
  a13=(a13-a107);
  a13=(a13-a118);
  a13=(a57*a13);
  a115=(a115+a13);
  a13=(a8*a116);
  a115=(a115+a13);
  a115=(a126*a115);
  a35=(a35+a115);
  a115=sin(a0);
  a65=(a7*a65);
  a76=(a7*a76);
  a65=(a65+a76);
  a65=(a115*a65);
  a35=(a35-a65);
  a65=cos(a0);
  a58=(a7*a58);
  a75=(a7*a75);
  a58=(a58-a75);
  a58=(a65*a58);
  a35=(a35+a58);
  a58=sin(a0);
  a45=(a2*a45);
  a61=(a2*a61);
  a45=(a45+a61);
  a45=(a58*a45);
  a35=(a35-a45);
  a45=cos(a0);
  a25=(a2*a25);
  a60=(a2*a60);
  a25=(a25-a60);
  a25=(a45*a25);
  a35=(a35+a25);
  if (res[0]!=0) res[0][8]=a35;
  a35=(a105*a50);
  a25=(a64*a130);
  a35=(a35-a25);
  a25=(a93*a35);
  a60=(a132*a130);
  a25=(a25+a60);
  a60=(a89*a25);
  a61=(a96*a35);
  a132=(a132*a50);
  a61=(a61-a132);
  a132=(a91*a61);
  a60=(a60-a132);
  a90=(a32*a90);
  a60=(a60+a90);
  a90=(a134*a50);
  a132=(a131*a130);
  a50=(a135*a50);
  a132=(a132-a50);
  a96=(a96*a132);
  a90=(a90+a96);
  a96=(a74*a19);
  a90=(a90-a96);
  a96=(a74*a150);
  a90=(a90+a96);
  a96=(a89*a90);
  a60=(a60+a96);
  a94=(a94-a103);
  a94=(a94+a133);
  a94=(a32*a94);
  a60=(a60+a94);
  a131=(a131*a35);
  a64=(a64*a132);
  a131=(a131+a64);
  a131=(a131-a43);
  a131=(a131-a149);
  a131=(a57*a131);
  a60=(a60+a131);
  a93=(a93*a132);
  a134=(a134*a130);
  a93=(a93-a134);
  a43=(a74*a43);
  a93=(a93+a43);
  a74=(a74*a149);
  a93=(a93+a74);
  a74=(a91*a93);
  a60=(a60+a74);
  a119=(a119*a60);
  a60=-4.8780487804877992e-01;
  a74=(a60*a154);
  a149=(a154*a74);
  a43=(a60*a147);
  a134=(a147*a43);
  a149=(a149+a134);
  a149=(a146*a149);
  a157=(a157*a149);
  a149=(a154*a43);
  a134=(a147*a74);
  a149=(a149-a134);
  a149=(a146*a149);
  a6=(a6*a149);
  a157=(a157+a6);
  a6=(a89*a61);
  a149=(a91*a25);
  a6=(a6+a149);
  a70=(a70-a122);
  a70=(a70-a136);
  a70=(a32*a70);
  a6=(a6+a70);
  a91=(a91*a90);
  a6=(a6+a91);
  a135=(a135*a35);
  a105=(a105*a132);
  a135=(a135+a105);
  a135=(a135+a19);
  a135=(a135-a150);
  a135=(a57*a135);
  a6=(a6+a135);
  a32=(a32*a125);
  a6=(a6+a32);
  a89=(a89*a93);
  a6=(a6-a89);
  a15=(a15*a6);
  a157=(a157+a15);
  a119=(a119-a157);
  a157=(a14*a37);
  a15=(a69*a82);
  a157=(a157-a15);
  a15=(a23*a157);
  a6=(a84*a37);
  a15=(a15-a6);
  a6=(a47*a15);
  a89=(a62*a157);
  a84=(a84*a82);
  a89=(a89+a84);
  a84=(a8*a89);
  a6=(a6+a84);
  a72=(a72-a44);
  a72=(a72-a88);
  a72=(a29*a72);
  a6=(a6+a72);
  a72=(a86*a37);
  a88=(a83*a82);
  a37=(a87*a37);
  a88=(a88-a37);
  a23=(a23*a88);
  a72=(a72+a23);
  a23=(a66*a104);
  a72=(a72-a23);
  a23=(a66*a111);
  a72=(a72+a23);
  a23=(a8*a72);
  a6=(a6+a23);
  a87=(a87*a157);
  a14=(a14*a88);
  a87=(a87+a14);
  a87=(a87+a104);
  a87=(a87-a111);
  a87=(a57*a87);
  a6=(a6+a87);
  a77=(a29*a77);
  a6=(a6+a77);
  a62=(a62*a88);
  a86=(a86*a82);
  a62=(a62-a86);
  a86=(a66*a68);
  a62=(a62+a86);
  a66=(a66*a110);
  a62=(a62+a66);
  a66=(a47*a62);
  a6=(a6-a66);
  a142=(a142*a6);
  a119=(a119-a142);
  a142=(a47*a89);
  a6=(a8*a15);
  a142=(a142-a6);
  a53=(a29*a53);
  a142=(a142+a53);
  a47=(a47*a72);
  a142=(a142+a47);
  a63=(a63-a73);
  a63=(a63+a85);
  a29=(a29*a63);
  a142=(a142+a29);
  a83=(a83*a157);
  a69=(a69*a88);
  a83=(a83+a69);
  a83=(a83-a68);
  a83=(a83-a110);
  a83=(a57*a83);
  a142=(a142+a83);
  a8=(a8*a62);
  a142=(a142+a8);
  a126=(a126*a142);
  a119=(a119+a126);
  a49=(a7*a49);
  a36=(a7*a36);
  a49=(a49+a36);
  a115=(a115*a49);
  a119=(a119-a115);
  a1=(a7*a1);
  a7=(a7*a21);
  a1=(a1-a7);
  a65=(a65*a1);
  a119=(a119+a65);
  a4=(a2*a4);
  a20=(a2*a20);
  a4=(a4+a20);
  a58=(a58*a4);
  a119=(a119-a58);
  a10=(a2*a10);
  a2=(a2*a3);
  a10=(a10-a2);
  a45=(a45*a10);
  a119=(a119+a45);
  if (res[0]!=0) res[0][9]=a119;
  a119=cos(a38);
  a45=(a57*a33);
  a33=(a33+a81);
  a10=(a33*a52);
  a2=(a81*a52);
  a10=(a10-a2);
  a2=(a45*a10);
  a3=cos(a0);
  a3=(a146*a3);
  a58=(a3*a52);
  a2=(a2-a58);
  a58=(a154*a3);
  a0=sin(a0);
  a146=(a146*a0);
  a0=(a147*a146);
  a58=(a58-a0);
  a0=(a154*a45);
  a4=(a0*a81);
  a58=(a58+a4);
  a4=(a33*a0);
  a58=(a58-a4);
  a4=(a51*a58);
  a2=(a2+a4);
  a4=(a146*a121);
  a2=(a2-a4);
  a2=(a119*a2);
  a4=sin(a38);
  a20=(a147*a45);
  a65=(a33*a20);
  a1=(a154*a146);
  a7=(a147*a3);
  a1=(a1+a7);
  a7=(a20*a81);
  a1=(a1+a7);
  a65=(a65-a1);
  a51=(a51*a65);
  a1=(a146*a52);
  a51=(a51-a1);
  a1=(a81*a121);
  a7=(a33*a121);
  a1=(a1-a7);
  a7=(a45*a1);
  a51=(a51+a7);
  a7=(a3*a121);
  a51=(a51+a7);
  a51=(a4*a51);
  a2=(a2-a51);
  a51=sin(a38);
  a7=(a97*a5);
  a21=(a78*a95);
  a7=(a7+a21);
  a21=(a78*a145);
  a7=(a7+a21);
  a21=(a97*a148);
  a7=(a7-a21);
  a7=(a51*a7);
  a2=(a2-a7);
  a7=cos(a38);
  a95=(a97*a95);
  a5=(a78*a5);
  a95=(a95-a5);
  a145=(a97*a145);
  a95=(a95+a145);
  a148=(a78*a148);
  a95=(a95+a148);
  a95=(a7*a95);
  a2=(a2+a95);
  a95=sin(a38);
  a148=(a67*a155);
  a145=(a48*a156);
  a148=(a148+a145);
  a145=(a48*a112);
  a148=(a148+a145);
  a145=(a67*a116);
  a148=(a148-a145);
  a148=(a95*a148);
  a2=(a2-a148);
  a38=cos(a38);
  a156=(a67*a156);
  a155=(a48*a155);
  a156=(a156-a155);
  a112=(a67*a112);
  a156=(a156+a112);
  a116=(a48*a116);
  a156=(a156+a116);
  a156=(a38*a156);
  a2=(a2+a156);
  if (res[0]!=0) res[0][10]=a2;
  a2=(a33*a74);
  a156=(a81*a74);
  a2=(a2-a156);
  a156=(a45*a2);
  a116=(a3*a74);
  a156=(a156-a116);
  a58=(a60*a58);
  a156=(a156+a58);
  a58=(a146*a43);
  a156=(a156-a58);
  a119=(a119*a156);
  a60=(a60*a65);
  a146=(a146*a74);
  a60=(a60-a146);
  a81=(a81*a43);
  a33=(a33*a43);
  a81=(a81-a33);
  a45=(a45*a81);
  a60=(a60+a45);
  a3=(a3*a43);
  a60=(a60+a3);
  a4=(a4*a60);
  a119=(a119-a4);
  a4=(a97*a61);
  a60=(a78*a25);
  a4=(a4+a60);
  a60=(a78*a90);
  a4=(a4+a60);
  a60=(a97*a93);
  a4=(a4-a60);
  a51=(a51*a4);
  a119=(a119-a51);
  a25=(a97*a25);
  a61=(a78*a61);
  a25=(a25-a61);
  a97=(a97*a90);
  a25=(a25+a97);
  a78=(a78*a93);
  a25=(a25+a78);
  a7=(a7*a25);
  a119=(a119+a7);
  a7=(a67*a15);
  a25=(a48*a89);
  a7=(a7+a25);
  a25=(a48*a72);
  a7=(a7+a25);
  a25=(a67*a62);
  a7=(a7-a25);
  a95=(a95*a7);
  a119=(a119-a95);
  a89=(a67*a89);
  a15=(a48*a15);
  a89=(a89-a15);
  a67=(a67*a72);
  a89=(a89+a67);
  a48=(a48*a62);
  a89=(a89+a48);
  a38=(a38*a89);
  a119=(a119+a38);
  if (res[0]!=0) res[0][11]=a119;
  a119=-1.;
  if (res[0]!=0) res[0][12]=a119;
  a38=(a20*a52);
  a89=(a0*a121);
  a38=(a38-a89);
  a10=(a147*a10);
  a1=(a154*a1);
  a10=(a10+a1);
  a10=(a57*a10);
  a10=(a38+a10);
  a1=(a128*a26);
  a10=(a10+a1);
  a1=(a80*a100);
  a10=(a10+a1);
  a71=(a34*a71);
  a10=(a10+a71);
  a55=(a22*a55);
  a10=(a10+a55);
  if (res[0]!=0) res[0][13]=a10;
  a10=(a20*a74);
  a55=(a0*a43);
  a10=(a10-a55);
  a147=(a147*a2);
  a154=(a154*a81);
  a147=(a147+a154);
  a57=(a57*a147);
  a57=(a10+a57);
  a147=(a128*a54);
  a57=(a57+a147);
  a147=(a80*a120);
  a57=(a57+a147);
  a9=(a34*a9);
  a57=(a57+a9);
  a11=(a22*a11);
  a57=(a57+a11);
  if (res[0]!=0) res[0][14]=a57;
  if (res[0]!=0) res[0][15]=a119;
  a52=(a20*a52);
  a38=(a38-a52);
  a121=(a0*a121);
  a38=(a38+a121);
  a26=(a127*a26);
  a38=(a38+a26);
  a100=(a79*a100);
  a38=(a38+a100);
  if (res[0]!=0) res[0][16]=a38;
  a20=(a20*a74);
  a10=(a10-a20);
  a0=(a0*a43);
  a10=(a10+a0);
  a54=(a127*a54);
  a10=(a10+a54);
  a120=(a79*a120);
  a10=(a10+a120);
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
  a120=1.4439765966454325e+00;
  a54=-2.7025639012821762e-01;
  a39=(a39*a12);
  a39=(a18*a39);
  a22=(a22*a39);
  a39=(a54*a22);
  a39=(a120*a39);
  a12=(a10*a39);
  a0=9.6278838983177628e-01;
  a22=(a0*a22);
  a12=(a12-a22);
  a12=(a28*a12);
  a12=(-a12);
  if (res[2]!=0) res[2][0]=a12;
  if (res[2]!=0) res[2][1]=a39;
  a27=(a27*a46);
  a27=(a18*a27);
  a34=(a34*a27);
  a27=(a54*a34);
  a27=(a120*a27);
  a46=(a10*a27);
  a34=(a0*a34);
  a46=(a46-a34);
  a46=(a28*a46);
  a46=(-a46);
  if (res[2]!=0) res[2][2]=a46;
  if (res[2]!=0) res[2][3]=a27;
  a92=(a92*a40);
  a92=(a18*a92);
  a80=(a80*a92);
  a40=(a54*a80);
  a27=9.6278838983177639e-01;
  a79=(a79*a92);
  a92=(a27*a79);
  a40=(a40+a92);
  a40=(a120*a40);
  a92=(a10*a40);
  a80=(a0*a80);
  a79=(a28*a79);
  a80=(a80+a79);
  a92=(a92-a80);
  a92=(a28*a92);
  a92=(-a92);
  if (res[2]!=0) res[2][4]=a92;
  if (res[2]!=0) res[2][5]=a40;
  a138=(a138*a124);
  a18=(a18*a138);
  a128=(a128*a18);
  a54=(a54*a128);
  a127=(a127*a18);
  a27=(a27*a127);
  a54=(a54+a27);
  a120=(a120*a54);
  a10=(a10*a120);
  a0=(a0*a128);
  a127=(a28*a127);
  a0=(a0+a127);
  a10=(a10-a0);
  a28=(a28*a10);
  a28=(-a28);
  if (res[2]!=0) res[2][6]=a28;
  if (res[2]!=0) res[2][7]=a120;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15204294_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
