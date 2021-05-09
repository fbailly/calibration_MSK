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
  #define CASADI_PREFIX(ID) model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_ ## ID
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
static const casadi_int casadi_s5[3] = {8, 0, 0};

/* model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a13=(a10*a12);
  a14=(a5*a11);
  a7=(a5*a7);
  a15=(a14*a7);
  a13=(a13-a15);
  a15=(a8+a9);
  a16=casadi_sq(a15);
  a17=casadi_sq(a14);
  a16=(a16+a17);
  a16=sqrt(a16);
  a13=(a13/a16);
  a17=700.;
  a18=arg[2]? arg[2][0] : 0;
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a19=casadi_sq(a14);
  a9=(a9+a19);
  a9=sqrt(a9);
  a19=4.0000000000000001e-02;
  a20=(a9-a19);
  a21=8.7758256189037276e-01;
  a20=(a20/a21);
  a22=arg[0]? arg[0][0] : 0;
  a23=(a20/a22);
  a24=1.;
  a25=(a23-a24);
  a26=casadi_sq(a25);
  a27=4.5000000000000001e-01;
  a26=(a26/a27);
  a26=(-a26);
  a26=exp(a26);
  a28=(a18*a26);
  a29=(a13*a1);
  a30=0.;
  a31=(a29<=a30);
  a32=fabs(a29);
  a33=10.;
  a32=(a32/a33);
  a32=(a24-a32);
  a34=fabs(a29);
  a34=(a34/a33);
  a34=(a24+a34);
  a32=(a32/a34);
  a35=(a31?a32:0);
  a36=(!a31);
  a37=1.3300000000000001e+00;
  a38=(a37*a29);
  a38=(a38/a33);
  a39=-8.2500000000000004e-02;
  a38=(a38/a39);
  a38=(a24-a38);
  a40=(a29/a33);
  a40=(a40/a39);
  a40=(a24-a40);
  a38=(a38/a40);
  a41=(a36?a38:0);
  a35=(a35+a41);
  a41=(a28*a35);
  a42=(a19<a20);
  a20=(a20/a22);
  a43=(a20-a24);
  a43=(a33*a43);
  a43=exp(a43);
  a44=(a43-a24);
  a45=1.4741315910257660e+02;
  a44=(a44/a45);
  a44=(a42?a44:0);
  a41=(a41+a44);
  a44=1.0000000000000001e-01;
  a46=(a33*a22);
  a47=(a29/a46);
  a48=(a44*a47);
  a41=(a41+a48);
  a41=(a17*a41);
  a48=(a13*a41);
  a49=sin(a6);
  a50=(a5*a49);
  a51=(a50+a5);
  a52=cos(a6);
  a53=(a5*a52);
  a54=(a51*a53);
  a55=(a5*a52);
  a49=(a5*a49);
  a56=(a55*a49);
  a54=(a54-a56);
  a56=(a50+a5);
  a57=casadi_sq(a56);
  a58=casadi_sq(a55);
  a57=(a57+a58);
  a57=sqrt(a57);
  a54=(a54/a57);
  a58=arg[2]? arg[2][1] : 0;
  a50=(a50+a5);
  a59=casadi_sq(a50);
  a60=casadi_sq(a55);
  a59=(a59+a60);
  a59=sqrt(a59);
  a60=(a59-a19);
  a60=(a60/a21);
  a61=arg[0]? arg[0][1] : 0;
  a62=(a60/a61);
  a63=(a62-a24);
  a64=casadi_sq(a63);
  a64=(a64/a27);
  a64=(-a64);
  a64=exp(a64);
  a65=(a58*a64);
  a66=(a54*a1);
  a67=(a66<=a30);
  a68=fabs(a66);
  a68=(a68/a33);
  a68=(a24-a68);
  a69=fabs(a66);
  a69=(a69/a33);
  a69=(a24+a69);
  a68=(a68/a69);
  a70=(a67?a68:0);
  a71=(!a67);
  a72=(a37*a66);
  a72=(a72/a33);
  a72=(a72/a39);
  a72=(a24-a72);
  a73=(a66/a33);
  a73=(a73/a39);
  a73=(a24-a73);
  a72=(a72/a73);
  a74=(a71?a72:0);
  a70=(a70+a74);
  a74=(a65*a70);
  a75=(a19<a60);
  a60=(a60/a61);
  a76=(a60-a24);
  a76=(a33*a76);
  a76=exp(a76);
  a77=(a76-a24);
  a77=(a77/a45);
  a77=(a75?a77:0);
  a74=(a74+a77);
  a77=(a33*a61);
  a78=(a66/a77);
  a79=(a44*a78);
  a74=(a74+a79);
  a74=(a17*a74);
  a79=(a54*a74);
  a48=(a48+a79);
  a79=arg[0]? arg[0][5] : 0;
  a80=sin(a79);
  a81=sin(a6);
  a82=(a80*a81);
  a83=cos(a79);
  a84=cos(a6);
  a85=(a83*a84);
  a82=(a82-a85);
  a85=(a5*a82);
  a86=1.2500000000000000e+00;
  a87=(a86*a81);
  a85=(a85-a87);
  a88=7.5000000000000000e-01;
  a89=(a88*a81);
  a90=(a85+a89);
  a91=(a88*a84);
  a92=(a86*a84);
  a93=(a83*a81);
  a94=(a80*a84);
  a93=(a93+a94);
  a94=(a5*a93);
  a94=(a92-a94);
  a91=(a91-a94);
  a95=(a90*a91);
  a96=(a5*a93);
  a96=(a92-a96);
  a97=(a88*a84);
  a98=(a96-a97);
  a99=(a5*a82);
  a99=(a99-a87);
  a100=(a88*a81);
  a100=(a99+a100);
  a101=(a98*a100);
  a95=(a95+a101);
  a101=(a85+a89);
  a102=casadi_sq(a101);
  a103=(a96-a97);
  a104=casadi_sq(a103);
  a102=(a102+a104);
  a102=sqrt(a102);
  a95=(a95/a102);
  a104=arg[2]? arg[2][2] : 0;
  a85=(a85+a89);
  a89=casadi_sq(a85);
  a96=(a96-a97);
  a97=casadi_sq(a96);
  a89=(a89+a97);
  a89=sqrt(a89);
  a97=(a89-a19);
  a97=(a97/a21);
  a105=arg[0]? arg[0][2] : 0;
  a106=(a97/a105);
  a107=(a106-a24);
  a108=casadi_sq(a107);
  a108=(a108/a27);
  a108=(-a108);
  a108=exp(a108);
  a109=(a104*a108);
  a110=(a95*a1);
  a111=(a80*a84);
  a112=(a83*a81);
  a111=(a111+a112);
  a112=(a82*a87);
  a113=(a93*a92);
  a112=(a112+a113);
  a113=(a111*a112);
  a114=(a111*a87);
  a115=(a83*a84);
  a116=(a80*a81);
  a115=(a115-a116);
  a116=(a115*a92);
  a114=(a114+a116);
  a116=(a82*a114);
  a113=(a113-a116);
  a113=(a113-a94);
  a94=(a90*a113);
  a116=(a93*a114);
  a117=(a115*a112);
  a116=(a116-a117);
  a116=(a116+a99);
  a99=(a98*a116);
  a94=(a94+a99);
  a94=(a94/a102);
  a99=(a94*a2);
  a110=(a110+a99);
  a99=(a110<=a30);
  a117=fabs(a110);
  a117=(a117/a33);
  a117=(a24-a117);
  a118=fabs(a110);
  a118=(a118/a33);
  a118=(a24+a118);
  a117=(a117/a118);
  a119=(a99?a117:0);
  a120=(!a99);
  a121=(a37*a110);
  a121=(a121/a33);
  a121=(a121/a39);
  a121=(a24-a121);
  a122=(a110/a33);
  a122=(a122/a39);
  a122=(a24-a122);
  a121=(a121/a122);
  a123=(a120?a121:0);
  a119=(a119+a123);
  a123=(a109*a119);
  a124=(a19<a97);
  a97=(a97/a105);
  a125=(a97-a24);
  a125=(a33*a125);
  a125=exp(a125);
  a126=(a125-a24);
  a126=(a126/a45);
  a126=(a124?a126:0);
  a123=(a123+a126);
  a126=(a33*a105);
  a127=(a110/a126);
  a128=(a44*a127);
  a123=(a123+a128);
  a123=(a17*a123);
  a128=(a95*a123);
  a48=(a48+a128);
  a128=sin(a79);
  a129=sin(a6);
  a130=(a128*a129);
  a131=cos(a79);
  a132=cos(a6);
  a133=(a131*a132);
  a130=(a130-a133);
  a133=(a5*a130);
  a134=(a86*a129);
  a133=(a133-a134);
  a135=1.7500000000000000e+00;
  a136=(a135*a129);
  a137=(a133+a136);
  a138=(a135*a132);
  a139=(a86*a132);
  a140=(a131*a129);
  a141=(a128*a132);
  a140=(a140+a141);
  a141=(a5*a140);
  a141=(a139-a141);
  a138=(a138-a141);
  a142=(a137*a138);
  a143=(a5*a140);
  a143=(a139-a143);
  a144=(a135*a132);
  a145=(a143-a144);
  a146=(a5*a130);
  a146=(a146-a134);
  a147=(a135*a129);
  a147=(a146+a147);
  a148=(a145*a147);
  a142=(a142+a148);
  a148=(a133+a136);
  a149=casadi_sq(a148);
  a150=(a143-a144);
  a151=casadi_sq(a150);
  a149=(a149+a151);
  a149=sqrt(a149);
  a142=(a142/a149);
  a151=arg[2]? arg[2][3] : 0;
  a133=(a133+a136);
  a136=casadi_sq(a133);
  a143=(a143-a144);
  a144=casadi_sq(a143);
  a136=(a136+a144);
  a136=sqrt(a136);
  a144=(a136-a19);
  a144=(a144/a21);
  a21=arg[0]? arg[0][3] : 0;
  a152=(a144/a21);
  a153=(a152-a24);
  a154=casadi_sq(a153);
  a154=(a154/a27);
  a154=(-a154);
  a154=exp(a154);
  a27=(a151*a154);
  a155=(a142*a1);
  a156=(a128*a132);
  a157=(a131*a129);
  a156=(a156+a157);
  a157=(a130*a134);
  a158=(a140*a139);
  a157=(a157+a158);
  a158=(a156*a157);
  a159=(a156*a134);
  a160=(a131*a132);
  a161=(a128*a129);
  a160=(a160-a161);
  a161=(a160*a139);
  a159=(a159+a161);
  a161=(a130*a159);
  a158=(a158-a161);
  a158=(a158-a141);
  a141=(a137*a158);
  a161=(a140*a159);
  a162=(a160*a157);
  a161=(a161-a162);
  a161=(a161+a146);
  a146=(a145*a161);
  a141=(a141+a146);
  a141=(a141/a149);
  a146=(a141*a2);
  a155=(a155+a146);
  a30=(a155<=a30);
  a146=fabs(a155);
  a146=(a146/a33);
  a146=(a24-a146);
  a162=fabs(a155);
  a162=(a162/a33);
  a162=(a24+a162);
  a146=(a146/a162);
  a163=(a30?a146:0);
  a164=(!a30);
  a165=(a37*a155);
  a165=(a165/a33);
  a165=(a165/a39);
  a165=(a24-a165);
  a166=(a155/a33);
  a166=(a166/a39);
  a166=(a24-a166);
  a165=(a165/a166);
  a39=(a164?a165:0);
  a163=(a163+a39);
  a39=(a27*a163);
  a19=(a19<a144);
  a144=(a144/a21);
  a167=(a144-a24);
  a167=(a33*a167);
  a167=exp(a167);
  a168=(a167-a24);
  a168=(a168/a45);
  a168=(a19?a168:0);
  a39=(a39+a168);
  a168=(a33*a21);
  a45=(a155/a168);
  a169=(a44*a45);
  a39=(a39+a169);
  a39=(a17*a39);
  a169=(a142*a39);
  a48=(a48+a169);
  a169=sin(a79);
  a170=cos(a79);
  a171=9.8100000000000005e+00;
  a172=cos(a6);
  a172=(a171*a172);
  a173=(a170*a172);
  a174=sin(a6);
  a174=(a171*a174);
  a175=(a169*a174);
  a173=(a173-a175);
  a175=(a86*a1);
  a176=(a170*a175);
  a177=(a176*a2);
  a173=(a173+a177);
  a177=(a1+a2);
  a178=(a177*a176);
  a173=(a173-a178);
  a178=(a169*a173);
  a179=(a169*a175);
  a180=(a177*a179);
  a181=(a170*a174);
  a182=(a169*a172);
  a181=(a181+a182);
  a182=(a179*a2);
  a181=(a181+a182);
  a180=(a180-a181);
  a181=(a170*a180);
  a178=(a178+a181);
  a178=(a86*a178);
  a48=(a48+a178);
  a4=(a4*a48);
  a178=9.6278838983177639e-01;
  a181=(a94*a123);
  a182=(a141*a39);
  a181=(a181+a182);
  a178=(a178*a181);
  a4=(a4+a178);
  a178=6.9253199970355839e-01;
  a4=(a4/a178);
  a3=(a3*a4);
  a178=9.6278838983177628e-01;
  a178=(a178*a48);
  a48=2.7025639012821789e-01;
  a48=(a48*a181);
  a178=(a178+a48);
  a3=(a3-a178);
  a178=3.7001900289039211e+00;
  a3=(a3/a178);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  a0=6.7836549063042314e-03;
  a4=3.9024390243902418e-01;
  a3=(a4*a13);
  a3=(a17*a3);
  a178=(a0*a3);
  a178=(a178*a43);
  a178=(a33*a178);
  a20=(a20/a22);
  a48=(a178*a20);
  a48=(-a48);
  a48=(a42?a48:0);
  a47=(a47/a46);
  a181=(a44*a3);
  a182=(a47*a181);
  a182=(a33*a182);
  a48=(a48-a182);
  a23=(a23/a22);
  a25=(a25+a25);
  a182=2.2222222222222223e+00;
  a183=(a35*a3);
  a183=(a18*a183);
  a183=(a26*a183);
  a183=(a182*a183);
  a183=(a25*a183);
  a184=(a23*a183);
  a48=(a48+a184);
  if (res[1]!=0) res[1][0]=a48;
  a48=-3.9024390243902396e-01;
  a184=(a48*a13);
  a184=(a17*a184);
  a185=(a0*a184);
  a185=(a185*a43);
  a185=(a33*a185);
  a20=(a185*a20);
  a20=(-a20);
  a20=(a42?a20:0);
  a43=(a44*a184);
  a47=(a47*a43);
  a47=(a33*a47);
  a20=(a20-a47);
  a35=(a35*a184);
  a18=(a18*a35);
  a26=(a26*a18);
  a26=(a182*a26);
  a25=(a25*a26);
  a23=(a23*a25);
  a20=(a20+a23);
  if (res[1]!=0) res[1][1]=a20;
  a20=(a4*a54);
  a20=(a17*a20);
  a23=(a0*a20);
  a23=(a23*a76);
  a23=(a33*a23);
  a60=(a60/a61);
  a26=(a23*a60);
  a26=(-a26);
  a26=(a75?a26:0);
  a78=(a78/a77);
  a18=(a44*a20);
  a35=(a78*a18);
  a35=(a33*a35);
  a26=(a26-a35);
  a62=(a62/a61);
  a63=(a63+a63);
  a35=(a70*a20);
  a35=(a58*a35);
  a35=(a64*a35);
  a35=(a182*a35);
  a35=(a63*a35);
  a47=(a62*a35);
  a26=(a26+a47);
  if (res[1]!=0) res[1][2]=a26;
  a26=(a48*a54);
  a26=(a17*a26);
  a47=(a0*a26);
  a47=(a47*a76);
  a47=(a33*a47);
  a60=(a47*a60);
  a60=(-a60);
  a60=(a75?a60:0);
  a76=(a44*a26);
  a78=(a78*a76);
  a78=(a33*a78);
  a60=(a60-a78);
  a70=(a70*a26);
  a58=(a58*a70);
  a64=(a64*a58);
  a64=(a182*a64);
  a63=(a63*a64);
  a62=(a62*a63);
  a60=(a60+a62);
  if (res[1]!=0) res[1][3]=a60;
  a60=-3.9024390243902440e-01;
  a62=(a60*a94);
  a64=(a4*a95);
  a62=(a62+a64);
  a62=(a17*a62);
  a64=(a0*a62);
  a64=(a64*a125);
  a64=(a33*a64);
  a97=(a97/a105);
  a58=(a64*a97);
  a58=(-a58);
  a58=(a124?a58:0);
  a127=(a127/a126);
  a70=(a44*a62);
  a78=(a127*a70);
  a78=(a33*a78);
  a58=(a58-a78);
  a106=(a106/a105);
  a107=(a107+a107);
  a78=(a119*a62);
  a78=(a104*a78);
  a78=(a108*a78);
  a78=(a182*a78);
  a78=(a107*a78);
  a186=(a106*a78);
  a58=(a58+a186);
  if (res[1]!=0) res[1][4]=a58;
  a58=1.3902439024390245e+00;
  a186=(a58*a94);
  a187=(a48*a95);
  a186=(a186+a187);
  a186=(a17*a186);
  a187=(a0*a186);
  a187=(a187*a125);
  a187=(a33*a187);
  a97=(a187*a97);
  a97=(-a97);
  a97=(a124?a97:0);
  a125=(a44*a186);
  a127=(a127*a125);
  a127=(a33*a127);
  a97=(a97-a127);
  a119=(a119*a186);
  a104=(a104*a119);
  a108=(a108*a104);
  a108=(a182*a108);
  a107=(a107*a108);
  a106=(a106*a107);
  a97=(a97+a106);
  if (res[1]!=0) res[1][5]=a97;
  a97=(a60*a141);
  a106=(a4*a142);
  a97=(a97+a106);
  a97=(a17*a97);
  a106=(a0*a97);
  a106=(a106*a167);
  a106=(a33*a106);
  a144=(a144/a21);
  a108=(a106*a144);
  a108=(-a108);
  a108=(a19?a108:0);
  a45=(a45/a168);
  a104=(a44*a97);
  a119=(a45*a104);
  a119=(a33*a119);
  a108=(a108-a119);
  a152=(a152/a21);
  a153=(a153+a153);
  a119=(a163*a97);
  a119=(a151*a119);
  a119=(a154*a119);
  a119=(a182*a119);
  a119=(a153*a119);
  a127=(a152*a119);
  a108=(a108+a127);
  if (res[1]!=0) res[1][6]=a108;
  a108=(a58*a141);
  a127=(a48*a142);
  a108=(a108+a127);
  a17=(a17*a108);
  a0=(a0*a17);
  a0=(a0*a167);
  a0=(a33*a0);
  a144=(a0*a144);
  a144=(-a144);
  a144=(a19?a144:0);
  a167=(a44*a17);
  a45=(a45*a167);
  a33=(a33*a45);
  a144=(a144-a33);
  a163=(a163*a17);
  a151=(a151*a163);
  a154=(a154*a151);
  a182=(a182*a154);
  a153=(a153*a182);
  a152=(a152*a153);
  a144=(a144+a152);
  if (res[1]!=0) res[1][7]=a144;
  a144=cos(a6);
  a152=(a60*a39);
  a104=(a104/a168);
  a182=-1.2121212121212121e+01;
  a97=(a27*a97);
  a165=(a165/a166);
  a154=(a97*a165);
  a154=(a182*a154);
  a154=(a44*a154);
  a154=(a164?a154:0);
  a104=(a104+a154);
  a154=(a97/a166);
  a154=(a182*a154);
  a154=(a44*a154);
  a154=(a37*a154);
  a154=(-a154);
  a154=(a164?a154:0);
  a104=(a104+a154);
  a146=(a146/a162);
  a154=(a97*a146);
  a154=(a44*a154);
  a151=casadi_sign(a155);
  a154=(a154*a151);
  a154=(-a154);
  a154=(a30?a154:0);
  a104=(a104+a154);
  a97=(a97/a162);
  a97=(a44*a97);
  a155=casadi_sign(a155);
  a97=(a97*a155);
  a97=(-a97);
  a97=(a30?a97:0);
  a104=(a104+a97);
  a97=(a2*a104);
  a152=(a152+a97);
  a97=(a152/a149);
  a154=(a145*a97);
  a163=(a140*a154);
  a33=(a137*a97);
  a45=(a130*a33);
  a163=(a163-a45);
  a45=(a134*a163);
  a108=(a157*a33);
  a45=(a45+a108);
  a108=(a131*a45);
  a127=(a139*a163);
  a188=(a157*a154);
  a127=(a127-a188);
  a188=(a128*a127);
  a108=(a108-a188);
  a188=(a4*a39);
  a189=(a1*a104);
  a188=(a188+a189);
  a189=(a188/a149);
  a190=(a145*a189);
  a191=(a135*a190);
  a108=(a108+a191);
  a191=(a159*a154);
  a192=(a156*a33);
  a193=(a160*a154);
  a192=(a192-a193);
  a193=(a139*a192);
  a191=(a191+a193);
  a143=(a143+a143);
  a193=1.1394939273245490e+00;
  a106=(a106/a21);
  a106=(a19?a106:0);
  a119=(a119/a21);
  a106=(a106-a119);
  a106=(a193*a106);
  a136=(a136+a136);
  a106=(a106/a136);
  a119=(a143*a106);
  a150=(a150+a150);
  a194=(a141/a149);
  a152=(a194*a152);
  a195=(a142/a149);
  a188=(a195*a188);
  a152=(a152+a188);
  a188=(a149+a149);
  a152=(a152/a188);
  a196=(a150*a152);
  a197=(a119-a196);
  a198=(a161*a97);
  a199=(a147*a189);
  a198=(a198+a199);
  a197=(a197+a198);
  a199=(a5*a197);
  a191=(a191-a199);
  a199=(a137*a189);
  a200=(a33+a199);
  a201=(a5*a200);
  a191=(a191+a201);
  a201=(a131*a191);
  a108=(a108+a201);
  a133=(a133+a133);
  a106=(a133*a106);
  a148=(a148+a148);
  a152=(a148*a152);
  a201=(a106-a152);
  a97=(a158*a97);
  a189=(a138*a189);
  a97=(a97+a189);
  a201=(a201+a97);
  a201=(a135*a201);
  a108=(a108+a201);
  a201=(a156*a163);
  a189=(a130*a192);
  a201=(a201+a189);
  a154=(a154+a190);
  a201=(a201-a154);
  a106=(a106-a152);
  a106=(a106+a97);
  a201=(a201-a106);
  a201=(a86*a201);
  a108=(a108+a201);
  a201=(a134*a192);
  a33=(a159*a33);
  a201=(a201-a33);
  a154=(a5*a154);
  a201=(a201+a154);
  a106=(a5*a106);
  a201=(a201+a106);
  a106=(a128*a201);
  a108=(a108+a106);
  a108=(a144*a108);
  a106=cos(a6);
  a154=4.8780487804878025e-01;
  a33=(a154*a170);
  a97=(a170*a33);
  a152=(a154*a169);
  a190=(a169*a152);
  a97=(a97+a190);
  a97=(a171*a97);
  a97=(a106*a97);
  a190=sin(a6);
  a189=(a170*a152);
  a202=(a169*a33);
  a189=(a189-a202);
  a189=(a171*a189);
  a189=(a190*a189);
  a97=(a97+a189);
  a189=sin(a6);
  a202=(a131*a127);
  a203=(a128*a45);
  a202=(a202+a203);
  a196=(a196-a119);
  a196=(a196-a198);
  a196=(a135*a196);
  a202=(a202+a196);
  a196=(a128*a191);
  a202=(a202+a196);
  a163=(a160*a163);
  a192=(a140*a192);
  a163=(a163+a192);
  a163=(a163+a197);
  a163=(a163-a200);
  a163=(a86*a163);
  a202=(a202+a163);
  a199=(a135*a199);
  a202=(a202+a199);
  a199=(a131*a201);
  a202=(a202-a199);
  a202=(a189*a202);
  a97=(a97+a202);
  a108=(a108-a97);
  a97=sin(a6);
  a60=(a60*a123);
  a70=(a70/a126);
  a62=(a109*a62);
  a121=(a121/a122);
  a202=(a62*a121);
  a202=(a182*a202);
  a202=(a44*a202);
  a202=(a120?a202:0);
  a70=(a70+a202);
  a202=(a62/a122);
  a202=(a182*a202);
  a202=(a44*a202);
  a202=(a37*a202);
  a202=(-a202);
  a202=(a120?a202:0);
  a70=(a70+a202);
  a117=(a117/a118);
  a202=(a62*a117);
  a202=(a44*a202);
  a199=casadi_sign(a110);
  a202=(a202*a199);
  a202=(-a202);
  a202=(a99?a202:0);
  a70=(a70+a202);
  a62=(a62/a118);
  a62=(a44*a62);
  a110=casadi_sign(a110);
  a62=(a62*a110);
  a62=(-a62);
  a62=(a99?a62:0);
  a70=(a70+a62);
  a62=(a2*a70);
  a60=(a60+a62);
  a62=(a60/a102);
  a202=(a98*a62);
  a163=(a93*a202);
  a200=(a90*a62);
  a197=(a82*a200);
  a163=(a163-a197);
  a197=(a92*a163);
  a192=(a112*a202);
  a197=(a197-a192);
  a192=(a83*a197);
  a196=(a87*a163);
  a198=(a112*a200);
  a196=(a196+a198);
  a198=(a80*a196);
  a192=(a192+a198);
  a103=(a103+a103);
  a198=(a94/a102);
  a60=(a198*a60);
  a119=(a95/a102);
  a203=(a4*a123);
  a204=(a1*a70);
  a203=(a203+a204);
  a204=(a119*a203);
  a60=(a60+a204);
  a204=(a102+a102);
  a60=(a60/a204);
  a205=(a103*a60);
  a96=(a96+a96);
  a64=(a64/a105);
  a64=(a124?a64:0);
  a78=(a78/a105);
  a64=(a64-a78);
  a64=(a193*a64);
  a89=(a89+a89);
  a64=(a64/a89);
  a78=(a96*a64);
  a206=(a205-a78);
  a207=(a116*a62);
  a203=(a203/a102);
  a208=(a100*a203);
  a207=(a207+a208);
  a206=(a206-a207);
  a206=(a88*a206);
  a192=(a192+a206);
  a206=(a114*a202);
  a208=(a111*a200);
  a209=(a115*a202);
  a208=(a208-a209);
  a209=(a92*a208);
  a206=(a206+a209);
  a78=(a78-a205);
  a78=(a78+a207);
  a207=(a5*a78);
  a206=(a206-a207);
  a207=(a90*a203);
  a205=(a200+a207);
  a209=(a5*a205);
  a206=(a206+a209);
  a209=(a80*a206);
  a192=(a192+a209);
  a209=(a115*a163);
  a210=(a93*a208);
  a209=(a209+a210);
  a209=(a209+a78);
  a209=(a209-a205);
  a209=(a86*a209);
  a192=(a192+a209);
  a207=(a88*a207);
  a192=(a192+a207);
  a207=(a87*a208);
  a200=(a114*a200);
  a207=(a207-a200);
  a200=(a98*a203);
  a202=(a202+a200);
  a209=(a5*a202);
  a207=(a207+a209);
  a85=(a85+a85);
  a64=(a85*a64);
  a101=(a101+a101);
  a60=(a101*a60);
  a209=(a64-a60);
  a62=(a113*a62);
  a203=(a91*a203);
  a62=(a62+a203);
  a209=(a209+a62);
  a203=(a5*a209);
  a207=(a207+a203);
  a203=(a83*a207);
  a192=(a192-a203);
  a192=(a97*a192);
  a108=(a108-a192);
  a192=cos(a6);
  a203=(a83*a196);
  a205=(a80*a197);
  a203=(a203-a205);
  a200=(a88*a200);
  a203=(a203+a200);
  a200=(a83*a206);
  a203=(a203+a200);
  a64=(a64-a60);
  a64=(a64+a62);
  a64=(a88*a64);
  a203=(a203+a64);
  a163=(a111*a163);
  a208=(a82*a208);
  a163=(a163+a208);
  a163=(a163-a202);
  a163=(a163-a209);
  a163=(a86*a163);
  a203=(a203+a163);
  a163=(a80*a207);
  a203=(a203+a163);
  a203=(a192*a203);
  a108=(a108+a203);
  a203=sin(a6);
  a23=(a23/a61);
  a23=(a75?a23:0);
  a35=(a35/a61);
  a23=(a23-a35);
  a23=(a193*a23);
  a59=(a59+a59);
  a23=(a23/a59);
  a35=(a52*a23);
  a163=(a54/a57);
  a209=(a4*a74);
  a18=(a18/a77);
  a20=(a65*a20);
  a72=(a72/a73);
  a202=(a20*a72);
  a202=(a182*a202);
  a202=(a44*a202);
  a202=(a71?a202:0);
  a18=(a18+a202);
  a202=(a20/a73);
  a202=(a182*a202);
  a202=(a44*a202);
  a202=(a37*a202);
  a202=(-a202);
  a202=(a71?a202:0);
  a18=(a18+a202);
  a68=(a68/a69);
  a202=(a20*a68);
  a202=(a44*a202);
  a208=casadi_sign(a66);
  a202=(a202*a208);
  a202=(-a202);
  a202=(a67?a202:0);
  a18=(a18+a202);
  a20=(a20/a69);
  a20=(a44*a20);
  a66=casadi_sign(a66);
  a20=(a20*a66);
  a20=(-a20);
  a20=(a67?a20:0);
  a18=(a18+a20);
  a20=(a1*a18);
  a209=(a209+a20);
  a20=(a163*a209);
  a202=(a57+a57);
  a20=(a20/a202);
  a64=(a52*a20);
  a35=(a35-a64);
  a209=(a209/a57);
  a64=(a49*a209);
  a35=(a35-a64);
  a35=(a5*a35);
  a64=(a51*a209);
  a64=(a5*a64);
  a35=(a35+a64);
  a35=(a203*a35);
  a108=(a108-a35);
  a35=cos(a6);
  a50=(a50+a50);
  a23=(a50*a23);
  a56=(a56+a56);
  a20=(a56*a20);
  a23=(a23-a20);
  a20=(a53*a209);
  a23=(a23+a20);
  a23=(a5*a23);
  a209=(a55*a209);
  a209=(a5*a209);
  a23=(a23-a209);
  a23=(a35*a23);
  a108=(a108+a23);
  a23=sin(a6);
  a178=(a178/a22);
  a178=(a42?a178:0);
  a183=(a183/a22);
  a178=(a178-a183);
  a178=(a193*a178);
  a9=(a9+a9);
  a178=(a178/a9);
  a183=(a11*a178);
  a209=(a13/a16);
  a4=(a4*a41);
  a181=(a181/a46);
  a3=(a28*a3);
  a38=(a38/a40);
  a20=(a3*a38);
  a20=(a182*a20);
  a20=(a44*a20);
  a20=(a36?a20:0);
  a181=(a181+a20);
  a20=(a3/a40);
  a20=(a182*a20);
  a20=(a44*a20);
  a20=(a37*a20);
  a20=(-a20);
  a20=(a36?a20:0);
  a181=(a181+a20);
  a32=(a32/a34);
  a20=(a3*a32);
  a20=(a44*a20);
  a64=casadi_sign(a29);
  a20=(a20*a64);
  a20=(-a20);
  a20=(a31?a20:0);
  a181=(a181+a20);
  a3=(a3/a34);
  a3=(a44*a3);
  a29=casadi_sign(a29);
  a3=(a3*a29);
  a3=(-a3);
  a3=(a31?a3:0);
  a181=(a181+a3);
  a3=(a1*a181);
  a4=(a4+a3);
  a3=(a209*a4);
  a20=(a16+a16);
  a3=(a3/a20);
  a62=(a11*a3);
  a183=(a183-a62);
  a4=(a4/a16);
  a62=(a7*a4);
  a183=(a183-a62);
  a183=(a5*a183);
  a62=(a10*a4);
  a62=(a5*a62);
  a183=(a183+a62);
  a183=(a23*a183);
  a108=(a108-a183);
  a6=cos(a6);
  a8=(a8+a8);
  a178=(a8*a178);
  a15=(a15+a15);
  a3=(a15*a3);
  a178=(a178-a3);
  a3=(a12*a4);
  a178=(a178+a3);
  a178=(a5*a178);
  a4=(a14*a4);
  a4=(a5*a4);
  a178=(a178-a4);
  a178=(a6*a178);
  a108=(a108+a178);
  if (res[1]!=0) res[1][8]=a108;
  a108=(a58*a39);
  a167=(a167/a168);
  a27=(a27*a17);
  a165=(a27*a165);
  a165=(a182*a165);
  a165=(a44*a165);
  a165=(a164?a165:0);
  a167=(a167+a165);
  a166=(a27/a166);
  a166=(a182*a166);
  a166=(a44*a166);
  a166=(a37*a166);
  a166=(-a166);
  a164=(a164?a166:0);
  a167=(a167+a164);
  a146=(a27*a146);
  a146=(a44*a146);
  a146=(a146*a151);
  a146=(-a146);
  a146=(a30?a146:0);
  a167=(a167+a146);
  a27=(a27/a162);
  a27=(a44*a27);
  a27=(a27*a155);
  a27=(-a27);
  a30=(a30?a27:0);
  a167=(a167+a30);
  a30=(a2*a167);
  a108=(a108+a30);
  a30=(a108/a149);
  a27=(a145*a30);
  a155=(a140*a27);
  a162=(a137*a30);
  a146=(a130*a162);
  a155=(a155-a146);
  a146=(a134*a155);
  a151=(a157*a162);
  a146=(a146+a151);
  a151=(a131*a146);
  a164=(a139*a155);
  a157=(a157*a27);
  a164=(a164-a157);
  a157=(a128*a164);
  a151=(a151-a157);
  a39=(a48*a39);
  a157=(a1*a167);
  a39=(a39+a157);
  a149=(a39/a149);
  a145=(a145*a149);
  a157=(a135*a145);
  a151=(a151+a157);
  a157=(a159*a27);
  a166=(a156*a162);
  a165=(a160*a27);
  a166=(a166-a165);
  a139=(a139*a166);
  a157=(a157+a139);
  a0=(a0/a21);
  a19=(a19?a0:0);
  a153=(a153/a21);
  a19=(a19-a153);
  a19=(a193*a19);
  a19=(a19/a136);
  a143=(a143*a19);
  a194=(a194*a108);
  a195=(a195*a39);
  a194=(a194+a195);
  a194=(a194/a188);
  a150=(a150*a194);
  a188=(a143-a150);
  a161=(a161*a30);
  a147=(a147*a149);
  a161=(a161+a147);
  a188=(a188+a161);
  a147=(a5*a188);
  a157=(a157-a147);
  a137=(a137*a149);
  a147=(a162+a137);
  a195=(a5*a147);
  a157=(a157+a195);
  a195=(a131*a157);
  a151=(a151+a195);
  a133=(a133*a19);
  a148=(a148*a194);
  a194=(a133-a148);
  a158=(a158*a30);
  a138=(a138*a149);
  a158=(a158+a138);
  a194=(a194+a158);
  a194=(a135*a194);
  a151=(a151+a194);
  a156=(a156*a155);
  a130=(a130*a166);
  a156=(a156+a130);
  a27=(a27+a145);
  a156=(a156-a27);
  a133=(a133-a148);
  a133=(a133+a158);
  a156=(a156-a133);
  a156=(a86*a156);
  a151=(a151+a156);
  a134=(a134*a166);
  a159=(a159*a162);
  a134=(a134-a159);
  a27=(a5*a27);
  a134=(a134+a27);
  a133=(a5*a133);
  a134=(a134+a133);
  a133=(a128*a134);
  a151=(a151+a133);
  a144=(a144*a151);
  a151=-4.8780487804877992e-01;
  a133=(a151*a170);
  a27=(a170*a133);
  a159=(a151*a169);
  a162=(a169*a159);
  a27=(a27+a162);
  a27=(a171*a27);
  a106=(a106*a27);
  a27=(a170*a159);
  a162=(a169*a133);
  a27=(a27-a162);
  a171=(a171*a27);
  a190=(a190*a171);
  a106=(a106+a190);
  a190=(a131*a164);
  a171=(a128*a146);
  a190=(a190+a171);
  a150=(a150-a143);
  a150=(a150-a161);
  a150=(a135*a150);
  a190=(a190+a150);
  a128=(a128*a157);
  a190=(a190+a128);
  a160=(a160*a155);
  a140=(a140*a166);
  a160=(a160+a140);
  a160=(a160+a188);
  a160=(a160-a147);
  a160=(a86*a160);
  a190=(a190+a160);
  a135=(a135*a137);
  a190=(a190+a135);
  a131=(a131*a134);
  a190=(a190-a131);
  a189=(a189*a190);
  a106=(a106+a189);
  a144=(a144-a106);
  a58=(a58*a123);
  a125=(a125/a126);
  a109=(a109*a186);
  a121=(a109*a121);
  a121=(a182*a121);
  a121=(a44*a121);
  a121=(a120?a121:0);
  a125=(a125+a121);
  a122=(a109/a122);
  a122=(a182*a122);
  a122=(a44*a122);
  a122=(a37*a122);
  a122=(-a122);
  a120=(a120?a122:0);
  a125=(a125+a120);
  a117=(a109*a117);
  a117=(a44*a117);
  a117=(a117*a199);
  a117=(-a117);
  a117=(a99?a117:0);
  a125=(a125+a117);
  a109=(a109/a118);
  a109=(a44*a109);
  a109=(a109*a110);
  a109=(-a109);
  a99=(a99?a109:0);
  a125=(a125+a99);
  a99=(a2*a125);
  a58=(a58+a99);
  a99=(a58/a102);
  a109=(a98*a99);
  a110=(a93*a109);
  a118=(a90*a99);
  a117=(a82*a118);
  a110=(a110-a117);
  a117=(a92*a110);
  a199=(a112*a109);
  a117=(a117-a199);
  a199=(a83*a117);
  a120=(a87*a110);
  a112=(a112*a118);
  a120=(a120+a112);
  a112=(a80*a120);
  a199=(a199+a112);
  a198=(a198*a58);
  a123=(a48*a123);
  a58=(a1*a125);
  a123=(a123+a58);
  a119=(a119*a123);
  a198=(a198+a119);
  a198=(a198/a204);
  a103=(a103*a198);
  a187=(a187/a105);
  a124=(a124?a187:0);
  a107=(a107/a105);
  a124=(a124-a107);
  a124=(a193*a124);
  a124=(a124/a89);
  a96=(a96*a124);
  a89=(a103-a96);
  a116=(a116*a99);
  a123=(a123/a102);
  a100=(a100*a123);
  a116=(a116+a100);
  a89=(a89-a116);
  a89=(a88*a89);
  a199=(a199+a89);
  a89=(a114*a109);
  a100=(a111*a118);
  a102=(a115*a109);
  a100=(a100-a102);
  a92=(a92*a100);
  a89=(a89+a92);
  a96=(a96-a103);
  a96=(a96+a116);
  a116=(a5*a96);
  a89=(a89-a116);
  a90=(a90*a123);
  a116=(a118+a90);
  a103=(a5*a116);
  a89=(a89+a103);
  a103=(a80*a89);
  a199=(a199+a103);
  a115=(a115*a110);
  a93=(a93*a100);
  a115=(a115+a93);
  a115=(a115+a96);
  a115=(a115-a116);
  a115=(a86*a115);
  a199=(a199+a115);
  a90=(a88*a90);
  a199=(a199+a90);
  a87=(a87*a100);
  a114=(a114*a118);
  a87=(a87-a114);
  a98=(a98*a123);
  a109=(a109+a98);
  a114=(a5*a109);
  a87=(a87+a114);
  a85=(a85*a124);
  a101=(a101*a198);
  a198=(a85-a101);
  a113=(a113*a99);
  a91=(a91*a123);
  a113=(a113+a91);
  a198=(a198+a113);
  a91=(a5*a198);
  a87=(a87+a91);
  a91=(a83*a87);
  a199=(a199-a91);
  a97=(a97*a199);
  a144=(a144-a97);
  a97=(a83*a120);
  a199=(a80*a117);
  a97=(a97-a199);
  a98=(a88*a98);
  a97=(a97+a98);
  a83=(a83*a89);
  a97=(a97+a83);
  a85=(a85-a101);
  a85=(a85+a113);
  a88=(a88*a85);
  a97=(a97+a88);
  a111=(a111*a110);
  a82=(a82*a100);
  a111=(a111+a82);
  a111=(a111-a109);
  a111=(a111-a198);
  a111=(a86*a111);
  a97=(a97+a111);
  a80=(a80*a87);
  a97=(a97+a80);
  a192=(a192*a97);
  a144=(a144+a192);
  a47=(a47/a61);
  a75=(a75?a47:0);
  a63=(a63/a61);
  a75=(a75-a63);
  a75=(a193*a75);
  a75=(a75/a59);
  a59=(a52*a75);
  a74=(a48*a74);
  a76=(a76/a77);
  a65=(a65*a26);
  a72=(a65*a72);
  a72=(a182*a72);
  a72=(a44*a72);
  a72=(a71?a72:0);
  a76=(a76+a72);
  a73=(a65/a73);
  a73=(a182*a73);
  a73=(a44*a73);
  a73=(a37*a73);
  a73=(-a73);
  a71=(a71?a73:0);
  a76=(a76+a71);
  a68=(a65*a68);
  a68=(a44*a68);
  a68=(a68*a208);
  a68=(-a68);
  a68=(a67?a68:0);
  a76=(a76+a68);
  a65=(a65/a69);
  a65=(a44*a65);
  a65=(a65*a66);
  a65=(-a65);
  a67=(a67?a65:0);
  a76=(a76+a67);
  a67=(a1*a76);
  a74=(a74+a67);
  a163=(a163*a74);
  a163=(a163/a202);
  a52=(a52*a163);
  a59=(a59-a52);
  a74=(a74/a57);
  a49=(a49*a74);
  a59=(a59-a49);
  a59=(a5*a59);
  a51=(a51*a74);
  a51=(a5*a51);
  a59=(a59+a51);
  a203=(a203*a59);
  a144=(a144-a203);
  a50=(a50*a75);
  a56=(a56*a163);
  a50=(a50-a56);
  a53=(a53*a74);
  a50=(a50+a53);
  a50=(a5*a50);
  a55=(a55*a74);
  a55=(a5*a55);
  a50=(a50-a55);
  a35=(a35*a50);
  a144=(a144+a35);
  a185=(a185/a22);
  a42=(a42?a185:0);
  a25=(a25/a22);
  a42=(a42-a25);
  a193=(a193*a42);
  a193=(a193/a9);
  a9=(a11*a193);
  a48=(a48*a41);
  a43=(a43/a46);
  a28=(a28*a184);
  a38=(a28*a38);
  a38=(a182*a38);
  a38=(a44*a38);
  a38=(a36?a38:0);
  a43=(a43+a38);
  a40=(a28/a40);
  a182=(a182*a40);
  a182=(a44*a182);
  a37=(a37*a182);
  a37=(-a37);
  a36=(a36?a37:0);
  a43=(a43+a36);
  a32=(a28*a32);
  a32=(a44*a32);
  a32=(a32*a64);
  a32=(-a32);
  a32=(a31?a32:0);
  a43=(a43+a32);
  a28=(a28/a34);
  a44=(a44*a28);
  a44=(a44*a29);
  a44=(-a44);
  a31=(a31?a44:0);
  a43=(a43+a31);
  a1=(a1*a43);
  a48=(a48+a1);
  a209=(a209*a48);
  a209=(a209/a20);
  a11=(a11*a209);
  a9=(a9-a11);
  a48=(a48/a16);
  a7=(a7*a48);
  a9=(a9-a7);
  a9=(a5*a9);
  a10=(a10*a48);
  a10=(a5*a10);
  a9=(a9+a10);
  a23=(a23*a9);
  a144=(a144-a23);
  a8=(a8*a193);
  a15=(a15*a209);
  a8=(a8-a15);
  a12=(a12*a48);
  a8=(a8+a12);
  a8=(a5*a8);
  a14=(a14*a48);
  a5=(a5*a14);
  a8=(a8-a5);
  a6=(a6*a8);
  a144=(a144+a6);
  if (res[1]!=0) res[1][9]=a144;
  a144=cos(a79);
  a6=(a177*a33);
  a8=(a2*a33);
  a6=(a6-a8);
  a8=(a175*a6);
  a5=(a172*a33);
  a8=(a8-a5);
  a5=(a154*a173);
  a8=(a8+a5);
  a5=(a174*a152);
  a8=(a8-a5);
  a8=(a144*a8);
  a5=sin(a79);
  a154=(a154*a180);
  a14=(a174*a33);
  a154=(a154-a14);
  a14=(a2*a152);
  a48=(a177*a152);
  a14=(a14-a48);
  a48=(a175*a14);
  a154=(a154+a48);
  a48=(a172*a152);
  a154=(a154+a48);
  a154=(a5*a154);
  a8=(a8-a154);
  a154=sin(a79);
  a48=(a132*a127);
  a12=(a129*a45);
  a48=(a48+a12);
  a12=(a129*a191);
  a48=(a48+a12);
  a12=(a132*a201);
  a48=(a48-a12);
  a48=(a154*a48);
  a8=(a8-a48);
  a48=cos(a79);
  a45=(a132*a45);
  a127=(a129*a127);
  a45=(a45-a127);
  a191=(a132*a191);
  a45=(a45+a191);
  a201=(a129*a201);
  a45=(a45+a201);
  a45=(a48*a45);
  a8=(a8+a45);
  a45=sin(a79);
  a201=(a84*a197);
  a191=(a81*a196);
  a201=(a201+a191);
  a191=(a81*a206);
  a201=(a201+a191);
  a191=(a84*a207);
  a201=(a201-a191);
  a201=(a45*a201);
  a8=(a8-a201);
  a79=cos(a79);
  a196=(a84*a196);
  a197=(a81*a197);
  a196=(a196-a197);
  a206=(a84*a206);
  a196=(a196+a206);
  a207=(a81*a207);
  a196=(a196+a207);
  a196=(a79*a196);
  a8=(a8+a196);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a177*a133);
  a196=(a2*a133);
  a8=(a8-a196);
  a196=(a175*a8);
  a207=(a172*a133);
  a196=(a196-a207);
  a173=(a151*a173);
  a196=(a196+a173);
  a173=(a174*a159);
  a196=(a196-a173);
  a144=(a144*a196);
  a151=(a151*a180);
  a174=(a174*a133);
  a151=(a151-a174);
  a2=(a2*a159);
  a177=(a177*a159);
  a2=(a2-a177);
  a175=(a175*a2);
  a151=(a151+a175);
  a172=(a172*a159);
  a151=(a151+a172);
  a5=(a5*a151);
  a144=(a144-a5);
  a5=(a132*a164);
  a151=(a129*a146);
  a5=(a5+a151);
  a151=(a129*a157);
  a5=(a5+a151);
  a151=(a132*a134);
  a5=(a5-a151);
  a154=(a154*a5);
  a144=(a144-a154);
  a146=(a132*a146);
  a164=(a129*a164);
  a146=(a146-a164);
  a132=(a132*a157);
  a146=(a146+a132);
  a129=(a129*a134);
  a146=(a146+a129);
  a48=(a48*a146);
  a144=(a144+a48);
  a48=(a84*a117);
  a146=(a81*a120);
  a48=(a48+a146);
  a146=(a81*a89);
  a48=(a48+a146);
  a146=(a84*a87);
  a48=(a48-a146);
  a45=(a45*a48);
  a144=(a144-a45);
  a120=(a84*a120);
  a117=(a81*a117);
  a120=(a120-a117);
  a84=(a84*a89);
  a120=(a120+a84);
  a81=(a81*a87);
  a120=(a120+a81);
  a79=(a79*a120);
  a144=(a144+a79);
  if (res[1]!=0) res[1][11]=a144;
  a144=-1.;
  if (res[1]!=0) res[1][12]=a144;
  a79=(a179*a33);
  a120=(a176*a152);
  a79=(a79-a120);
  a6=(a169*a6);
  a14=(a170*a14);
  a6=(a6+a14);
  a6=(a86*a6);
  a6=(a79+a6);
  a14=(a142*a104);
  a6=(a6+a14);
  a14=(a95*a70);
  a6=(a6+a14);
  a18=(a54*a18);
  a6=(a6+a18);
  a181=(a13*a181);
  a6=(a6+a181);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a179*a133);
  a181=(a176*a159);
  a6=(a6-a181);
  a169=(a169*a8);
  a170=(a170*a2);
  a169=(a169+a170);
  a86=(a86*a169);
  a86=(a6+a86);
  a142=(a142*a167);
  a86=(a86+a142);
  a95=(a95*a125);
  a86=(a86+a95);
  a54=(a54*a76);
  a86=(a86+a54);
  a13=(a13*a43);
  a86=(a86+a13);
  if (res[1]!=0) res[1][14]=a86;
  if (res[1]!=0) res[1][15]=a144;
  a33=(a179*a33);
  a79=(a79-a33);
  a152=(a176*a152);
  a79=(a79+a152);
  a104=(a141*a104);
  a79=(a79+a104);
  a70=(a94*a70);
  a79=(a79+a70);
  if (res[1]!=0) res[1][16]=a79;
  a179=(a179*a133);
  a6=(a6-a179);
  a176=(a176*a159);
  a6=(a6+a176);
  a141=(a141*a167);
  a6=(a6+a141);
  a94=(a94*a125);
  a6=(a6+a94);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a24;
  if (res[2]!=0) res[2][1]=a24;
  if (res[2]!=0) res[2][2]=a24;
  if (res[2]!=0) res[2][3]=a24;
  if (res[2]!=0) res[2][4]=a24;
  if (res[2]!=0) res[2][5]=a24;
  if (res[2]!=0) res[2][6]=a24;
  if (res[2]!=0) res[2][7]=a24;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112391_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
