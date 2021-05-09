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
  #define CASADI_PREFIX(ID) model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
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
  a17=3.0000006152352142e+02;
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
  a58=7.3182931678360399e+02;
  a59=arg[2]? arg[2][1] : 0;
  a50=(a50+a5);
  a60=casadi_sq(a50);
  a61=casadi_sq(a55);
  a60=(a60+a61);
  a60=sqrt(a60);
  a61=(a60-a19);
  a61=(a61/a21);
  a62=arg[0]? arg[0][1] : 0;
  a63=(a61/a62);
  a64=(a63-a24);
  a65=casadi_sq(a64);
  a65=(a65/a27);
  a65=(-a65);
  a65=exp(a65);
  a66=(a59*a65);
  a67=(a54*a1);
  a68=(a67<=a30);
  a69=fabs(a67);
  a69=(a69/a33);
  a69=(a24-a69);
  a70=fabs(a67);
  a70=(a70/a33);
  a70=(a24+a70);
  a69=(a69/a70);
  a71=(a68?a69:0);
  a72=(!a68);
  a73=(a37*a67);
  a73=(a73/a33);
  a73=(a73/a39);
  a73=(a24-a73);
  a74=(a67/a33);
  a74=(a74/a39);
  a74=(a24-a74);
  a73=(a73/a74);
  a75=(a72?a73:0);
  a71=(a71+a75);
  a75=(a66*a71);
  a76=(a19<a61);
  a61=(a61/a62);
  a77=(a61-a24);
  a77=(a33*a77);
  a77=exp(a77);
  a78=(a77-a24);
  a78=(a78/a45);
  a78=(a76?a78:0);
  a75=(a75+a78);
  a78=(a33*a62);
  a79=(a67/a78);
  a80=(a44*a79);
  a75=(a75+a80);
  a75=(a58*a75);
  a80=(a54*a75);
  a48=(a48+a80);
  a80=arg[0]? arg[0][5] : 0;
  a81=sin(a80);
  a82=sin(a6);
  a83=(a81*a82);
  a84=cos(a80);
  a85=cos(a6);
  a86=(a84*a85);
  a83=(a83-a86);
  a86=(a5*a83);
  a87=1.2500000000000000e+00;
  a88=(a87*a82);
  a86=(a86-a88);
  a89=7.5000000000000000e-01;
  a90=(a89*a82);
  a91=(a86+a90);
  a92=(a89*a85);
  a93=(a87*a85);
  a94=(a84*a82);
  a95=(a81*a85);
  a94=(a94+a95);
  a95=(a5*a94);
  a95=(a93-a95);
  a92=(a92-a95);
  a96=(a91*a92);
  a97=(a5*a94);
  a97=(a93-a97);
  a98=(a89*a85);
  a99=(a97-a98);
  a100=(a5*a83);
  a100=(a100-a88);
  a101=(a89*a82);
  a101=(a100+a101);
  a102=(a99*a101);
  a96=(a96+a102);
  a102=(a86+a90);
  a103=casadi_sq(a102);
  a104=(a97-a98);
  a105=casadi_sq(a104);
  a103=(a103+a105);
  a103=sqrt(a103);
  a96=(a96/a103);
  a105=7.3436559281095219e+02;
  a106=arg[2]? arg[2][2] : 0;
  a86=(a86+a90);
  a90=casadi_sq(a86);
  a97=(a97-a98);
  a98=casadi_sq(a97);
  a90=(a90+a98);
  a90=sqrt(a90);
  a98=(a90-a19);
  a98=(a98/a21);
  a107=arg[0]? arg[0][2] : 0;
  a108=(a98/a107);
  a109=(a108-a24);
  a110=casadi_sq(a109);
  a110=(a110/a27);
  a110=(-a110);
  a110=exp(a110);
  a111=(a106*a110);
  a112=(a96*a1);
  a113=(a81*a85);
  a114=(a84*a82);
  a113=(a113+a114);
  a114=(a83*a88);
  a115=(a94*a93);
  a114=(a114+a115);
  a115=(a113*a114);
  a116=(a113*a88);
  a117=(a84*a85);
  a118=(a81*a82);
  a117=(a117-a118);
  a118=(a117*a93);
  a116=(a116+a118);
  a118=(a83*a116);
  a115=(a115-a118);
  a115=(a115-a95);
  a95=(a91*a115);
  a118=(a94*a116);
  a119=(a117*a114);
  a118=(a118-a119);
  a118=(a118+a100);
  a100=(a99*a118);
  a95=(a95+a100);
  a95=(a95/a103);
  a100=(a95*a2);
  a112=(a112+a100);
  a100=(a112<=a30);
  a119=fabs(a112);
  a119=(a119/a33);
  a119=(a24-a119);
  a120=fabs(a112);
  a120=(a120/a33);
  a120=(a24+a120);
  a119=(a119/a120);
  a121=(a100?a119:0);
  a122=(!a100);
  a123=(a37*a112);
  a123=(a123/a33);
  a123=(a123/a39);
  a123=(a24-a123);
  a124=(a112/a33);
  a124=(a124/a39);
  a124=(a24-a124);
  a123=(a123/a124);
  a125=(a122?a123:0);
  a121=(a121+a125);
  a125=(a111*a121);
  a126=(a19<a98);
  a98=(a98/a107);
  a127=(a98-a24);
  a127=(a33*a127);
  a127=exp(a127);
  a128=(a127-a24);
  a128=(a128/a45);
  a128=(a126?a128:0);
  a125=(a125+a128);
  a128=(a33*a107);
  a129=(a112/a128);
  a130=(a44*a129);
  a125=(a125+a130);
  a125=(a105*a125);
  a130=(a96*a125);
  a48=(a48+a130);
  a130=sin(a80);
  a131=sin(a6);
  a132=(a130*a131);
  a133=cos(a80);
  a134=cos(a6);
  a135=(a133*a134);
  a132=(a132-a135);
  a135=(a5*a132);
  a136=(a87*a131);
  a135=(a135-a136);
  a137=1.7500000000000000e+00;
  a138=(a137*a131);
  a139=(a135+a138);
  a140=(a137*a134);
  a141=(a87*a134);
  a142=(a133*a131);
  a143=(a130*a134);
  a142=(a142+a143);
  a143=(a5*a142);
  a143=(a141-a143);
  a140=(a140-a143);
  a144=(a139*a140);
  a145=(a5*a142);
  a145=(a141-a145);
  a146=(a137*a134);
  a147=(a145-a146);
  a148=(a5*a132);
  a148=(a148-a136);
  a149=(a137*a131);
  a149=(a148+a149);
  a150=(a147*a149);
  a144=(a144+a150);
  a150=(a135+a138);
  a151=casadi_sq(a150);
  a152=(a145-a146);
  a153=casadi_sq(a152);
  a151=(a151+a153);
  a151=sqrt(a151);
  a144=(a144/a151);
  a153=6.8799326207123602e+02;
  a154=arg[2]? arg[2][3] : 0;
  a135=(a135+a138);
  a138=casadi_sq(a135);
  a145=(a145-a146);
  a146=casadi_sq(a145);
  a138=(a138+a146);
  a138=sqrt(a138);
  a146=(a138-a19);
  a146=(a146/a21);
  a21=arg[0]? arg[0][3] : 0;
  a155=(a146/a21);
  a156=(a155-a24);
  a157=casadi_sq(a156);
  a157=(a157/a27);
  a157=(-a157);
  a157=exp(a157);
  a27=(a154*a157);
  a158=(a144*a1);
  a159=(a130*a134);
  a160=(a133*a131);
  a159=(a159+a160);
  a160=(a132*a136);
  a161=(a142*a141);
  a160=(a160+a161);
  a161=(a159*a160);
  a162=(a159*a136);
  a163=(a133*a134);
  a164=(a130*a131);
  a163=(a163-a164);
  a164=(a163*a141);
  a162=(a162+a164);
  a164=(a132*a162);
  a161=(a161-a164);
  a161=(a161-a143);
  a143=(a139*a161);
  a164=(a142*a162);
  a165=(a163*a160);
  a164=(a164-a165);
  a164=(a164+a148);
  a148=(a147*a164);
  a143=(a143+a148);
  a143=(a143/a151);
  a148=(a143*a2);
  a158=(a158+a148);
  a30=(a158<=a30);
  a148=fabs(a158);
  a148=(a148/a33);
  a148=(a24-a148);
  a165=fabs(a158);
  a165=(a165/a33);
  a165=(a24+a165);
  a148=(a148/a165);
  a166=(a30?a148:0);
  a167=(!a30);
  a168=(a37*a158);
  a168=(a168/a33);
  a168=(a168/a39);
  a168=(a24-a168);
  a169=(a158/a33);
  a169=(a169/a39);
  a169=(a24-a169);
  a168=(a168/a169);
  a39=(a167?a168:0);
  a166=(a166+a39);
  a39=(a27*a166);
  a19=(a19<a146);
  a146=(a146/a21);
  a170=(a146-a24);
  a170=(a33*a170);
  a170=exp(a170);
  a171=(a170-a24);
  a171=(a171/a45);
  a171=(a19?a171:0);
  a39=(a39+a171);
  a171=(a33*a21);
  a45=(a158/a171);
  a172=(a44*a45);
  a39=(a39+a172);
  a39=(a153*a39);
  a172=(a144*a39);
  a48=(a48+a172);
  a172=sin(a80);
  a173=cos(a80);
  a174=9.8100000000000005e+00;
  a175=cos(a6);
  a175=(a174*a175);
  a176=(a173*a175);
  a177=sin(a6);
  a177=(a174*a177);
  a178=(a172*a177);
  a176=(a176-a178);
  a178=(a87*a1);
  a179=(a173*a178);
  a180=(a179*a2);
  a176=(a176+a180);
  a180=(a1+a2);
  a181=(a180*a179);
  a176=(a176-a181);
  a181=(a172*a176);
  a182=(a172*a178);
  a183=(a180*a182);
  a184=(a173*a177);
  a185=(a172*a175);
  a184=(a184+a185);
  a185=(a182*a2);
  a184=(a184+a185);
  a183=(a183-a184);
  a184=(a173*a183);
  a181=(a181+a184);
  a181=(a87*a181);
  a48=(a48+a181);
  a4=(a4*a48);
  a181=9.6278838983177639e-01;
  a184=(a95*a125);
  a185=(a143*a39);
  a184=(a184+a185);
  a181=(a181*a184);
  a4=(a4+a181);
  a181=6.9253199970355839e-01;
  a4=(a4/a181);
  a3=(a3*a4);
  a181=9.6278838983177628e-01;
  a181=(a181*a48);
  a48=2.7025639012821789e-01;
  a48=(a48*a184);
  a181=(a181+a48);
  a3=(a3-a181);
  a181=3.7001900289039211e+00;
  a3=(a3/a181);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  a0=6.7836549063042314e-03;
  a4=3.9024390243902418e-01;
  a3=(a4*a13);
  a3=(a17*a3);
  a181=(a0*a3);
  a181=(a181*a43);
  a181=(a33*a181);
  a20=(a20/a22);
  a48=(a181*a20);
  a48=(-a48);
  a48=(a42?a48:0);
  a47=(a47/a46);
  a184=(a44*a3);
  a185=(a47*a184);
  a185=(a33*a185);
  a48=(a48-a185);
  a23=(a23/a22);
  a25=(a25+a25);
  a185=2.2222222222222223e+00;
  a186=(a35*a3);
  a186=(a18*a186);
  a186=(a26*a186);
  a186=(a185*a186);
  a186=(a25*a186);
  a187=(a23*a186);
  a48=(a48+a187);
  if (res[1]!=0) res[1][0]=a48;
  a48=-3.9024390243902396e-01;
  a187=(a48*a13);
  a17=(a17*a187);
  a187=(a0*a17);
  a187=(a187*a43);
  a187=(a33*a187);
  a20=(a187*a20);
  a20=(-a20);
  a20=(a42?a20:0);
  a43=(a44*a17);
  a47=(a47*a43);
  a47=(a33*a47);
  a20=(a20-a47);
  a35=(a35*a17);
  a18=(a18*a35);
  a26=(a26*a18);
  a26=(a185*a26);
  a25=(a25*a26);
  a23=(a23*a25);
  a20=(a20+a23);
  if (res[1]!=0) res[1][1]=a20;
  a20=(a4*a54);
  a20=(a58*a20);
  a23=(a0*a20);
  a23=(a23*a77);
  a23=(a33*a23);
  a61=(a61/a62);
  a26=(a23*a61);
  a26=(-a26);
  a26=(a76?a26:0);
  a79=(a79/a78);
  a18=(a44*a20);
  a35=(a79*a18);
  a35=(a33*a35);
  a26=(a26-a35);
  a63=(a63/a62);
  a64=(a64+a64);
  a35=(a71*a20);
  a35=(a59*a35);
  a35=(a65*a35);
  a35=(a185*a35);
  a35=(a64*a35);
  a47=(a63*a35);
  a26=(a26+a47);
  if (res[1]!=0) res[1][2]=a26;
  a26=(a48*a54);
  a58=(a58*a26);
  a26=(a0*a58);
  a26=(a26*a77);
  a26=(a33*a26);
  a61=(a26*a61);
  a61=(-a61);
  a61=(a76?a61:0);
  a77=(a44*a58);
  a79=(a79*a77);
  a79=(a33*a79);
  a61=(a61-a79);
  a71=(a71*a58);
  a59=(a59*a71);
  a65=(a65*a59);
  a65=(a185*a65);
  a64=(a64*a65);
  a63=(a63*a64);
  a61=(a61+a63);
  if (res[1]!=0) res[1][3]=a61;
  a61=-3.9024390243902440e-01;
  a63=(a61*a95);
  a65=(a4*a96);
  a63=(a63+a65);
  a63=(a105*a63);
  a65=(a0*a63);
  a65=(a65*a127);
  a65=(a33*a65);
  a98=(a98/a107);
  a59=(a65*a98);
  a59=(-a59);
  a59=(a126?a59:0);
  a129=(a129/a128);
  a71=(a44*a63);
  a79=(a129*a71);
  a79=(a33*a79);
  a59=(a59-a79);
  a108=(a108/a107);
  a109=(a109+a109);
  a79=(a121*a63);
  a79=(a106*a79);
  a79=(a110*a79);
  a79=(a185*a79);
  a79=(a109*a79);
  a47=(a108*a79);
  a59=(a59+a47);
  if (res[1]!=0) res[1][4]=a59;
  a59=1.3902439024390245e+00;
  a47=(a59*a95);
  a188=(a48*a96);
  a47=(a47+a188);
  a105=(a105*a47);
  a47=(a0*a105);
  a47=(a47*a127);
  a47=(a33*a47);
  a98=(a47*a98);
  a98=(-a98);
  a98=(a126?a98:0);
  a127=(a44*a105);
  a129=(a129*a127);
  a129=(a33*a129);
  a98=(a98-a129);
  a121=(a121*a105);
  a106=(a106*a121);
  a110=(a110*a106);
  a110=(a185*a110);
  a109=(a109*a110);
  a108=(a108*a109);
  a98=(a98+a108);
  if (res[1]!=0) res[1][5]=a98;
  a98=(a61*a143);
  a108=(a4*a144);
  a98=(a98+a108);
  a98=(a153*a98);
  a108=(a0*a98);
  a108=(a108*a170);
  a108=(a33*a108);
  a146=(a146/a21);
  a110=(a108*a146);
  a110=(-a110);
  a110=(a19?a110:0);
  a45=(a45/a171);
  a106=(a44*a98);
  a121=(a45*a106);
  a121=(a33*a121);
  a110=(a110-a121);
  a155=(a155/a21);
  a156=(a156+a156);
  a121=(a166*a98);
  a121=(a154*a121);
  a121=(a157*a121);
  a121=(a185*a121);
  a121=(a156*a121);
  a129=(a155*a121);
  a110=(a110+a129);
  if (res[1]!=0) res[1][6]=a110;
  a110=(a59*a143);
  a129=(a48*a144);
  a110=(a110+a129);
  a153=(a153*a110);
  a0=(a0*a153);
  a0=(a0*a170);
  a0=(a33*a0);
  a146=(a0*a146);
  a146=(-a146);
  a146=(a19?a146:0);
  a170=(a44*a153);
  a45=(a45*a170);
  a33=(a33*a45);
  a146=(a146-a33);
  a166=(a166*a153);
  a154=(a154*a166);
  a157=(a157*a154);
  a185=(a185*a157);
  a156=(a156*a185);
  a155=(a155*a156);
  a146=(a146+a155);
  if (res[1]!=0) res[1][7]=a146;
  a146=cos(a6);
  a155=(a61*a39);
  a106=(a106/a171);
  a185=-1.2121212121212121e+01;
  a98=(a27*a98);
  a168=(a168/a169);
  a157=(a98*a168);
  a157=(a185*a157);
  a157=(a44*a157);
  a157=(a167?a157:0);
  a106=(a106+a157);
  a157=(a98/a169);
  a157=(a185*a157);
  a157=(a44*a157);
  a157=(a37*a157);
  a157=(-a157);
  a157=(a167?a157:0);
  a106=(a106+a157);
  a148=(a148/a165);
  a157=(a98*a148);
  a157=(a44*a157);
  a154=casadi_sign(a158);
  a157=(a157*a154);
  a157=(-a157);
  a157=(a30?a157:0);
  a106=(a106+a157);
  a98=(a98/a165);
  a98=(a44*a98);
  a158=casadi_sign(a158);
  a98=(a98*a158);
  a98=(-a98);
  a98=(a30?a98:0);
  a106=(a106+a98);
  a98=(a2*a106);
  a155=(a155+a98);
  a98=(a155/a151);
  a157=(a147*a98);
  a166=(a142*a157);
  a33=(a139*a98);
  a45=(a132*a33);
  a166=(a166-a45);
  a45=(a136*a166);
  a110=(a160*a33);
  a45=(a45+a110);
  a110=(a133*a45);
  a129=(a141*a166);
  a188=(a160*a157);
  a129=(a129-a188);
  a188=(a130*a129);
  a110=(a110-a188);
  a188=(a4*a39);
  a189=(a1*a106);
  a188=(a188+a189);
  a189=(a188/a151);
  a190=(a147*a189);
  a191=(a137*a190);
  a110=(a110+a191);
  a191=(a162*a157);
  a192=(a159*a33);
  a193=(a163*a157);
  a192=(a192-a193);
  a193=(a141*a192);
  a191=(a191+a193);
  a145=(a145+a145);
  a193=1.1394939273245490e+00;
  a108=(a108/a21);
  a108=(a19?a108:0);
  a121=(a121/a21);
  a108=(a108-a121);
  a108=(a193*a108);
  a138=(a138+a138);
  a108=(a108/a138);
  a121=(a145*a108);
  a152=(a152+a152);
  a194=(a143/a151);
  a155=(a194*a155);
  a195=(a144/a151);
  a188=(a195*a188);
  a155=(a155+a188);
  a188=(a151+a151);
  a155=(a155/a188);
  a196=(a152*a155);
  a197=(a121-a196);
  a198=(a164*a98);
  a199=(a149*a189);
  a198=(a198+a199);
  a197=(a197+a198);
  a199=(a5*a197);
  a191=(a191-a199);
  a199=(a139*a189);
  a200=(a33+a199);
  a201=(a5*a200);
  a191=(a191+a201);
  a201=(a133*a191);
  a110=(a110+a201);
  a135=(a135+a135);
  a108=(a135*a108);
  a150=(a150+a150);
  a155=(a150*a155);
  a201=(a108-a155);
  a98=(a161*a98);
  a189=(a140*a189);
  a98=(a98+a189);
  a201=(a201+a98);
  a201=(a137*a201);
  a110=(a110+a201);
  a201=(a159*a166);
  a189=(a132*a192);
  a201=(a201+a189);
  a157=(a157+a190);
  a201=(a201-a157);
  a108=(a108-a155);
  a108=(a108+a98);
  a201=(a201-a108);
  a201=(a87*a201);
  a110=(a110+a201);
  a201=(a136*a192);
  a33=(a162*a33);
  a201=(a201-a33);
  a157=(a5*a157);
  a201=(a201+a157);
  a108=(a5*a108);
  a201=(a201+a108);
  a108=(a130*a201);
  a110=(a110+a108);
  a110=(a146*a110);
  a108=cos(a6);
  a157=4.8780487804878025e-01;
  a33=(a157*a173);
  a98=(a173*a33);
  a155=(a157*a172);
  a190=(a172*a155);
  a98=(a98+a190);
  a98=(a174*a98);
  a98=(a108*a98);
  a190=sin(a6);
  a189=(a173*a155);
  a202=(a172*a33);
  a189=(a189-a202);
  a189=(a174*a189);
  a189=(a190*a189);
  a98=(a98+a189);
  a189=sin(a6);
  a202=(a133*a129);
  a203=(a130*a45);
  a202=(a202+a203);
  a196=(a196-a121);
  a196=(a196-a198);
  a196=(a137*a196);
  a202=(a202+a196);
  a196=(a130*a191);
  a202=(a202+a196);
  a166=(a163*a166);
  a192=(a142*a192);
  a166=(a166+a192);
  a166=(a166+a197);
  a166=(a166-a200);
  a166=(a87*a166);
  a202=(a202+a166);
  a199=(a137*a199);
  a202=(a202+a199);
  a199=(a133*a201);
  a202=(a202-a199);
  a202=(a189*a202);
  a98=(a98+a202);
  a110=(a110-a98);
  a98=sin(a6);
  a61=(a61*a125);
  a71=(a71/a128);
  a63=(a111*a63);
  a123=(a123/a124);
  a202=(a63*a123);
  a202=(a185*a202);
  a202=(a44*a202);
  a202=(a122?a202:0);
  a71=(a71+a202);
  a202=(a63/a124);
  a202=(a185*a202);
  a202=(a44*a202);
  a202=(a37*a202);
  a202=(-a202);
  a202=(a122?a202:0);
  a71=(a71+a202);
  a119=(a119/a120);
  a202=(a63*a119);
  a202=(a44*a202);
  a199=casadi_sign(a112);
  a202=(a202*a199);
  a202=(-a202);
  a202=(a100?a202:0);
  a71=(a71+a202);
  a63=(a63/a120);
  a63=(a44*a63);
  a112=casadi_sign(a112);
  a63=(a63*a112);
  a63=(-a63);
  a63=(a100?a63:0);
  a71=(a71+a63);
  a63=(a2*a71);
  a61=(a61+a63);
  a63=(a61/a103);
  a202=(a99*a63);
  a166=(a94*a202);
  a200=(a91*a63);
  a197=(a83*a200);
  a166=(a166-a197);
  a197=(a93*a166);
  a192=(a114*a202);
  a197=(a197-a192);
  a192=(a84*a197);
  a196=(a88*a166);
  a198=(a114*a200);
  a196=(a196+a198);
  a198=(a81*a196);
  a192=(a192+a198);
  a104=(a104+a104);
  a198=(a95/a103);
  a61=(a198*a61);
  a121=(a96/a103);
  a203=(a4*a125);
  a204=(a1*a71);
  a203=(a203+a204);
  a204=(a121*a203);
  a61=(a61+a204);
  a204=(a103+a103);
  a61=(a61/a204);
  a205=(a104*a61);
  a97=(a97+a97);
  a65=(a65/a107);
  a65=(a126?a65:0);
  a79=(a79/a107);
  a65=(a65-a79);
  a65=(a193*a65);
  a90=(a90+a90);
  a65=(a65/a90);
  a79=(a97*a65);
  a206=(a205-a79);
  a207=(a118*a63);
  a203=(a203/a103);
  a208=(a101*a203);
  a207=(a207+a208);
  a206=(a206-a207);
  a206=(a89*a206);
  a192=(a192+a206);
  a206=(a116*a202);
  a208=(a113*a200);
  a209=(a117*a202);
  a208=(a208-a209);
  a209=(a93*a208);
  a206=(a206+a209);
  a79=(a79-a205);
  a79=(a79+a207);
  a207=(a5*a79);
  a206=(a206-a207);
  a207=(a91*a203);
  a205=(a200+a207);
  a209=(a5*a205);
  a206=(a206+a209);
  a209=(a81*a206);
  a192=(a192+a209);
  a209=(a117*a166);
  a210=(a94*a208);
  a209=(a209+a210);
  a209=(a209+a79);
  a209=(a209-a205);
  a209=(a87*a209);
  a192=(a192+a209);
  a207=(a89*a207);
  a192=(a192+a207);
  a207=(a88*a208);
  a200=(a116*a200);
  a207=(a207-a200);
  a200=(a99*a203);
  a202=(a202+a200);
  a209=(a5*a202);
  a207=(a207+a209);
  a86=(a86+a86);
  a65=(a86*a65);
  a102=(a102+a102);
  a61=(a102*a61);
  a209=(a65-a61);
  a63=(a115*a63);
  a203=(a92*a203);
  a63=(a63+a203);
  a209=(a209+a63);
  a203=(a5*a209);
  a207=(a207+a203);
  a203=(a84*a207);
  a192=(a192-a203);
  a192=(a98*a192);
  a110=(a110-a192);
  a192=cos(a6);
  a203=(a84*a196);
  a205=(a81*a197);
  a203=(a203-a205);
  a200=(a89*a200);
  a203=(a203+a200);
  a200=(a84*a206);
  a203=(a203+a200);
  a65=(a65-a61);
  a65=(a65+a63);
  a65=(a89*a65);
  a203=(a203+a65);
  a166=(a113*a166);
  a208=(a83*a208);
  a166=(a166+a208);
  a166=(a166-a202);
  a166=(a166-a209);
  a166=(a87*a166);
  a203=(a203+a166);
  a166=(a81*a207);
  a203=(a203+a166);
  a203=(a192*a203);
  a110=(a110+a203);
  a203=sin(a6);
  a23=(a23/a62);
  a23=(a76?a23:0);
  a35=(a35/a62);
  a23=(a23-a35);
  a23=(a193*a23);
  a60=(a60+a60);
  a23=(a23/a60);
  a35=(a52*a23);
  a166=(a54/a57);
  a209=(a4*a75);
  a18=(a18/a78);
  a20=(a66*a20);
  a73=(a73/a74);
  a202=(a20*a73);
  a202=(a185*a202);
  a202=(a44*a202);
  a202=(a72?a202:0);
  a18=(a18+a202);
  a202=(a20/a74);
  a202=(a185*a202);
  a202=(a44*a202);
  a202=(a37*a202);
  a202=(-a202);
  a202=(a72?a202:0);
  a18=(a18+a202);
  a69=(a69/a70);
  a202=(a20*a69);
  a202=(a44*a202);
  a208=casadi_sign(a67);
  a202=(a202*a208);
  a202=(-a202);
  a202=(a68?a202:0);
  a18=(a18+a202);
  a20=(a20/a70);
  a20=(a44*a20);
  a67=casadi_sign(a67);
  a20=(a20*a67);
  a20=(-a20);
  a20=(a68?a20:0);
  a18=(a18+a20);
  a20=(a1*a18);
  a209=(a209+a20);
  a20=(a166*a209);
  a202=(a57+a57);
  a20=(a20/a202);
  a65=(a52*a20);
  a35=(a35-a65);
  a209=(a209/a57);
  a65=(a49*a209);
  a35=(a35-a65);
  a35=(a5*a35);
  a65=(a51*a209);
  a65=(a5*a65);
  a35=(a35+a65);
  a35=(a203*a35);
  a110=(a110-a35);
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
  a110=(a110+a23);
  a23=sin(a6);
  a181=(a181/a22);
  a181=(a42?a181:0);
  a186=(a186/a22);
  a181=(a181-a186);
  a181=(a193*a181);
  a9=(a9+a9);
  a181=(a181/a9);
  a186=(a11*a181);
  a209=(a13/a16);
  a4=(a4*a41);
  a184=(a184/a46);
  a3=(a28*a3);
  a38=(a38/a40);
  a20=(a3*a38);
  a20=(a185*a20);
  a20=(a44*a20);
  a20=(a36?a20:0);
  a184=(a184+a20);
  a20=(a3/a40);
  a20=(a185*a20);
  a20=(a44*a20);
  a20=(a37*a20);
  a20=(-a20);
  a20=(a36?a20:0);
  a184=(a184+a20);
  a32=(a32/a34);
  a20=(a3*a32);
  a20=(a44*a20);
  a65=casadi_sign(a29);
  a20=(a20*a65);
  a20=(-a20);
  a20=(a31?a20:0);
  a184=(a184+a20);
  a3=(a3/a34);
  a3=(a44*a3);
  a29=casadi_sign(a29);
  a3=(a3*a29);
  a3=(-a3);
  a3=(a31?a3:0);
  a184=(a184+a3);
  a3=(a1*a184);
  a4=(a4+a3);
  a3=(a209*a4);
  a20=(a16+a16);
  a3=(a3/a20);
  a63=(a11*a3);
  a186=(a186-a63);
  a4=(a4/a16);
  a63=(a7*a4);
  a186=(a186-a63);
  a186=(a5*a186);
  a63=(a10*a4);
  a63=(a5*a63);
  a186=(a186+a63);
  a186=(a23*a186);
  a110=(a110-a186);
  a6=cos(a6);
  a8=(a8+a8);
  a181=(a8*a181);
  a15=(a15+a15);
  a3=(a15*a3);
  a181=(a181-a3);
  a3=(a12*a4);
  a181=(a181+a3);
  a181=(a5*a181);
  a4=(a14*a4);
  a4=(a5*a4);
  a181=(a181-a4);
  a181=(a6*a181);
  a110=(a110+a181);
  if (res[1]!=0) res[1][8]=a110;
  a110=(a59*a39);
  a170=(a170/a171);
  a27=(a27*a153);
  a168=(a27*a168);
  a168=(a185*a168);
  a168=(a44*a168);
  a168=(a167?a168:0);
  a170=(a170+a168);
  a169=(a27/a169);
  a169=(a185*a169);
  a169=(a44*a169);
  a169=(a37*a169);
  a169=(-a169);
  a167=(a167?a169:0);
  a170=(a170+a167);
  a148=(a27*a148);
  a148=(a44*a148);
  a148=(a148*a154);
  a148=(-a148);
  a148=(a30?a148:0);
  a170=(a170+a148);
  a27=(a27/a165);
  a27=(a44*a27);
  a27=(a27*a158);
  a27=(-a27);
  a30=(a30?a27:0);
  a170=(a170+a30);
  a30=(a2*a170);
  a110=(a110+a30);
  a30=(a110/a151);
  a27=(a147*a30);
  a158=(a142*a27);
  a165=(a139*a30);
  a148=(a132*a165);
  a158=(a158-a148);
  a148=(a136*a158);
  a154=(a160*a165);
  a148=(a148+a154);
  a154=(a133*a148);
  a167=(a141*a158);
  a160=(a160*a27);
  a167=(a167-a160);
  a160=(a130*a167);
  a154=(a154-a160);
  a39=(a48*a39);
  a160=(a1*a170);
  a39=(a39+a160);
  a151=(a39/a151);
  a147=(a147*a151);
  a160=(a137*a147);
  a154=(a154+a160);
  a160=(a162*a27);
  a169=(a159*a165);
  a168=(a163*a27);
  a169=(a169-a168);
  a141=(a141*a169);
  a160=(a160+a141);
  a0=(a0/a21);
  a19=(a19?a0:0);
  a156=(a156/a21);
  a19=(a19-a156);
  a19=(a193*a19);
  a19=(a19/a138);
  a145=(a145*a19);
  a194=(a194*a110);
  a195=(a195*a39);
  a194=(a194+a195);
  a194=(a194/a188);
  a152=(a152*a194);
  a188=(a145-a152);
  a164=(a164*a30);
  a149=(a149*a151);
  a164=(a164+a149);
  a188=(a188+a164);
  a149=(a5*a188);
  a160=(a160-a149);
  a139=(a139*a151);
  a149=(a165+a139);
  a195=(a5*a149);
  a160=(a160+a195);
  a195=(a133*a160);
  a154=(a154+a195);
  a135=(a135*a19);
  a150=(a150*a194);
  a194=(a135-a150);
  a161=(a161*a30);
  a140=(a140*a151);
  a161=(a161+a140);
  a194=(a194+a161);
  a194=(a137*a194);
  a154=(a154+a194);
  a159=(a159*a158);
  a132=(a132*a169);
  a159=(a159+a132);
  a27=(a27+a147);
  a159=(a159-a27);
  a135=(a135-a150);
  a135=(a135+a161);
  a159=(a159-a135);
  a159=(a87*a159);
  a154=(a154+a159);
  a136=(a136*a169);
  a162=(a162*a165);
  a136=(a136-a162);
  a27=(a5*a27);
  a136=(a136+a27);
  a135=(a5*a135);
  a136=(a136+a135);
  a135=(a130*a136);
  a154=(a154+a135);
  a146=(a146*a154);
  a154=-4.8780487804877992e-01;
  a135=(a154*a173);
  a27=(a173*a135);
  a162=(a154*a172);
  a165=(a172*a162);
  a27=(a27+a165);
  a27=(a174*a27);
  a108=(a108*a27);
  a27=(a173*a162);
  a165=(a172*a135);
  a27=(a27-a165);
  a174=(a174*a27);
  a190=(a190*a174);
  a108=(a108+a190);
  a190=(a133*a167);
  a174=(a130*a148);
  a190=(a190+a174);
  a152=(a152-a145);
  a152=(a152-a164);
  a152=(a137*a152);
  a190=(a190+a152);
  a130=(a130*a160);
  a190=(a190+a130);
  a163=(a163*a158);
  a142=(a142*a169);
  a163=(a163+a142);
  a163=(a163+a188);
  a163=(a163-a149);
  a163=(a87*a163);
  a190=(a190+a163);
  a137=(a137*a139);
  a190=(a190+a137);
  a133=(a133*a136);
  a190=(a190-a133);
  a189=(a189*a190);
  a108=(a108+a189);
  a146=(a146-a108);
  a59=(a59*a125);
  a127=(a127/a128);
  a111=(a111*a105);
  a123=(a111*a123);
  a123=(a185*a123);
  a123=(a44*a123);
  a123=(a122?a123:0);
  a127=(a127+a123);
  a124=(a111/a124);
  a124=(a185*a124);
  a124=(a44*a124);
  a124=(a37*a124);
  a124=(-a124);
  a122=(a122?a124:0);
  a127=(a127+a122);
  a119=(a111*a119);
  a119=(a44*a119);
  a119=(a119*a199);
  a119=(-a119);
  a119=(a100?a119:0);
  a127=(a127+a119);
  a111=(a111/a120);
  a111=(a44*a111);
  a111=(a111*a112);
  a111=(-a111);
  a100=(a100?a111:0);
  a127=(a127+a100);
  a100=(a2*a127);
  a59=(a59+a100);
  a100=(a59/a103);
  a111=(a99*a100);
  a112=(a94*a111);
  a120=(a91*a100);
  a119=(a83*a120);
  a112=(a112-a119);
  a119=(a93*a112);
  a199=(a114*a111);
  a119=(a119-a199);
  a199=(a84*a119);
  a122=(a88*a112);
  a114=(a114*a120);
  a122=(a122+a114);
  a114=(a81*a122);
  a199=(a199+a114);
  a198=(a198*a59);
  a125=(a48*a125);
  a59=(a1*a127);
  a125=(a125+a59);
  a121=(a121*a125);
  a198=(a198+a121);
  a198=(a198/a204);
  a104=(a104*a198);
  a47=(a47/a107);
  a126=(a126?a47:0);
  a109=(a109/a107);
  a126=(a126-a109);
  a126=(a193*a126);
  a126=(a126/a90);
  a97=(a97*a126);
  a90=(a104-a97);
  a118=(a118*a100);
  a125=(a125/a103);
  a101=(a101*a125);
  a118=(a118+a101);
  a90=(a90-a118);
  a90=(a89*a90);
  a199=(a199+a90);
  a90=(a116*a111);
  a101=(a113*a120);
  a103=(a117*a111);
  a101=(a101-a103);
  a93=(a93*a101);
  a90=(a90+a93);
  a97=(a97-a104);
  a97=(a97+a118);
  a118=(a5*a97);
  a90=(a90-a118);
  a91=(a91*a125);
  a118=(a120+a91);
  a104=(a5*a118);
  a90=(a90+a104);
  a104=(a81*a90);
  a199=(a199+a104);
  a117=(a117*a112);
  a94=(a94*a101);
  a117=(a117+a94);
  a117=(a117+a97);
  a117=(a117-a118);
  a117=(a87*a117);
  a199=(a199+a117);
  a91=(a89*a91);
  a199=(a199+a91);
  a88=(a88*a101);
  a116=(a116*a120);
  a88=(a88-a116);
  a99=(a99*a125);
  a111=(a111+a99);
  a116=(a5*a111);
  a88=(a88+a116);
  a86=(a86*a126);
  a102=(a102*a198);
  a198=(a86-a102);
  a115=(a115*a100);
  a92=(a92*a125);
  a115=(a115+a92);
  a198=(a198+a115);
  a92=(a5*a198);
  a88=(a88+a92);
  a92=(a84*a88);
  a199=(a199-a92);
  a98=(a98*a199);
  a146=(a146-a98);
  a98=(a84*a122);
  a199=(a81*a119);
  a98=(a98-a199);
  a99=(a89*a99);
  a98=(a98+a99);
  a84=(a84*a90);
  a98=(a98+a84);
  a86=(a86-a102);
  a86=(a86+a115);
  a89=(a89*a86);
  a98=(a98+a89);
  a113=(a113*a112);
  a83=(a83*a101);
  a113=(a113+a83);
  a113=(a113-a111);
  a113=(a113-a198);
  a113=(a87*a113);
  a98=(a98+a113);
  a81=(a81*a88);
  a98=(a98+a81);
  a192=(a192*a98);
  a146=(a146+a192);
  a26=(a26/a62);
  a76=(a76?a26:0);
  a64=(a64/a62);
  a76=(a76-a64);
  a76=(a193*a76);
  a76=(a76/a60);
  a60=(a52*a76);
  a75=(a48*a75);
  a77=(a77/a78);
  a66=(a66*a58);
  a73=(a66*a73);
  a73=(a185*a73);
  a73=(a44*a73);
  a73=(a72?a73:0);
  a77=(a77+a73);
  a74=(a66/a74);
  a74=(a185*a74);
  a74=(a44*a74);
  a74=(a37*a74);
  a74=(-a74);
  a72=(a72?a74:0);
  a77=(a77+a72);
  a69=(a66*a69);
  a69=(a44*a69);
  a69=(a69*a208);
  a69=(-a69);
  a69=(a68?a69:0);
  a77=(a77+a69);
  a66=(a66/a70);
  a66=(a44*a66);
  a66=(a66*a67);
  a66=(-a66);
  a68=(a68?a66:0);
  a77=(a77+a68);
  a68=(a1*a77);
  a75=(a75+a68);
  a166=(a166*a75);
  a166=(a166/a202);
  a52=(a52*a166);
  a60=(a60-a52);
  a75=(a75/a57);
  a49=(a49*a75);
  a60=(a60-a49);
  a60=(a5*a60);
  a51=(a51*a75);
  a51=(a5*a51);
  a60=(a60+a51);
  a203=(a203*a60);
  a146=(a146-a203);
  a50=(a50*a76);
  a56=(a56*a166);
  a50=(a50-a56);
  a53=(a53*a75);
  a50=(a50+a53);
  a50=(a5*a50);
  a55=(a55*a75);
  a55=(a5*a55);
  a50=(a50-a55);
  a35=(a35*a50);
  a146=(a146+a35);
  a187=(a187/a22);
  a42=(a42?a187:0);
  a25=(a25/a22);
  a42=(a42-a25);
  a193=(a193*a42);
  a193=(a193/a9);
  a9=(a11*a193);
  a48=(a48*a41);
  a43=(a43/a46);
  a28=(a28*a17);
  a38=(a28*a38);
  a38=(a185*a38);
  a38=(a44*a38);
  a38=(a36?a38:0);
  a43=(a43+a38);
  a40=(a28/a40);
  a185=(a185*a40);
  a185=(a44*a185);
  a37=(a37*a185);
  a37=(-a37);
  a36=(a36?a37:0);
  a43=(a43+a36);
  a32=(a28*a32);
  a32=(a44*a32);
  a32=(a32*a65);
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
  a146=(a146-a23);
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
  a146=(a146+a6);
  if (res[1]!=0) res[1][9]=a146;
  a146=cos(a80);
  a6=(a180*a33);
  a8=(a2*a33);
  a6=(a6-a8);
  a8=(a178*a6);
  a5=(a175*a33);
  a8=(a8-a5);
  a5=(a157*a176);
  a8=(a8+a5);
  a5=(a177*a155);
  a8=(a8-a5);
  a8=(a146*a8);
  a5=sin(a80);
  a157=(a157*a183);
  a14=(a177*a33);
  a157=(a157-a14);
  a14=(a2*a155);
  a48=(a180*a155);
  a14=(a14-a48);
  a48=(a178*a14);
  a157=(a157+a48);
  a48=(a175*a155);
  a157=(a157+a48);
  a157=(a5*a157);
  a8=(a8-a157);
  a157=sin(a80);
  a48=(a134*a129);
  a12=(a131*a45);
  a48=(a48+a12);
  a12=(a131*a191);
  a48=(a48+a12);
  a12=(a134*a201);
  a48=(a48-a12);
  a48=(a157*a48);
  a8=(a8-a48);
  a48=cos(a80);
  a45=(a134*a45);
  a129=(a131*a129);
  a45=(a45-a129);
  a191=(a134*a191);
  a45=(a45+a191);
  a201=(a131*a201);
  a45=(a45+a201);
  a45=(a48*a45);
  a8=(a8+a45);
  a45=sin(a80);
  a201=(a85*a197);
  a191=(a82*a196);
  a201=(a201+a191);
  a191=(a82*a206);
  a201=(a201+a191);
  a191=(a85*a207);
  a201=(a201-a191);
  a201=(a45*a201);
  a8=(a8-a201);
  a80=cos(a80);
  a196=(a85*a196);
  a197=(a82*a197);
  a196=(a196-a197);
  a206=(a85*a206);
  a196=(a196+a206);
  a207=(a82*a207);
  a196=(a196+a207);
  a196=(a80*a196);
  a8=(a8+a196);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a180*a135);
  a196=(a2*a135);
  a8=(a8-a196);
  a196=(a178*a8);
  a207=(a175*a135);
  a196=(a196-a207);
  a176=(a154*a176);
  a196=(a196+a176);
  a176=(a177*a162);
  a196=(a196-a176);
  a146=(a146*a196);
  a154=(a154*a183);
  a177=(a177*a135);
  a154=(a154-a177);
  a2=(a2*a162);
  a180=(a180*a162);
  a2=(a2-a180);
  a178=(a178*a2);
  a154=(a154+a178);
  a175=(a175*a162);
  a154=(a154+a175);
  a5=(a5*a154);
  a146=(a146-a5);
  a5=(a134*a167);
  a154=(a131*a148);
  a5=(a5+a154);
  a154=(a131*a160);
  a5=(a5+a154);
  a154=(a134*a136);
  a5=(a5-a154);
  a157=(a157*a5);
  a146=(a146-a157);
  a148=(a134*a148);
  a167=(a131*a167);
  a148=(a148-a167);
  a134=(a134*a160);
  a148=(a148+a134);
  a131=(a131*a136);
  a148=(a148+a131);
  a48=(a48*a148);
  a146=(a146+a48);
  a48=(a85*a119);
  a148=(a82*a122);
  a48=(a48+a148);
  a148=(a82*a90);
  a48=(a48+a148);
  a148=(a85*a88);
  a48=(a48-a148);
  a45=(a45*a48);
  a146=(a146-a45);
  a122=(a85*a122);
  a119=(a82*a119);
  a122=(a122-a119);
  a85=(a85*a90);
  a122=(a122+a85);
  a82=(a82*a88);
  a122=(a122+a82);
  a80=(a80*a122);
  a146=(a146+a80);
  if (res[1]!=0) res[1][11]=a146;
  a146=-1.;
  if (res[1]!=0) res[1][12]=a146;
  a80=(a182*a33);
  a122=(a179*a155);
  a80=(a80-a122);
  a6=(a172*a6);
  a14=(a173*a14);
  a6=(a6+a14);
  a6=(a87*a6);
  a6=(a80+a6);
  a14=(a144*a106);
  a6=(a6+a14);
  a14=(a96*a71);
  a6=(a6+a14);
  a18=(a54*a18);
  a6=(a6+a18);
  a184=(a13*a184);
  a6=(a6+a184);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a182*a135);
  a184=(a179*a162);
  a6=(a6-a184);
  a172=(a172*a8);
  a173=(a173*a2);
  a172=(a172+a173);
  a87=(a87*a172);
  a87=(a6+a87);
  a144=(a144*a170);
  a87=(a87+a144);
  a96=(a96*a127);
  a87=(a87+a96);
  a54=(a54*a77);
  a87=(a87+a54);
  a13=(a13*a43);
  a87=(a87+a13);
  if (res[1]!=0) res[1][14]=a87;
  if (res[1]!=0) res[1][15]=a146;
  a33=(a182*a33);
  a80=(a80-a33);
  a155=(a179*a155);
  a80=(a80+a155);
  a106=(a143*a106);
  a80=(a80+a106);
  a71=(a95*a71);
  a80=(a80+a71);
  if (res[1]!=0) res[1][16]=a80;
  a182=(a182*a135);
  a6=(a6-a182);
  a179=(a179*a162);
  a6=(a6+a179);
  a143=(a143*a170);
  a6=(a6+a143);
  a95=(a95*a127);
  a6=(a6+a95);
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

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14112775_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
