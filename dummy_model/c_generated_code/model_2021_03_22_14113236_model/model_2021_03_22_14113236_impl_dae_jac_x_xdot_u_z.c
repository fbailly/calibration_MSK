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
  #define CASADI_PREFIX(ID) model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_ ## ID
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

/* model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8x8,18nz],o1[8x8,8nz],o2[8x4,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a215, a216, a217, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=4.0000000000000001e-02;
  a1=5.0000000000000000e-01;
  a2=arg[0]? arg[0][4] : 0;
  a3=sin(a2);
  a4=(a1*a3);
  a5=-5.0000000000000000e-01;
  a6=(a4+a5);
  a7=casadi_sq(a6);
  a8=cos(a2);
  a9=(a1*a8);
  a10=casadi_sq(a9);
  a7=(a7+a10);
  a7=sqrt(a7);
  a10=(a7-a0);
  a11=8.7758256189037276e-01;
  a10=(a10/a11);
  a12=(a0<a10);
  a13=10.;
  a14=6.7836549063042314e-03;
  a15=3.0000000261848010e+02;
  a16=3.9024390243902418e-01;
  a17=(a4+a5);
  a18=(a1*a8);
  a19=(a17*a18);
  a3=(a1*a3);
  a20=(a9*a3);
  a19=(a19-a20);
  a4=(a4+a5);
  a5=casadi_sq(a4);
  a20=casadi_sq(a9);
  a5=(a5+a20);
  a5=sqrt(a5);
  a19=(a19/a5);
  a20=(a16*a19);
  a20=(a15*a20);
  a21=(a14*a20);
  a22=arg[0]? arg[0][0] : 0;
  a23=(a10/a22);
  a24=1.;
  a25=(a23-a24);
  a25=(a13*a25);
  a25=exp(a25);
  a21=(a21*a25);
  a21=(a13*a21);
  a23=(a23/a22);
  a26=(a21*a23);
  a26=(-a26);
  a26=(a12?a26:0);
  a27=arg[0]? arg[0][6] : 0;
  a28=(a19*a27);
  a29=(a13*a22);
  a30=(a28/a29);
  a31=(a30/a29);
  a32=1.0000000000000001e-01;
  a33=(a32*a20);
  a34=(a31*a33);
  a34=(a13*a34);
  a26=(a26-a34);
  a10=(a10/a22);
  a34=(a10/a22);
  a10=(a10-a24);
  a35=(a10+a10);
  a36=2.2222222222222223e+00;
  a10=casadi_sq(a10);
  a37=4.5000000000000001e-01;
  a10=(a10/a37);
  a10=(-a10);
  a10=exp(a10);
  a38=arg[2]? arg[2][0] : 0;
  a39=0.;
  a40=(a28<=a39);
  a41=fabs(a28);
  a41=(a41/a13);
  a41=(a24-a41);
  a42=fabs(a28);
  a42=(a42/a13);
  a42=(a24+a42);
  a41=(a41/a42);
  a43=(a40?a41:0);
  a44=(!a40);
  a45=1.3300000000000001e+00;
  a46=(a45*a28);
  a46=(a46/a13);
  a47=-8.2500000000000004e-02;
  a46=(a46/a47);
  a46=(a24-a46);
  a48=(a28/a13);
  a48=(a48/a47);
  a48=(a24-a48);
  a46=(a46/a48);
  a49=(a44?a46:0);
  a43=(a43+a49);
  a49=(a43*a20);
  a49=(a38*a49);
  a49=(a10*a49);
  a49=(a36*a49);
  a49=(a35*a49);
  a50=(a34*a49);
  a26=(a26+a50);
  if (res[0]!=0) res[0][0]=a26;
  a26=-3.9024390243902396e-01;
  a50=(a26*a19);
  a50=(a15*a50);
  a51=(a14*a50);
  a51=(a51*a25);
  a51=(a13*a51);
  a23=(a51*a23);
  a23=(-a23);
  a23=(a12?a23:0);
  a52=(a32*a50);
  a31=(a31*a52);
  a31=(a13*a31);
  a23=(a23-a31);
  a31=(a43*a50);
  a31=(a38*a31);
  a31=(a10*a31);
  a31=(a36*a31);
  a35=(a35*a31);
  a34=(a34*a35);
  a23=(a23+a34);
  if (res[0]!=0) res[0][1]=a23;
  a23=sin(a2);
  a34=(a1*a23);
  a31=(a34+a1);
  a53=casadi_sq(a31);
  a54=cos(a2);
  a55=(a1*a54);
  a56=casadi_sq(a55);
  a53=(a53+a56);
  a53=sqrt(a53);
  a56=(a53-a0);
  a56=(a56/a11);
  a57=(a0<a56);
  a58=3.0000000115586607e+02;
  a59=(a34+a1);
  a60=(a1*a54);
  a61=(a59*a60);
  a23=(a1*a23);
  a62=(a55*a23);
  a61=(a61-a62);
  a34=(a34+a1);
  a62=casadi_sq(a34);
  a63=casadi_sq(a55);
  a62=(a62+a63);
  a62=sqrt(a62);
  a61=(a61/a62);
  a63=(a16*a61);
  a63=(a58*a63);
  a64=(a14*a63);
  a65=arg[0]? arg[0][1] : 0;
  a66=(a56/a65);
  a67=(a66-a24);
  a67=(a13*a67);
  a67=exp(a67);
  a64=(a64*a67);
  a64=(a13*a64);
  a66=(a66/a65);
  a68=(a64*a66);
  a68=(-a68);
  a68=(a57?a68:0);
  a69=(a61*a27);
  a70=(a13*a65);
  a71=(a69/a70);
  a72=(a71/a70);
  a73=(a32*a63);
  a74=(a72*a73);
  a74=(a13*a74);
  a68=(a68-a74);
  a56=(a56/a65);
  a74=(a56/a65);
  a56=(a56-a24);
  a75=(a56+a56);
  a56=casadi_sq(a56);
  a56=(a56/a37);
  a56=(-a56);
  a56=exp(a56);
  a76=arg[2]? arg[2][1] : 0;
  a77=(a69<=a39);
  a78=fabs(a69);
  a78=(a78/a13);
  a78=(a24-a78);
  a79=fabs(a69);
  a79=(a79/a13);
  a79=(a24+a79);
  a78=(a78/a79);
  a80=(a77?a78:0);
  a81=(!a77);
  a82=(a45*a69);
  a82=(a82/a13);
  a82=(a82/a47);
  a82=(a24-a82);
  a83=(a69/a13);
  a83=(a83/a47);
  a83=(a24-a83);
  a82=(a82/a83);
  a84=(a81?a82:0);
  a80=(a80+a84);
  a84=(a80*a63);
  a84=(a76*a84);
  a84=(a56*a84);
  a84=(a36*a84);
  a84=(a75*a84);
  a85=(a74*a84);
  a68=(a68+a85);
  if (res[0]!=0) res[0][2]=a68;
  a68=(a26*a61);
  a68=(a58*a68);
  a85=(a14*a68);
  a85=(a85*a67);
  a85=(a13*a85);
  a66=(a85*a66);
  a66=(-a66);
  a66=(a57?a66:0);
  a86=(a32*a68);
  a72=(a72*a86);
  a72=(a13*a72);
  a66=(a66-a72);
  a72=(a80*a68);
  a72=(a76*a72);
  a72=(a56*a72);
  a72=(a36*a72);
  a75=(a75*a72);
  a74=(a74*a75);
  a66=(a66+a74);
  if (res[0]!=0) res[0][3]=a66;
  a66=arg[0]? arg[0][5] : 0;
  a74=sin(a66);
  a72=sin(a2);
  a87=(a74*a72);
  a88=cos(a66);
  a89=cos(a2);
  a90=(a88*a89);
  a87=(a87-a90);
  a90=(a1*a87);
  a91=1.2500000000000000e+00;
  a92=(a91*a72);
  a90=(a90-a92);
  a93=7.5000000000000000e-01;
  a94=(a93*a72);
  a95=(a90+a94);
  a96=casadi_sq(a95);
  a97=(a91*a89);
  a98=(a88*a72);
  a99=(a74*a89);
  a98=(a98+a99);
  a99=(a1*a98);
  a99=(a97-a99);
  a100=(a93*a89);
  a101=(a99-a100);
  a102=casadi_sq(a101);
  a96=(a96+a102);
  a96=sqrt(a96);
  a102=(a96-a0);
  a102=(a102/a11);
  a103=(a0<a102);
  a104=6.8289955952840182e+02;
  a105=-3.9024390243902440e-01;
  a106=(a90+a94);
  a107=(a74*a89);
  a108=(a88*a72);
  a107=(a107+a108);
  a108=(a87*a92);
  a109=(a98*a97);
  a108=(a108+a109);
  a109=(a107*a108);
  a110=(a107*a92);
  a111=(a88*a89);
  a112=(a74*a72);
  a111=(a111-a112);
  a112=(a111*a97);
  a110=(a110+a112);
  a112=(a87*a110);
  a109=(a109-a112);
  a112=(a1*a98);
  a112=(a97-a112);
  a109=(a109-a112);
  a113=(a106*a109);
  a114=(a99-a100);
  a115=(a98*a110);
  a116=(a111*a108);
  a115=(a115-a116);
  a116=(a1*a87);
  a116=(a116-a92);
  a115=(a115+a116);
  a117=(a114*a115);
  a113=(a113+a117);
  a90=(a90+a94);
  a94=casadi_sq(a90);
  a99=(a99-a100);
  a100=casadi_sq(a99);
  a94=(a94+a100);
  a94=sqrt(a94);
  a113=(a113/a94);
  a100=(a105*a113);
  a117=(a93*a89);
  a117=(a117-a112);
  a112=(a106*a117);
  a118=(a93*a72);
  a116=(a116+a118);
  a118=(a114*a116);
  a112=(a112+a118);
  a112=(a112/a94);
  a118=(a16*a112);
  a100=(a100+a118);
  a100=(a104*a100);
  a118=(a14*a100);
  a119=arg[0]? arg[0][2] : 0;
  a120=(a102/a119);
  a121=(a120-a24);
  a121=(a13*a121);
  a121=exp(a121);
  a118=(a118*a121);
  a118=(a13*a118);
  a120=(a120/a119);
  a122=(a118*a120);
  a122=(-a122);
  a122=(a103?a122:0);
  a123=(a112*a27);
  a124=arg[0]? arg[0][7] : 0;
  a125=(a113*a124);
  a123=(a123+a125);
  a125=(a13*a119);
  a126=(a123/a125);
  a127=(a126/a125);
  a128=(a32*a100);
  a129=(a127*a128);
  a129=(a13*a129);
  a122=(a122-a129);
  a102=(a102/a119);
  a129=(a102/a119);
  a102=(a102-a24);
  a130=(a102+a102);
  a102=casadi_sq(a102);
  a102=(a102/a37);
  a102=(-a102);
  a102=exp(a102);
  a131=arg[2]? arg[2][2] : 0;
  a132=(a123<=a39);
  a133=fabs(a123);
  a133=(a133/a13);
  a133=(a24-a133);
  a134=fabs(a123);
  a134=(a134/a13);
  a134=(a24+a134);
  a133=(a133/a134);
  a135=(a132?a133:0);
  a136=(!a132);
  a137=(a45*a123);
  a137=(a137/a13);
  a137=(a137/a47);
  a137=(a24-a137);
  a138=(a123/a13);
  a138=(a138/a47);
  a138=(a24-a138);
  a137=(a137/a138);
  a139=(a136?a137:0);
  a135=(a135+a139);
  a139=(a135*a100);
  a139=(a131*a139);
  a139=(a102*a139);
  a139=(a36*a139);
  a139=(a130*a139);
  a140=(a129*a139);
  a122=(a122+a140);
  if (res[0]!=0) res[0][4]=a122;
  a122=1.3902439024390245e+00;
  a140=(a122*a113);
  a141=(a26*a112);
  a140=(a140+a141);
  a140=(a104*a140);
  a141=(a14*a140);
  a141=(a141*a121);
  a141=(a13*a141);
  a120=(a141*a120);
  a120=(-a120);
  a120=(a103?a120:0);
  a142=(a32*a140);
  a127=(a127*a142);
  a127=(a13*a127);
  a120=(a120-a127);
  a127=(a135*a140);
  a127=(a131*a127);
  a127=(a102*a127);
  a127=(a36*a127);
  a130=(a130*a127);
  a129=(a129*a130);
  a120=(a120+a129);
  if (res[0]!=0) res[0][5]=a120;
  a120=sin(a66);
  a129=sin(a2);
  a127=(a120*a129);
  a143=cos(a66);
  a144=cos(a2);
  a145=(a143*a144);
  a127=(a127-a145);
  a145=(a1*a127);
  a146=(a91*a129);
  a145=(a145-a146);
  a147=1.7500000000000000e+00;
  a148=(a147*a129);
  a149=(a145+a148);
  a150=casadi_sq(a149);
  a151=(a91*a144);
  a152=(a143*a129);
  a153=(a120*a144);
  a152=(a152+a153);
  a153=(a1*a152);
  a153=(a151-a153);
  a154=(a147*a144);
  a155=(a153-a154);
  a156=casadi_sq(a155);
  a150=(a150+a156);
  a150=sqrt(a150);
  a156=(a150-a0);
  a156=(a156/a11);
  a0=(a0<a156);
  a11=7.6447495683762986e+02;
  a157=(a145+a148);
  a158=(a120*a144);
  a159=(a143*a129);
  a158=(a158+a159);
  a159=(a127*a146);
  a160=(a152*a151);
  a159=(a159+a160);
  a160=(a158*a159);
  a161=(a158*a146);
  a162=(a143*a144);
  a163=(a120*a129);
  a162=(a162-a163);
  a163=(a162*a151);
  a161=(a161+a163);
  a163=(a127*a161);
  a160=(a160-a163);
  a163=(a1*a152);
  a163=(a151-a163);
  a160=(a160-a163);
  a164=(a157*a160);
  a165=(a153-a154);
  a166=(a152*a161);
  a167=(a162*a159);
  a166=(a166-a167);
  a167=(a1*a127);
  a167=(a167-a146);
  a166=(a166+a167);
  a168=(a165*a166);
  a164=(a164+a168);
  a145=(a145+a148);
  a148=casadi_sq(a145);
  a153=(a153-a154);
  a154=casadi_sq(a153);
  a148=(a148+a154);
  a148=sqrt(a148);
  a164=(a164/a148);
  a154=(a105*a164);
  a168=(a147*a144);
  a168=(a168-a163);
  a163=(a157*a168);
  a169=(a147*a129);
  a167=(a167+a169);
  a169=(a165*a167);
  a163=(a163+a169);
  a163=(a163/a148);
  a169=(a16*a163);
  a154=(a154+a169);
  a154=(a11*a154);
  a169=(a14*a154);
  a170=arg[0]? arg[0][3] : 0;
  a171=(a156/a170);
  a172=(a171-a24);
  a172=(a13*a172);
  a172=exp(a172);
  a169=(a169*a172);
  a169=(a13*a169);
  a171=(a171/a170);
  a173=(a169*a171);
  a173=(-a173);
  a173=(a0?a173:0);
  a174=(a163*a27);
  a175=(a164*a124);
  a174=(a174+a175);
  a175=(a13*a170);
  a176=(a174/a175);
  a177=(a176/a175);
  a178=(a32*a154);
  a179=(a177*a178);
  a179=(a13*a179);
  a173=(a173-a179);
  a156=(a156/a170);
  a179=(a156/a170);
  a156=(a156-a24);
  a180=(a156+a156);
  a156=casadi_sq(a156);
  a156=(a156/a37);
  a156=(-a156);
  a156=exp(a156);
  a37=arg[2]? arg[2][3] : 0;
  a39=(a174<=a39);
  a181=fabs(a174);
  a181=(a181/a13);
  a181=(a24-a181);
  a182=fabs(a174);
  a182=(a182/a13);
  a182=(a24+a182);
  a181=(a181/a182);
  a183=(a39?a181:0);
  a184=(!a39);
  a185=(a45*a174);
  a185=(a185/a13);
  a185=(a185/a47);
  a185=(a24-a185);
  a186=(a174/a13);
  a186=(a186/a47);
  a186=(a24-a186);
  a185=(a185/a186);
  a47=(a184?a185:0);
  a183=(a183+a47);
  a47=(a183*a154);
  a47=(a37*a47);
  a47=(a156*a47);
  a47=(a36*a47);
  a47=(a180*a47);
  a187=(a179*a47);
  a173=(a173+a187);
  if (res[0]!=0) res[0][6]=a173;
  a173=(a122*a164);
  a187=(a26*a163);
  a173=(a173+a187);
  a173=(a11*a173);
  a14=(a14*a173);
  a14=(a14*a172);
  a14=(a13*a14);
  a171=(a14*a171);
  a171=(-a171);
  a171=(a0?a171:0);
  a187=(a32*a173);
  a177=(a177*a187);
  a13=(a13*a177);
  a171=(a171-a13);
  a13=(a183*a173);
  a13=(a37*a13);
  a13=(a156*a13);
  a36=(a36*a13);
  a180=(a180*a36);
  a179=(a179*a180);
  a171=(a171+a179);
  if (res[0]!=0) res[0][7]=a171;
  a171=cos(a2);
  a37=(a37*a156);
  a179=(a37*a183);
  a172=(a172-a24);
  a36=1.4741315910257660e+02;
  a172=(a172/a36);
  a172=(a0?a172:0);
  a179=(a179+a172);
  a176=(a32*a176);
  a179=(a179+a176);
  a179=(a11*a179);
  a176=(a105*a179);
  a178=(a178/a175);
  a172=-1.2121212121212121e+01;
  a154=(a37*a154);
  a185=(a185/a186);
  a13=(a154*a185);
  a13=(a172*a13);
  a13=(a32*a13);
  a13=(a184?a13:0);
  a178=(a178+a13);
  a13=(a154/a186);
  a13=(a172*a13);
  a13=(a32*a13);
  a13=(a45*a13);
  a13=(-a13);
  a13=(a184?a13:0);
  a178=(a178+a13);
  a181=(a181/a182);
  a13=(a154*a181);
  a13=(a32*a13);
  a177=casadi_sign(a174);
  a13=(a13*a177);
  a13=(-a13);
  a13=(a39?a13:0);
  a178=(a178+a13);
  a154=(a154/a182);
  a154=(a32*a154);
  a174=casadi_sign(a174);
  a154=(a154*a174);
  a154=(-a154);
  a154=(a39?a154:0);
  a178=(a178+a154);
  a154=(a124*a178);
  a176=(a176+a154);
  a154=(a176/a148);
  a13=(a165*a154);
  a188=(a152*a13);
  a189=(a157*a154);
  a190=(a127*a189);
  a188=(a188-a190);
  a190=(a146*a188);
  a191=(a159*a189);
  a190=(a190+a191);
  a191=(a143*a190);
  a192=(a151*a188);
  a193=(a159*a13);
  a192=(a192-a193);
  a193=(a120*a192);
  a191=(a191-a193);
  a193=(a16*a179);
  a194=(a27*a178);
  a193=(a193+a194);
  a194=(a193/a148);
  a195=(a165*a194);
  a196=(a147*a195);
  a191=(a191+a196);
  a196=(a161*a13);
  a197=(a158*a189);
  a198=(a162*a13);
  a197=(a197-a198);
  a198=(a151*a197);
  a196=(a196+a198);
  a155=(a155+a155);
  a198=1.1394939273245490e+00;
  a169=(a169/a170);
  a169=(a0?a169:0);
  a47=(a47/a170);
  a169=(a169-a47);
  a169=(a198*a169);
  a150=(a150+a150);
  a169=(a169/a150);
  a47=(a155*a169);
  a153=(a153+a153);
  a199=(a164/a148);
  a176=(a199*a176);
  a200=(a163/a148);
  a193=(a200*a193);
  a176=(a176+a193);
  a193=(a148+a148);
  a176=(a176/a193);
  a201=(a153*a176);
  a202=(a47-a201);
  a203=(a166*a154);
  a204=(a167*a194);
  a203=(a203+a204);
  a202=(a202+a203);
  a204=(a1*a202);
  a196=(a196-a204);
  a204=(a157*a194);
  a205=(a189+a204);
  a206=(a1*a205);
  a196=(a196+a206);
  a206=(a143*a196);
  a191=(a191+a206);
  a149=(a149+a149);
  a169=(a149*a169);
  a145=(a145+a145);
  a176=(a145*a176);
  a206=(a169-a176);
  a154=(a160*a154);
  a194=(a168*a194);
  a154=(a154+a194);
  a206=(a206+a154);
  a206=(a147*a206);
  a191=(a191+a206);
  a206=(a158*a188);
  a194=(a127*a197);
  a206=(a206+a194);
  a13=(a13+a195);
  a206=(a206-a13);
  a169=(a169-a176);
  a169=(a169+a154);
  a206=(a206-a169);
  a206=(a91*a206);
  a191=(a191+a206);
  a206=(a146*a197);
  a189=(a161*a189);
  a206=(a206-a189);
  a13=(a1*a13);
  a206=(a206+a13);
  a169=(a1*a169);
  a206=(a206+a169);
  a169=(a120*a206);
  a191=(a191+a169);
  a191=(a171*a191);
  a169=cos(a2);
  a13=9.8100000000000005e+00;
  a189=cos(a66);
  a154=4.8780487804878025e-01;
  a176=(a154*a189);
  a195=(a189*a176);
  a194=sin(a66);
  a207=(a154*a194);
  a208=(a194*a207);
  a195=(a195+a208);
  a195=(a13*a195);
  a195=(a169*a195);
  a208=sin(a2);
  a209=(a189*a207);
  a210=(a194*a176);
  a209=(a209-a210);
  a209=(a13*a209);
  a209=(a208*a209);
  a195=(a195+a209);
  a209=sin(a2);
  a210=(a143*a192);
  a211=(a120*a190);
  a210=(a210+a211);
  a201=(a201-a47);
  a201=(a201-a203);
  a201=(a147*a201);
  a210=(a210+a201);
  a201=(a120*a196);
  a210=(a210+a201);
  a188=(a162*a188);
  a197=(a152*a197);
  a188=(a188+a197);
  a188=(a188+a202);
  a188=(a188-a205);
  a188=(a91*a188);
  a210=(a210+a188);
  a204=(a147*a204);
  a210=(a210+a204);
  a204=(a143*a206);
  a210=(a210-a204);
  a210=(a209*a210);
  a195=(a195+a210);
  a191=(a191-a195);
  a195=sin(a2);
  a131=(a131*a102);
  a210=(a131*a135);
  a121=(a121-a24);
  a121=(a121/a36);
  a121=(a103?a121:0);
  a210=(a210+a121);
  a126=(a32*a126);
  a210=(a210+a126);
  a210=(a104*a210);
  a105=(a105*a210);
  a128=(a128/a125);
  a100=(a131*a100);
  a137=(a137/a138);
  a126=(a100*a137);
  a126=(a172*a126);
  a126=(a32*a126);
  a126=(a136?a126:0);
  a128=(a128+a126);
  a126=(a100/a138);
  a126=(a172*a126);
  a126=(a32*a126);
  a126=(a45*a126);
  a126=(-a126);
  a126=(a136?a126:0);
  a128=(a128+a126);
  a133=(a133/a134);
  a126=(a100*a133);
  a126=(a32*a126);
  a121=casadi_sign(a123);
  a126=(a126*a121);
  a126=(-a126);
  a126=(a132?a126:0);
  a128=(a128+a126);
  a100=(a100/a134);
  a100=(a32*a100);
  a123=casadi_sign(a123);
  a100=(a100*a123);
  a100=(-a100);
  a100=(a132?a100:0);
  a128=(a128+a100);
  a100=(a124*a128);
  a105=(a105+a100);
  a100=(a105/a94);
  a126=(a114*a100);
  a204=(a98*a126);
  a188=(a106*a100);
  a205=(a87*a188);
  a204=(a204-a205);
  a205=(a97*a204);
  a202=(a108*a126);
  a205=(a205-a202);
  a202=(a88*a205);
  a197=(a92*a204);
  a201=(a108*a188);
  a197=(a197+a201);
  a201=(a74*a197);
  a202=(a202+a201);
  a99=(a99+a99);
  a201=(a113/a94);
  a105=(a201*a105);
  a203=(a112/a94);
  a47=(a16*a210);
  a211=(a27*a128);
  a47=(a47+a211);
  a211=(a203*a47);
  a105=(a105+a211);
  a211=(a94+a94);
  a105=(a105/a211);
  a212=(a99*a105);
  a101=(a101+a101);
  a118=(a118/a119);
  a118=(a103?a118:0);
  a139=(a139/a119);
  a118=(a118-a139);
  a118=(a198*a118);
  a96=(a96+a96);
  a118=(a118/a96);
  a139=(a101*a118);
  a213=(a212-a139);
  a214=(a115*a100);
  a47=(a47/a94);
  a215=(a116*a47);
  a214=(a214+a215);
  a213=(a213-a214);
  a213=(a93*a213);
  a202=(a202+a213);
  a213=(a110*a126);
  a215=(a107*a188);
  a216=(a111*a126);
  a215=(a215-a216);
  a216=(a97*a215);
  a213=(a213+a216);
  a139=(a139-a212);
  a139=(a139+a214);
  a214=(a1*a139);
  a213=(a213-a214);
  a214=(a106*a47);
  a212=(a188+a214);
  a216=(a1*a212);
  a213=(a213+a216);
  a216=(a74*a213);
  a202=(a202+a216);
  a216=(a111*a204);
  a217=(a98*a215);
  a216=(a216+a217);
  a216=(a216+a139);
  a216=(a216-a212);
  a216=(a91*a216);
  a202=(a202+a216);
  a214=(a93*a214);
  a202=(a202+a214);
  a214=(a92*a215);
  a188=(a110*a188);
  a214=(a214-a188);
  a188=(a114*a47);
  a126=(a126+a188);
  a216=(a1*a126);
  a214=(a214+a216);
  a95=(a95+a95);
  a118=(a95*a118);
  a90=(a90+a90);
  a105=(a90*a105);
  a216=(a118-a105);
  a100=(a109*a100);
  a47=(a117*a47);
  a100=(a100+a47);
  a216=(a216+a100);
  a47=(a1*a216);
  a214=(a214+a47);
  a47=(a88*a214);
  a202=(a202-a47);
  a202=(a195*a202);
  a191=(a191-a202);
  a202=cos(a2);
  a47=(a88*a197);
  a212=(a74*a205);
  a47=(a47-a212);
  a188=(a93*a188);
  a47=(a47+a188);
  a188=(a88*a213);
  a47=(a47+a188);
  a118=(a118-a105);
  a118=(a118+a100);
  a118=(a93*a118);
  a47=(a47+a118);
  a204=(a107*a204);
  a215=(a87*a215);
  a204=(a204+a215);
  a204=(a204-a126);
  a204=(a204-a216);
  a204=(a91*a204);
  a47=(a47+a204);
  a204=(a74*a214);
  a47=(a47+a204);
  a47=(a202*a47);
  a191=(a191+a47);
  a47=sin(a2);
  a64=(a64/a65);
  a64=(a57?a64:0);
  a84=(a84/a65);
  a64=(a64-a84);
  a64=(a198*a64);
  a53=(a53+a53);
  a64=(a64/a53);
  a84=(a54*a64);
  a204=(a61/a62);
  a76=(a76*a56);
  a216=(a76*a80);
  a67=(a67-a24);
  a67=(a67/a36);
  a67=(a57?a67:0);
  a216=(a216+a67);
  a71=(a32*a71);
  a216=(a216+a71);
  a216=(a58*a216);
  a71=(a16*a216);
  a73=(a73/a70);
  a63=(a76*a63);
  a82=(a82/a83);
  a67=(a63*a82);
  a67=(a172*a67);
  a67=(a32*a67);
  a67=(a81?a67:0);
  a73=(a73+a67);
  a67=(a63/a83);
  a67=(a172*a67);
  a67=(a32*a67);
  a67=(a45*a67);
  a67=(-a67);
  a67=(a81?a67:0);
  a73=(a73+a67);
  a78=(a78/a79);
  a67=(a63*a78);
  a67=(a32*a67);
  a126=casadi_sign(a69);
  a67=(a67*a126);
  a67=(-a67);
  a67=(a77?a67:0);
  a73=(a73+a67);
  a63=(a63/a79);
  a63=(a32*a63);
  a69=casadi_sign(a69);
  a63=(a63*a69);
  a63=(-a63);
  a63=(a77?a63:0);
  a73=(a73+a63);
  a63=(a27*a73);
  a71=(a71+a63);
  a63=(a204*a71);
  a67=(a62+a62);
  a63=(a63/a67);
  a215=(a54*a63);
  a84=(a84-a215);
  a71=(a71/a62);
  a215=(a23*a71);
  a84=(a84-a215);
  a84=(a1*a84);
  a215=(a59*a71);
  a215=(a1*a215);
  a84=(a84+a215);
  a84=(a47*a84);
  a191=(a191-a84);
  a84=cos(a2);
  a31=(a31+a31);
  a64=(a31*a64);
  a34=(a34+a34);
  a63=(a34*a63);
  a64=(a64-a63);
  a63=(a60*a71);
  a64=(a64+a63);
  a64=(a1*a64);
  a71=(a55*a71);
  a71=(a1*a71);
  a64=(a64-a71);
  a64=(a84*a64);
  a191=(a191+a64);
  a64=sin(a2);
  a21=(a21/a22);
  a21=(a12?a21:0);
  a49=(a49/a22);
  a21=(a21-a49);
  a21=(a198*a21);
  a7=(a7+a7);
  a21=(a21/a7);
  a49=(a8*a21);
  a71=(a19/a5);
  a38=(a38*a10);
  a63=(a38*a43);
  a25=(a25-a24);
  a25=(a25/a36);
  a25=(a12?a25:0);
  a63=(a63+a25);
  a30=(a32*a30);
  a63=(a63+a30);
  a63=(a15*a63);
  a16=(a16*a63);
  a33=(a33/a29);
  a20=(a38*a20);
  a46=(a46/a48);
  a30=(a20*a46);
  a30=(a172*a30);
  a30=(a32*a30);
  a30=(a44?a30:0);
  a33=(a33+a30);
  a30=(a20/a48);
  a30=(a172*a30);
  a30=(a32*a30);
  a30=(a45*a30);
  a30=(-a30);
  a30=(a44?a30:0);
  a33=(a33+a30);
  a41=(a41/a42);
  a30=(a20*a41);
  a30=(a32*a30);
  a25=casadi_sign(a28);
  a30=(a30*a25);
  a30=(-a30);
  a30=(a40?a30:0);
  a33=(a33+a30);
  a20=(a20/a42);
  a20=(a32*a20);
  a28=casadi_sign(a28);
  a20=(a20*a28);
  a20=(-a20);
  a20=(a40?a20:0);
  a33=(a33+a20);
  a20=(a27*a33);
  a16=(a16+a20);
  a20=(a71*a16);
  a30=(a5+a5);
  a20=(a20/a30);
  a36=(a8*a20);
  a49=(a49-a36);
  a16=(a16/a5);
  a36=(a3*a16);
  a49=(a49-a36);
  a49=(a1*a49);
  a36=(a17*a16);
  a36=(a1*a36);
  a49=(a49+a36);
  a49=(a64*a49);
  a191=(a191-a49);
  a49=cos(a2);
  a6=(a6+a6);
  a21=(a6*a21);
  a4=(a4+a4);
  a20=(a4*a20);
  a21=(a21-a20);
  a20=(a18*a16);
  a21=(a21+a20);
  a21=(a1*a21);
  a16=(a9*a16);
  a16=(a1*a16);
  a21=(a21-a16);
  a21=(a49*a21);
  a191=(a191+a21);
  if (res[0]!=0) res[0][8]=a191;
  a191=(a122*a179);
  a187=(a187/a175);
  a37=(a37*a173);
  a185=(a37*a185);
  a185=(a172*a185);
  a185=(a32*a185);
  a185=(a184?a185:0);
  a187=(a187+a185);
  a186=(a37/a186);
  a186=(a172*a186);
  a186=(a32*a186);
  a186=(a45*a186);
  a186=(-a186);
  a184=(a184?a186:0);
  a187=(a187+a184);
  a181=(a37*a181);
  a181=(a32*a181);
  a181=(a181*a177);
  a181=(-a181);
  a181=(a39?a181:0);
  a187=(a187+a181);
  a37=(a37/a182);
  a37=(a32*a37);
  a37=(a37*a174);
  a37=(-a37);
  a39=(a39?a37:0);
  a187=(a187+a39);
  a39=(a124*a187);
  a191=(a191+a39);
  a39=(a191/a148);
  a37=(a165*a39);
  a174=(a152*a37);
  a182=(a157*a39);
  a181=(a127*a182);
  a174=(a174-a181);
  a181=(a146*a174);
  a177=(a159*a182);
  a181=(a181+a177);
  a177=(a143*a181);
  a184=(a151*a174);
  a159=(a159*a37);
  a184=(a184-a159);
  a159=(a120*a184);
  a177=(a177-a159);
  a179=(a26*a179);
  a159=(a27*a187);
  a179=(a179+a159);
  a148=(a179/a148);
  a165=(a165*a148);
  a159=(a147*a165);
  a177=(a177+a159);
  a159=(a161*a37);
  a186=(a158*a182);
  a185=(a162*a37);
  a186=(a186-a185);
  a151=(a151*a186);
  a159=(a159+a151);
  a14=(a14/a170);
  a0=(a0?a14:0);
  a180=(a180/a170);
  a0=(a0-a180);
  a0=(a198*a0);
  a0=(a0/a150);
  a155=(a155*a0);
  a199=(a199*a191);
  a200=(a200*a179);
  a199=(a199+a200);
  a199=(a199/a193);
  a153=(a153*a199);
  a193=(a155-a153);
  a166=(a166*a39);
  a167=(a167*a148);
  a166=(a166+a167);
  a193=(a193+a166);
  a167=(a1*a193);
  a159=(a159-a167);
  a157=(a157*a148);
  a167=(a182+a157);
  a200=(a1*a167);
  a159=(a159+a200);
  a200=(a143*a159);
  a177=(a177+a200);
  a149=(a149*a0);
  a145=(a145*a199);
  a199=(a149-a145);
  a160=(a160*a39);
  a168=(a168*a148);
  a160=(a160+a168);
  a199=(a199+a160);
  a199=(a147*a199);
  a177=(a177+a199);
  a158=(a158*a174);
  a127=(a127*a186);
  a158=(a158+a127);
  a37=(a37+a165);
  a158=(a158-a37);
  a149=(a149-a145);
  a149=(a149+a160);
  a158=(a158-a149);
  a158=(a91*a158);
  a177=(a177+a158);
  a146=(a146*a186);
  a161=(a161*a182);
  a146=(a146-a161);
  a37=(a1*a37);
  a146=(a146+a37);
  a149=(a1*a149);
  a146=(a146+a149);
  a149=(a120*a146);
  a177=(a177+a149);
  a171=(a171*a177);
  a177=-4.8780487804877992e-01;
  a149=(a177*a189);
  a37=(a189*a149);
  a161=(a177*a194);
  a182=(a194*a161);
  a37=(a37+a182);
  a37=(a13*a37);
  a169=(a169*a37);
  a37=(a189*a161);
  a182=(a194*a149);
  a37=(a37-a182);
  a37=(a13*a37);
  a208=(a208*a37);
  a169=(a169+a208);
  a208=(a143*a184);
  a37=(a120*a181);
  a208=(a208+a37);
  a153=(a153-a155);
  a153=(a153-a166);
  a153=(a147*a153);
  a208=(a208+a153);
  a120=(a120*a159);
  a208=(a208+a120);
  a162=(a162*a174);
  a152=(a152*a186);
  a162=(a162+a152);
  a162=(a162+a193);
  a162=(a162-a167);
  a162=(a91*a162);
  a208=(a208+a162);
  a147=(a147*a157);
  a208=(a208+a147);
  a143=(a143*a146);
  a208=(a208-a143);
  a209=(a209*a208);
  a169=(a169+a209);
  a171=(a171-a169);
  a122=(a122*a210);
  a142=(a142/a125);
  a131=(a131*a140);
  a137=(a131*a137);
  a137=(a172*a137);
  a137=(a32*a137);
  a137=(a136?a137:0);
  a142=(a142+a137);
  a138=(a131/a138);
  a138=(a172*a138);
  a138=(a32*a138);
  a138=(a45*a138);
  a138=(-a138);
  a136=(a136?a138:0);
  a142=(a142+a136);
  a133=(a131*a133);
  a133=(a32*a133);
  a133=(a133*a121);
  a133=(-a133);
  a133=(a132?a133:0);
  a142=(a142+a133);
  a131=(a131/a134);
  a131=(a32*a131);
  a131=(a131*a123);
  a131=(-a131);
  a132=(a132?a131:0);
  a142=(a142+a132);
  a132=(a124*a142);
  a122=(a122+a132);
  a132=(a122/a94);
  a131=(a114*a132);
  a123=(a98*a131);
  a134=(a106*a132);
  a133=(a87*a134);
  a123=(a123-a133);
  a133=(a97*a123);
  a121=(a108*a131);
  a133=(a133-a121);
  a121=(a88*a133);
  a136=(a92*a123);
  a108=(a108*a134);
  a136=(a136+a108);
  a108=(a74*a136);
  a121=(a121+a108);
  a201=(a201*a122);
  a210=(a26*a210);
  a122=(a27*a142);
  a210=(a210+a122);
  a203=(a203*a210);
  a201=(a201+a203);
  a201=(a201/a211);
  a99=(a99*a201);
  a141=(a141/a119);
  a103=(a103?a141:0);
  a130=(a130/a119);
  a103=(a103-a130);
  a103=(a198*a103);
  a103=(a103/a96);
  a101=(a101*a103);
  a96=(a99-a101);
  a115=(a115*a132);
  a210=(a210/a94);
  a116=(a116*a210);
  a115=(a115+a116);
  a96=(a96-a115);
  a96=(a93*a96);
  a121=(a121+a96);
  a96=(a110*a131);
  a116=(a107*a134);
  a94=(a111*a131);
  a116=(a116-a94);
  a97=(a97*a116);
  a96=(a96+a97);
  a101=(a101-a99);
  a101=(a101+a115);
  a115=(a1*a101);
  a96=(a96-a115);
  a106=(a106*a210);
  a115=(a134+a106);
  a99=(a1*a115);
  a96=(a96+a99);
  a99=(a74*a96);
  a121=(a121+a99);
  a111=(a111*a123);
  a98=(a98*a116);
  a111=(a111+a98);
  a111=(a111+a101);
  a111=(a111-a115);
  a111=(a91*a111);
  a121=(a121+a111);
  a106=(a93*a106);
  a121=(a121+a106);
  a92=(a92*a116);
  a110=(a110*a134);
  a92=(a92-a110);
  a114=(a114*a210);
  a131=(a131+a114);
  a110=(a1*a131);
  a92=(a92+a110);
  a95=(a95*a103);
  a90=(a90*a201);
  a201=(a95-a90);
  a109=(a109*a132);
  a117=(a117*a210);
  a109=(a109+a117);
  a201=(a201+a109);
  a117=(a1*a201);
  a92=(a92+a117);
  a117=(a88*a92);
  a121=(a121-a117);
  a195=(a195*a121);
  a171=(a171-a195);
  a195=(a88*a136);
  a121=(a74*a133);
  a195=(a195-a121);
  a114=(a93*a114);
  a195=(a195+a114);
  a88=(a88*a96);
  a195=(a195+a88);
  a95=(a95-a90);
  a95=(a95+a109);
  a93=(a93*a95);
  a195=(a195+a93);
  a107=(a107*a123);
  a87=(a87*a116);
  a107=(a107+a87);
  a107=(a107-a131);
  a107=(a107-a201);
  a107=(a91*a107);
  a195=(a195+a107);
  a74=(a74*a92);
  a195=(a195+a74);
  a202=(a202*a195);
  a171=(a171+a202);
  a85=(a85/a65);
  a57=(a57?a85:0);
  a75=(a75/a65);
  a57=(a57-a75);
  a57=(a198*a57);
  a57=(a57/a53);
  a53=(a54*a57);
  a216=(a26*a216);
  a86=(a86/a70);
  a76=(a76*a68);
  a82=(a76*a82);
  a82=(a172*a82);
  a82=(a32*a82);
  a82=(a81?a82:0);
  a86=(a86+a82);
  a83=(a76/a83);
  a83=(a172*a83);
  a83=(a32*a83);
  a83=(a45*a83);
  a83=(-a83);
  a81=(a81?a83:0);
  a86=(a86+a81);
  a78=(a76*a78);
  a78=(a32*a78);
  a78=(a78*a126);
  a78=(-a78);
  a78=(a77?a78:0);
  a86=(a86+a78);
  a76=(a76/a79);
  a76=(a32*a76);
  a76=(a76*a69);
  a76=(-a76);
  a77=(a77?a76:0);
  a86=(a86+a77);
  a77=(a27*a86);
  a216=(a216+a77);
  a204=(a204*a216);
  a204=(a204/a67);
  a54=(a54*a204);
  a53=(a53-a54);
  a216=(a216/a62);
  a23=(a23*a216);
  a53=(a53-a23);
  a53=(a1*a53);
  a59=(a59*a216);
  a59=(a1*a59);
  a53=(a53+a59);
  a47=(a47*a53);
  a171=(a171-a47);
  a31=(a31*a57);
  a34=(a34*a204);
  a31=(a31-a34);
  a60=(a60*a216);
  a31=(a31+a60);
  a31=(a1*a31);
  a55=(a55*a216);
  a55=(a1*a55);
  a31=(a31-a55);
  a84=(a84*a31);
  a171=(a171+a84);
  a51=(a51/a22);
  a12=(a12?a51:0);
  a35=(a35/a22);
  a12=(a12-a35);
  a198=(a198*a12);
  a198=(a198/a7);
  a7=(a8*a198);
  a26=(a26*a63);
  a52=(a52/a29);
  a38=(a38*a50);
  a46=(a38*a46);
  a46=(a172*a46);
  a46=(a32*a46);
  a46=(a44?a46:0);
  a52=(a52+a46);
  a48=(a38/a48);
  a172=(a172*a48);
  a172=(a32*a172);
  a45=(a45*a172);
  a45=(-a45);
  a44=(a44?a45:0);
  a52=(a52+a44);
  a41=(a38*a41);
  a41=(a32*a41);
  a41=(a41*a25);
  a41=(-a41);
  a41=(a40?a41:0);
  a52=(a52+a41);
  a38=(a38/a42);
  a32=(a32*a38);
  a32=(a32*a28);
  a32=(-a32);
  a40=(a40?a32:0);
  a52=(a52+a40);
  a40=(a27*a52);
  a26=(a26+a40);
  a71=(a71*a26);
  a71=(a71/a30);
  a8=(a8*a71);
  a7=(a7-a8);
  a26=(a26/a5);
  a3=(a3*a26);
  a7=(a7-a3);
  a7=(a1*a7);
  a17=(a17*a26);
  a17=(a1*a17);
  a7=(a7+a17);
  a64=(a64*a7);
  a171=(a171-a64);
  a6=(a6*a198);
  a4=(a4*a71);
  a6=(a6-a4);
  a18=(a18*a26);
  a6=(a6+a18);
  a6=(a1*a6);
  a9=(a9*a26);
  a1=(a1*a9);
  a6=(a6-a1);
  a49=(a49*a6);
  a171=(a171+a49);
  if (res[0]!=0) res[0][9]=a171;
  a171=cos(a66);
  a49=(a91*a27);
  a27=(a27+a124);
  a6=(a27*a176);
  a1=(a124*a176);
  a6=(a6-a1);
  a1=(a49*a6);
  a9=cos(a2);
  a9=(a13*a9);
  a26=(a9*a176);
  a1=(a1-a26);
  a26=(a189*a9);
  a2=sin(a2);
  a13=(a13*a2);
  a2=(a194*a13);
  a26=(a26-a2);
  a2=(a189*a49);
  a18=(a2*a124);
  a26=(a26+a18);
  a18=(a27*a2);
  a26=(a26-a18);
  a18=(a154*a26);
  a1=(a1+a18);
  a18=(a13*a207);
  a1=(a1-a18);
  a1=(a171*a1);
  a18=sin(a66);
  a4=(a194*a49);
  a71=(a27*a4);
  a198=(a189*a13);
  a64=(a194*a9);
  a198=(a198+a64);
  a64=(a4*a124);
  a198=(a198+a64);
  a71=(a71-a198);
  a154=(a154*a71);
  a198=(a13*a176);
  a154=(a154-a198);
  a198=(a124*a207);
  a64=(a27*a207);
  a198=(a198-a64);
  a64=(a49*a198);
  a154=(a154+a64);
  a64=(a9*a207);
  a154=(a154+a64);
  a154=(a18*a154);
  a1=(a1-a154);
  a154=sin(a66);
  a64=(a144*a192);
  a7=(a129*a190);
  a64=(a64+a7);
  a7=(a129*a196);
  a64=(a64+a7);
  a7=(a144*a206);
  a64=(a64-a7);
  a64=(a154*a64);
  a1=(a1-a64);
  a64=cos(a66);
  a190=(a144*a190);
  a192=(a129*a192);
  a190=(a190-a192);
  a196=(a144*a196);
  a190=(a190+a196);
  a206=(a129*a206);
  a190=(a190+a206);
  a190=(a64*a190);
  a1=(a1+a190);
  a190=sin(a66);
  a206=(a89*a205);
  a196=(a72*a197);
  a206=(a206+a196);
  a196=(a72*a213);
  a206=(a206+a196);
  a196=(a89*a214);
  a206=(a206-a196);
  a206=(a190*a206);
  a1=(a1-a206);
  a66=cos(a66);
  a197=(a89*a197);
  a205=(a72*a205);
  a197=(a197-a205);
  a213=(a89*a213);
  a197=(a197+a213);
  a214=(a72*a214);
  a197=(a197+a214);
  a197=(a66*a197);
  a1=(a1+a197);
  if (res[0]!=0) res[0][10]=a1;
  a1=(a27*a149);
  a197=(a124*a149);
  a1=(a1-a197);
  a197=(a49*a1);
  a214=(a9*a149);
  a197=(a197-a214);
  a26=(a177*a26);
  a197=(a197+a26);
  a26=(a13*a161);
  a197=(a197-a26);
  a171=(a171*a197);
  a177=(a177*a71);
  a13=(a13*a149);
  a177=(a177-a13);
  a124=(a124*a161);
  a27=(a27*a161);
  a124=(a124-a27);
  a49=(a49*a124);
  a177=(a177+a49);
  a9=(a9*a161);
  a177=(a177+a9);
  a18=(a18*a177);
  a171=(a171-a18);
  a18=(a144*a184);
  a177=(a129*a181);
  a18=(a18+a177);
  a177=(a129*a159);
  a18=(a18+a177);
  a177=(a144*a146);
  a18=(a18-a177);
  a154=(a154*a18);
  a171=(a171-a154);
  a181=(a144*a181);
  a184=(a129*a184);
  a181=(a181-a184);
  a144=(a144*a159);
  a181=(a181+a144);
  a129=(a129*a146);
  a181=(a181+a129);
  a64=(a64*a181);
  a171=(a171+a64);
  a64=(a89*a133);
  a181=(a72*a136);
  a64=(a64+a181);
  a181=(a72*a96);
  a64=(a64+a181);
  a181=(a89*a92);
  a64=(a64-a181);
  a190=(a190*a64);
  a171=(a171-a190);
  a136=(a89*a136);
  a133=(a72*a133);
  a136=(a136-a133);
  a89=(a89*a96);
  a136=(a136+a89);
  a72=(a72*a92);
  a136=(a136+a72);
  a66=(a66*a136);
  a171=(a171+a66);
  if (res[0]!=0) res[0][11]=a171;
  a171=-1.;
  if (res[0]!=0) res[0][12]=a171;
  a66=(a4*a176);
  a136=(a2*a207);
  a66=(a66-a136);
  a6=(a194*a6);
  a198=(a189*a198);
  a6=(a6+a198);
  a6=(a91*a6);
  a6=(a66+a6);
  a198=(a163*a178);
  a6=(a6+a198);
  a198=(a112*a128);
  a6=(a6+a198);
  a73=(a61*a73);
  a6=(a6+a73);
  a33=(a19*a33);
  a6=(a6+a33);
  if (res[0]!=0) res[0][13]=a6;
  a6=(a4*a149);
  a33=(a2*a161);
  a6=(a6-a33);
  a194=(a194*a1);
  a189=(a189*a124);
  a194=(a194+a189);
  a91=(a91*a194);
  a91=(a6+a91);
  a194=(a163*a187);
  a91=(a91+a194);
  a194=(a112*a142);
  a91=(a91+a194);
  a86=(a61*a86);
  a91=(a91+a86);
  a52=(a19*a52);
  a91=(a91+a52);
  if (res[0]!=0) res[0][14]=a91;
  if (res[0]!=0) res[0][15]=a171;
  a176=(a4*a176);
  a66=(a66-a176);
  a207=(a2*a207);
  a66=(a66+a207);
  a178=(a164*a178);
  a66=(a66+a178);
  a128=(a113*a128);
  a66=(a66+a128);
  if (res[0]!=0) res[0][16]=a66;
  a4=(a4*a149);
  a6=(a6-a4);
  a2=(a2*a161);
  a6=(a6+a2);
  a187=(a164*a187);
  a6=(a6+a187);
  a142=(a113*a142);
  a6=(a6+a142);
  if (res[0]!=0) res[0][17]=a6;
  if (res[1]!=0) res[1][0]=a24;
  if (res[1]!=0) res[1][1]=a24;
  if (res[1]!=0) res[1][2]=a24;
  if (res[1]!=0) res[1][3]=a24;
  if (res[1]!=0) res[1][4]=a24;
  if (res[1]!=0) res[1][5]=a24;
  if (res[1]!=0) res[1][6]=a24;
  if (res[1]!=0) res[1][7]=a24;
  a24=2.7025639012821789e-01;
  a6=1.2330447799599942e+00;
  a142=1.4439765966454325e+00;
  a187=-2.7025639012821762e-01;
  a43=(a43*a10);
  a15=(a15*a43);
  a19=(a19*a15);
  a15=(a187*a19);
  a15=(a142*a15);
  a43=(a6*a15);
  a10=9.6278838983177628e-01;
  a19=(a10*a19);
  a43=(a43-a19);
  a43=(a24*a43);
  a43=(-a43);
  if (res[2]!=0) res[2][0]=a43;
  if (res[2]!=0) res[2][1]=a15;
  a80=(a80*a56);
  a58=(a58*a80);
  a61=(a61*a58);
  a58=(a187*a61);
  a58=(a142*a58);
  a80=(a6*a58);
  a61=(a10*a61);
  a80=(a80-a61);
  a80=(a24*a80);
  a80=(-a80);
  if (res[2]!=0) res[2][2]=a80;
  if (res[2]!=0) res[2][3]=a58;
  a135=(a135*a102);
  a104=(a104*a135);
  a112=(a112*a104);
  a135=(a187*a112);
  a102=9.6278838983177639e-01;
  a113=(a113*a104);
  a104=(a102*a113);
  a135=(a135+a104);
  a135=(a142*a135);
  a104=(a6*a135);
  a112=(a10*a112);
  a113=(a24*a113);
  a112=(a112+a113);
  a104=(a104-a112);
  a104=(a24*a104);
  a104=(-a104);
  if (res[2]!=0) res[2][4]=a104;
  if (res[2]!=0) res[2][5]=a135;
  a183=(a183*a156);
  a11=(a11*a183);
  a163=(a163*a11);
  a187=(a187*a163);
  a164=(a164*a11);
  a102=(a102*a164);
  a187=(a187+a102);
  a142=(a142*a187);
  a6=(a6*a142);
  a10=(a10*a163);
  a164=(a24*a164);
  a10=(a10+a164);
  a6=(a6-a10);
  a24=(a24*a6);
  a24=(-a24);
  if (res[2]!=0) res[2][6]=a24;
  if (res[2]!=0) res[2][7]=a142;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_14113236_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
