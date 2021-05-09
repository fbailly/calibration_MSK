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
  #define CASADI_PREFIX(ID) model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_ ## ID
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

static const casadi_int casadi_s0[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[33] = {12, 12, 0, 2, 4, 6, 8, 8, 8, 8, 8, 10, 12, 15, 18, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 8, 10, 11, 9, 10, 11};
static const casadi_int casadi_s4[27] = {12, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s5[15] = {12, 4, 0, 2, 4, 6, 8, 10, 11, 10, 11, 10, 11, 10, 11};
static const casadi_int casadi_s6[3] = {12, 0, 0};

/* model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z:(i0[12],i1[12],i2[4],i3[],i4[])->(o0[12x12,18nz],o1[12x12,12nz],o2[12x4,8nz],o3[12x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[2]? arg[2][0] : 0;
  a1=5.0000000000000000e-01;
  a2=arg[0]? arg[0][8] : 0;
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
  a10=4.0000000000000001e-02;
  a11=(a7-a10);
  a12=8.7758256189037276e-01;
  a11=(a11/a12);
  a13=arg[0]? arg[0][0] : 0;
  a14=(a11/a13);
  a15=1.;
  a16=(a14-a15);
  a17=casadi_sq(a16);
  a18=4.5000000000000001e-01;
  a17=(a17/a18);
  a17=(-a17);
  a17=exp(a17);
  a19=(a0*a17);
  a20=(a4+a5);
  a21=(a1*a8);
  a22=(a20*a21);
  a3=(a1*a3);
  a23=(a9*a3);
  a22=(a22-a23);
  a4=(a4+a5);
  a5=casadi_sq(a4);
  a23=casadi_sq(a9);
  a5=(a5+a23);
  a5=sqrt(a5);
  a22=(a22/a5);
  a23=arg[0]? arg[0][10] : 0;
  a24=(a22*a23);
  a25=0.;
  a26=(a24<=a25);
  a27=fabs(a24);
  a28=10.;
  a27=(a27/a28);
  a27=(a15-a27);
  a29=fabs(a24);
  a29=(a29/a28);
  a29=(a15+a29);
  a27=(a27/a29);
  a30=(a26?a27:0);
  a31=(!a26);
  a32=1.3300000000000001e+00;
  a33=(a32*a24);
  a33=(a33/a28);
  a34=-8.2500000000000004e-02;
  a33=(a33/a34);
  a33=(a15-a33);
  a35=(a24/a28);
  a35=(a35/a34);
  a35=(a15-a35);
  a33=(a33/a35);
  a36=(a31?a33:0);
  a30=(a30+a36);
  a36=(a19*a30);
  a37=(a10<a11);
  a11=(a11/a13);
  a38=(a11-a15);
  a38=(a28*a38);
  a38=exp(a38);
  a39=(a38-a15);
  a40=1.4741315910257660e+02;
  a39=(a39/a40);
  a39=(a37?a39:0);
  a36=(a36+a39);
  a39=1.0000000000000001e-01;
  a41=(a28*a13);
  a42=(a24/a41);
  a43=(a39*a42);
  a36=(a36+a43);
  a43=3.9024390243902418e-01;
  a44=(a43*a22);
  a45=(a36*a44);
  a42=(a42/a41);
  a44=(a13*a44);
  a46=(a39*a44);
  a47=(a42*a46);
  a47=(a28*a47);
  a45=(a45-a47);
  a47=6.7836549063042314e-03;
  a48=(a47*a44);
  a48=(a48*a38);
  a48=(a28*a48);
  a11=(a11/a13);
  a49=(a48*a11);
  a49=(-a49);
  a49=(a37?a49:0);
  a45=(a45+a49);
  a14=(a14/a13);
  a16=(a16+a16);
  a49=2.2222222222222223e+00;
  a50=(a30*a44);
  a50=(a0*a50);
  a50=(a17*a50);
  a50=(a49*a50);
  a50=(a16*a50);
  a51=(a14*a50);
  a45=(a45+a51);
  if (res[0]!=0) res[0][0]=a45;
  a45=-3.9024390243902396e-01;
  a51=(a45*a22);
  a52=(a36*a51);
  a51=(a13*a51);
  a53=(a39*a51);
  a42=(a42*a53);
  a42=(a28*a42);
  a52=(a52-a42);
  a42=(a47*a51);
  a42=(a42*a38);
  a42=(a28*a42);
  a11=(a42*a11);
  a11=(-a11);
  a11=(a37?a11:0);
  a52=(a52+a11);
  a11=(a30*a51);
  a0=(a0*a11);
  a0=(a17*a0);
  a0=(a49*a0);
  a16=(a16*a0);
  a14=(a14*a16);
  a52=(a52+a14);
  if (res[0]!=0) res[0][1]=a52;
  a52=arg[2]? arg[2][1] : 0;
  a14=sin(a2);
  a0=(a1*a14);
  a11=(a0+a1);
  a38=casadi_sq(a11);
  a54=cos(a2);
  a55=(a1*a54);
  a56=casadi_sq(a55);
  a38=(a38+a56);
  a38=sqrt(a38);
  a56=(a38-a10);
  a56=(a56/a12);
  a57=arg[0]? arg[0][1] : 0;
  a58=(a56/a57);
  a59=(a58-a15);
  a60=casadi_sq(a59);
  a60=(a60/a18);
  a60=(-a60);
  a60=exp(a60);
  a61=(a52*a60);
  a62=(a0+a1);
  a63=(a1*a54);
  a64=(a62*a63);
  a14=(a1*a14);
  a65=(a55*a14);
  a64=(a64-a65);
  a0=(a0+a1);
  a65=casadi_sq(a0);
  a66=casadi_sq(a55);
  a65=(a65+a66);
  a65=sqrt(a65);
  a64=(a64/a65);
  a66=(a64*a23);
  a67=(a66<=a25);
  a68=fabs(a66);
  a68=(a68/a28);
  a68=(a15-a68);
  a69=fabs(a66);
  a69=(a69/a28);
  a69=(a15+a69);
  a68=(a68/a69);
  a70=(a67?a68:0);
  a71=(!a67);
  a72=(a32*a66);
  a72=(a72/a28);
  a72=(a72/a34);
  a72=(a15-a72);
  a73=(a66/a28);
  a73=(a73/a34);
  a73=(a15-a73);
  a72=(a72/a73);
  a74=(a71?a72:0);
  a70=(a70+a74);
  a74=(a61*a70);
  a75=(a10<a56);
  a56=(a56/a57);
  a76=(a56-a15);
  a76=(a28*a76);
  a76=exp(a76);
  a77=(a76-a15);
  a77=(a77/a40);
  a77=(a75?a77:0);
  a74=(a74+a77);
  a77=(a28*a57);
  a78=(a66/a77);
  a79=(a39*a78);
  a74=(a74+a79);
  a79=(a43*a64);
  a80=(a74*a79);
  a78=(a78/a77);
  a79=(a57*a79);
  a81=(a39*a79);
  a82=(a78*a81);
  a82=(a28*a82);
  a80=(a80-a82);
  a82=(a47*a79);
  a82=(a82*a76);
  a82=(a28*a82);
  a56=(a56/a57);
  a83=(a82*a56);
  a83=(-a83);
  a83=(a75?a83:0);
  a80=(a80+a83);
  a58=(a58/a57);
  a59=(a59+a59);
  a83=(a70*a79);
  a83=(a52*a83);
  a83=(a60*a83);
  a83=(a49*a83);
  a83=(a59*a83);
  a84=(a58*a83);
  a80=(a80+a84);
  if (res[0]!=0) res[0][2]=a80;
  a80=(a45*a64);
  a84=(a74*a80);
  a80=(a57*a80);
  a85=(a39*a80);
  a78=(a78*a85);
  a78=(a28*a78);
  a84=(a84-a78);
  a78=(a47*a80);
  a78=(a78*a76);
  a78=(a28*a78);
  a56=(a78*a56);
  a56=(-a56);
  a56=(a75?a56:0);
  a84=(a84+a56);
  a56=(a70*a80);
  a52=(a52*a56);
  a52=(a60*a52);
  a52=(a49*a52);
  a59=(a59*a52);
  a58=(a58*a59);
  a84=(a84+a58);
  if (res[0]!=0) res[0][3]=a84;
  a84=arg[2]? arg[2][2] : 0;
  a58=arg[0]? arg[0][9] : 0;
  a52=sin(a58);
  a56=sin(a2);
  a76=(a52*a56);
  a86=cos(a58);
  a87=cos(a2);
  a88=(a86*a87);
  a76=(a76-a88);
  a88=(a1*a76);
  a89=1.2500000000000000e+00;
  a90=(a89*a56);
  a88=(a88-a90);
  a91=7.5000000000000000e-01;
  a92=(a91*a56);
  a93=(a88+a92);
  a94=casadi_sq(a93);
  a95=(a89*a87);
  a96=(a86*a56);
  a97=(a52*a87);
  a96=(a96+a97);
  a97=(a1*a96);
  a97=(a95-a97);
  a98=(a91*a87);
  a99=(a97-a98);
  a100=casadi_sq(a99);
  a94=(a94+a100);
  a94=sqrt(a94);
  a100=(a94-a10);
  a100=(a100/a12);
  a101=arg[0]? arg[0][2] : 0;
  a102=(a100/a101);
  a103=(a102-a15);
  a104=casadi_sq(a103);
  a104=(a104/a18);
  a104=(-a104);
  a104=exp(a104);
  a105=(a84*a104);
  a106=(a88+a92);
  a107=(a91*a87);
  a108=(a1*a96);
  a108=(a95-a108);
  a107=(a107-a108);
  a109=(a106*a107);
  a110=(a97-a98);
  a111=(a1*a76);
  a111=(a111-a90);
  a112=(a91*a56);
  a112=(a111+a112);
  a113=(a110*a112);
  a109=(a109+a113);
  a88=(a88+a92);
  a92=casadi_sq(a88);
  a97=(a97-a98);
  a98=casadi_sq(a97);
  a92=(a92+a98);
  a92=sqrt(a92);
  a109=(a109/a92);
  a98=(a109*a23);
  a113=(a52*a87);
  a114=(a86*a56);
  a113=(a113+a114);
  a114=(a76*a90);
  a115=(a96*a95);
  a114=(a114+a115);
  a115=(a113*a114);
  a116=(a113*a90);
  a117=(a86*a87);
  a118=(a52*a56);
  a117=(a117-a118);
  a118=(a117*a95);
  a116=(a116+a118);
  a118=(a76*a116);
  a115=(a115-a118);
  a115=(a115-a108);
  a108=(a106*a115);
  a118=(a96*a116);
  a119=(a117*a114);
  a118=(a118-a119);
  a118=(a118+a111);
  a111=(a110*a118);
  a108=(a108+a111);
  a108=(a108/a92);
  a111=arg[0]? arg[0][11] : 0;
  a119=(a108*a111);
  a98=(a98+a119);
  a119=(a98<=a25);
  a120=fabs(a98);
  a120=(a120/a28);
  a120=(a15-a120);
  a121=fabs(a98);
  a121=(a121/a28);
  a121=(a15+a121);
  a120=(a120/a121);
  a122=(a119?a120:0);
  a123=(!a119);
  a124=(a32*a98);
  a124=(a124/a28);
  a124=(a124/a34);
  a124=(a15-a124);
  a125=(a98/a28);
  a125=(a125/a34);
  a125=(a15-a125);
  a124=(a124/a125);
  a126=(a123?a124:0);
  a122=(a122+a126);
  a126=(a105*a122);
  a127=(a10<a100);
  a100=(a100/a101);
  a128=(a100-a15);
  a128=(a28*a128);
  a128=exp(a128);
  a129=(a128-a15);
  a129=(a129/a40);
  a129=(a127?a129:0);
  a126=(a126+a129);
  a129=(a28*a101);
  a130=(a98/a129);
  a131=(a39*a130);
  a126=(a126+a131);
  a131=-3.9024390243902440e-01;
  a132=(a131*a108);
  a133=(a43*a109);
  a132=(a132+a133);
  a133=(a126*a132);
  a130=(a130/a129);
  a132=(a101*a132);
  a134=(a39*a132);
  a135=(a130*a134);
  a135=(a28*a135);
  a133=(a133-a135);
  a135=(a47*a132);
  a135=(a135*a128);
  a135=(a28*a135);
  a100=(a100/a101);
  a136=(a135*a100);
  a136=(-a136);
  a136=(a127?a136:0);
  a133=(a133+a136);
  a102=(a102/a101);
  a103=(a103+a103);
  a136=(a122*a132);
  a136=(a84*a136);
  a136=(a104*a136);
  a136=(a49*a136);
  a136=(a103*a136);
  a137=(a102*a136);
  a133=(a133+a137);
  if (res[0]!=0) res[0][4]=a133;
  a133=1.3902439024390245e+00;
  a137=(a133*a108);
  a138=(a45*a109);
  a137=(a137+a138);
  a138=(a126*a137);
  a137=(a101*a137);
  a139=(a39*a137);
  a130=(a130*a139);
  a130=(a28*a130);
  a138=(a138-a130);
  a130=(a47*a137);
  a130=(a130*a128);
  a130=(a28*a130);
  a100=(a130*a100);
  a100=(-a100);
  a100=(a127?a100:0);
  a138=(a138+a100);
  a100=(a122*a137);
  a84=(a84*a100);
  a84=(a104*a84);
  a84=(a49*a84);
  a103=(a103*a84);
  a102=(a102*a103);
  a138=(a138+a102);
  if (res[0]!=0) res[0][5]=a138;
  a138=arg[2]? arg[2][3] : 0;
  a102=sin(a58);
  a84=sin(a2);
  a100=(a102*a84);
  a128=cos(a58);
  a140=cos(a2);
  a141=(a128*a140);
  a100=(a100-a141);
  a141=(a1*a100);
  a142=(a89*a84);
  a141=(a141-a142);
  a143=1.7500000000000000e+00;
  a144=(a143*a84);
  a145=(a141+a144);
  a146=casadi_sq(a145);
  a147=(a89*a140);
  a148=(a128*a84);
  a149=(a102*a140);
  a148=(a148+a149);
  a149=(a1*a148);
  a149=(a147-a149);
  a150=(a143*a140);
  a151=(a149-a150);
  a152=casadi_sq(a151);
  a146=(a146+a152);
  a146=sqrt(a146);
  a152=(a146-a10);
  a152=(a152/a12);
  a12=arg[0]? arg[0][3] : 0;
  a153=(a152/a12);
  a154=(a153-a15);
  a155=casadi_sq(a154);
  a155=(a155/a18);
  a155=(-a155);
  a155=exp(a155);
  a18=(a138*a155);
  a156=(a141+a144);
  a157=(a143*a140);
  a158=(a1*a148);
  a158=(a147-a158);
  a157=(a157-a158);
  a159=(a156*a157);
  a160=(a149-a150);
  a161=(a1*a100);
  a161=(a161-a142);
  a162=(a143*a84);
  a162=(a161+a162);
  a163=(a160*a162);
  a159=(a159+a163);
  a141=(a141+a144);
  a144=casadi_sq(a141);
  a149=(a149-a150);
  a150=casadi_sq(a149);
  a144=(a144+a150);
  a144=sqrt(a144);
  a159=(a159/a144);
  a150=(a159*a23);
  a163=(a102*a140);
  a164=(a128*a84);
  a163=(a163+a164);
  a164=(a100*a142);
  a165=(a148*a147);
  a164=(a164+a165);
  a165=(a163*a164);
  a166=(a163*a142);
  a167=(a128*a140);
  a168=(a102*a84);
  a167=(a167-a168);
  a168=(a167*a147);
  a166=(a166+a168);
  a168=(a100*a166);
  a165=(a165-a168);
  a165=(a165-a158);
  a158=(a156*a165);
  a168=(a148*a166);
  a169=(a167*a164);
  a168=(a168-a169);
  a168=(a168+a161);
  a161=(a160*a168);
  a158=(a158+a161);
  a158=(a158/a144);
  a161=(a158*a111);
  a150=(a150+a161);
  a25=(a150<=a25);
  a161=fabs(a150);
  a161=(a161/a28);
  a161=(a15-a161);
  a169=fabs(a150);
  a169=(a169/a28);
  a169=(a15+a169);
  a161=(a161/a169);
  a170=(a25?a161:0);
  a171=(!a25);
  a172=(a32*a150);
  a172=(a172/a28);
  a172=(a172/a34);
  a172=(a15-a172);
  a173=(a150/a28);
  a173=(a173/a34);
  a173=(a15-a173);
  a172=(a172/a173);
  a34=(a171?a172:0);
  a170=(a170+a34);
  a34=(a18*a170);
  a10=(a10<a152);
  a152=(a152/a12);
  a174=(a152-a15);
  a174=(a28*a174);
  a174=exp(a174);
  a175=(a174-a15);
  a175=(a175/a40);
  a175=(a10?a175:0);
  a34=(a34+a175);
  a175=(a28*a12);
  a40=(a150/a175);
  a176=(a39*a40);
  a34=(a34+a176);
  a176=(a131*a158);
  a177=(a43*a159);
  a176=(a176+a177);
  a177=(a34*a176);
  a40=(a40/a175);
  a176=(a12*a176);
  a178=(a39*a176);
  a179=(a40*a178);
  a179=(a28*a179);
  a177=(a177-a179);
  a179=(a47*a176);
  a179=(a179*a174);
  a179=(a28*a179);
  a152=(a152/a12);
  a180=(a179*a152);
  a180=(-a180);
  a180=(a10?a180:0);
  a177=(a177+a180);
  a153=(a153/a12);
  a154=(a154+a154);
  a180=(a170*a176);
  a180=(a138*a180);
  a180=(a155*a180);
  a180=(a49*a180);
  a180=(a154*a180);
  a181=(a153*a180);
  a177=(a177+a181);
  if (res[0]!=0) res[0][6]=a177;
  a177=(a133*a158);
  a181=(a45*a159);
  a177=(a177+a181);
  a181=(a34*a177);
  a177=(a12*a177);
  a182=(a39*a177);
  a40=(a40*a182);
  a40=(a28*a40);
  a181=(a181-a40);
  a47=(a47*a177);
  a47=(a47*a174);
  a28=(a28*a47);
  a152=(a28*a152);
  a152=(-a152);
  a152=(a10?a152:0);
  a181=(a181+a152);
  a152=(a170*a177);
  a138=(a138*a152);
  a138=(a155*a138);
  a49=(a49*a138);
  a154=(a154*a49);
  a153=(a153*a154);
  a181=(a181+a153);
  if (res[0]!=0) res[0][7]=a181;
  a181=cos(a2);
  a34=(a12*a34);
  a153=(a131*a34);
  a178=(a178/a175);
  a49=-1.2121212121212121e+01;
  a176=(a18*a176);
  a172=(a172/a173);
  a138=(a176*a172);
  a138=(a49*a138);
  a138=(a39*a138);
  a138=(a171?a138:0);
  a178=(a178+a138);
  a138=(a176/a173);
  a138=(a49*a138);
  a138=(a39*a138);
  a138=(a32*a138);
  a138=(-a138);
  a138=(a171?a138:0);
  a178=(a178+a138);
  a161=(a161/a169);
  a138=(a176*a161);
  a138=(a39*a138);
  a152=casadi_sign(a150);
  a138=(a138*a152);
  a138=(-a138);
  a138=(a25?a138:0);
  a178=(a178+a138);
  a176=(a176/a169);
  a176=(a39*a176);
  a150=casadi_sign(a150);
  a176=(a176*a150);
  a176=(-a176);
  a176=(a25?a176:0);
  a178=(a178+a176);
  a176=(a111*a178);
  a153=(a153+a176);
  a176=(a153/a144);
  a138=(a160*a176);
  a47=(a148*a138);
  a174=(a156*a176);
  a40=(a100*a174);
  a47=(a47-a40);
  a40=(a142*a47);
  a183=(a164*a174);
  a40=(a40+a183);
  a183=(a128*a40);
  a184=(a147*a47);
  a185=(a164*a138);
  a184=(a184-a185);
  a185=(a102*a184);
  a183=(a183-a185);
  a185=(a43*a34);
  a186=(a23*a178);
  a185=(a185+a186);
  a186=(a185/a144);
  a187=(a160*a186);
  a188=(a143*a187);
  a183=(a183+a188);
  a188=(a166*a138);
  a189=(a163*a174);
  a190=(a167*a138);
  a189=(a189-a190);
  a190=(a147*a189);
  a188=(a188+a190);
  a151=(a151+a151);
  a190=1.1394939273245490e+00;
  a179=(a179/a12);
  a179=(a10?a179:0);
  a180=(a180/a12);
  a179=(a179-a180);
  a179=(a190*a179);
  a146=(a146+a146);
  a179=(a179/a146);
  a180=(a151*a179);
  a149=(a149+a149);
  a191=(a158/a144);
  a153=(a191*a153);
  a192=(a159/a144);
  a185=(a192*a185);
  a153=(a153+a185);
  a185=(a144+a144);
  a153=(a153/a185);
  a193=(a149*a153);
  a194=(a180-a193);
  a195=(a168*a176);
  a196=(a162*a186);
  a195=(a195+a196);
  a194=(a194+a195);
  a196=(a1*a194);
  a188=(a188-a196);
  a196=(a156*a186);
  a197=(a174+a196);
  a198=(a1*a197);
  a188=(a188+a198);
  a198=(a128*a188);
  a183=(a183+a198);
  a145=(a145+a145);
  a179=(a145*a179);
  a141=(a141+a141);
  a153=(a141*a153);
  a198=(a179-a153);
  a176=(a165*a176);
  a186=(a157*a186);
  a176=(a176+a186);
  a198=(a198+a176);
  a198=(a143*a198);
  a183=(a183+a198);
  a198=(a163*a47);
  a186=(a100*a189);
  a198=(a198+a186);
  a138=(a138+a187);
  a198=(a198-a138);
  a179=(a179-a153);
  a179=(a179+a176);
  a198=(a198-a179);
  a198=(a89*a198);
  a183=(a183+a198);
  a198=(a142*a189);
  a174=(a166*a174);
  a198=(a198-a174);
  a138=(a1*a138);
  a198=(a198+a138);
  a179=(a1*a179);
  a198=(a198+a179);
  a179=(a102*a198);
  a183=(a183+a179);
  a183=(a181*a183);
  a179=cos(a2);
  a138=9.8100000000000005e+00;
  a174=cos(a58);
  a176=4.8780487804878025e-01;
  a153=(a176*a174);
  a187=(a174*a153);
  a186=sin(a58);
  a199=(a176*a186);
  a200=(a186*a199);
  a187=(a187+a200);
  a187=(a138*a187);
  a187=(a179*a187);
  a200=sin(a2);
  a201=(a174*a199);
  a202=(a186*a153);
  a201=(a201-a202);
  a201=(a138*a201);
  a201=(a200*a201);
  a187=(a187+a201);
  a201=sin(a2);
  a202=(a128*a184);
  a203=(a102*a40);
  a202=(a202+a203);
  a193=(a193-a180);
  a193=(a193-a195);
  a193=(a143*a193);
  a202=(a202+a193);
  a193=(a102*a188);
  a202=(a202+a193);
  a47=(a167*a47);
  a189=(a148*a189);
  a47=(a47+a189);
  a47=(a47+a194);
  a47=(a47-a197);
  a47=(a89*a47);
  a202=(a202+a47);
  a196=(a143*a196);
  a202=(a202+a196);
  a196=(a128*a198);
  a202=(a202-a196);
  a202=(a201*a202);
  a187=(a187+a202);
  a183=(a183-a187);
  a187=sin(a2);
  a126=(a101*a126);
  a131=(a131*a126);
  a134=(a134/a129);
  a132=(a105*a132);
  a124=(a124/a125);
  a202=(a132*a124);
  a202=(a49*a202);
  a202=(a39*a202);
  a202=(a123?a202:0);
  a134=(a134+a202);
  a202=(a132/a125);
  a202=(a49*a202);
  a202=(a39*a202);
  a202=(a32*a202);
  a202=(-a202);
  a202=(a123?a202:0);
  a134=(a134+a202);
  a120=(a120/a121);
  a202=(a132*a120);
  a202=(a39*a202);
  a196=casadi_sign(a98);
  a202=(a202*a196);
  a202=(-a202);
  a202=(a119?a202:0);
  a134=(a134+a202);
  a132=(a132/a121);
  a132=(a39*a132);
  a98=casadi_sign(a98);
  a132=(a132*a98);
  a132=(-a132);
  a132=(a119?a132:0);
  a134=(a134+a132);
  a132=(a111*a134);
  a131=(a131+a132);
  a132=(a131/a92);
  a202=(a110*a132);
  a47=(a96*a202);
  a197=(a106*a132);
  a194=(a76*a197);
  a47=(a47-a194);
  a194=(a95*a47);
  a189=(a114*a202);
  a194=(a194-a189);
  a189=(a86*a194);
  a193=(a90*a47);
  a195=(a114*a197);
  a193=(a193+a195);
  a195=(a52*a193);
  a189=(a189+a195);
  a97=(a97+a97);
  a195=(a108/a92);
  a131=(a195*a131);
  a180=(a109/a92);
  a203=(a43*a126);
  a204=(a23*a134);
  a203=(a203+a204);
  a204=(a180*a203);
  a131=(a131+a204);
  a204=(a92+a92);
  a131=(a131/a204);
  a205=(a97*a131);
  a99=(a99+a99);
  a135=(a135/a101);
  a135=(a127?a135:0);
  a136=(a136/a101);
  a135=(a135-a136);
  a135=(a190*a135);
  a94=(a94+a94);
  a135=(a135/a94);
  a136=(a99*a135);
  a206=(a205-a136);
  a207=(a118*a132);
  a203=(a203/a92);
  a208=(a112*a203);
  a207=(a207+a208);
  a206=(a206-a207);
  a206=(a91*a206);
  a189=(a189+a206);
  a206=(a116*a202);
  a208=(a113*a197);
  a209=(a117*a202);
  a208=(a208-a209);
  a209=(a95*a208);
  a206=(a206+a209);
  a136=(a136-a205);
  a136=(a136+a207);
  a207=(a1*a136);
  a206=(a206-a207);
  a207=(a106*a203);
  a205=(a197+a207);
  a209=(a1*a205);
  a206=(a206+a209);
  a209=(a52*a206);
  a189=(a189+a209);
  a209=(a117*a47);
  a210=(a96*a208);
  a209=(a209+a210);
  a209=(a209+a136);
  a209=(a209-a205);
  a209=(a89*a209);
  a189=(a189+a209);
  a207=(a91*a207);
  a189=(a189+a207);
  a207=(a90*a208);
  a197=(a116*a197);
  a207=(a207-a197);
  a197=(a110*a203);
  a202=(a202+a197);
  a209=(a1*a202);
  a207=(a207+a209);
  a93=(a93+a93);
  a135=(a93*a135);
  a88=(a88+a88);
  a131=(a88*a131);
  a209=(a135-a131);
  a132=(a115*a132);
  a203=(a107*a203);
  a132=(a132+a203);
  a209=(a209+a132);
  a203=(a1*a209);
  a207=(a207+a203);
  a203=(a86*a207);
  a189=(a189-a203);
  a189=(a187*a189);
  a183=(a183-a189);
  a189=cos(a2);
  a203=(a86*a193);
  a205=(a52*a194);
  a203=(a203-a205);
  a197=(a91*a197);
  a203=(a203+a197);
  a197=(a86*a206);
  a203=(a203+a197);
  a135=(a135-a131);
  a135=(a135+a132);
  a135=(a91*a135);
  a203=(a203+a135);
  a47=(a113*a47);
  a208=(a76*a208);
  a47=(a47+a208);
  a47=(a47-a202);
  a47=(a47-a209);
  a47=(a89*a47);
  a203=(a203+a47);
  a47=(a52*a207);
  a203=(a203+a47);
  a203=(a189*a203);
  a183=(a183+a203);
  a203=sin(a2);
  a82=(a82/a57);
  a82=(a75?a82:0);
  a83=(a83/a57);
  a82=(a82-a83);
  a82=(a190*a82);
  a38=(a38+a38);
  a82=(a82/a38);
  a83=(a54*a82);
  a47=(a64/a65);
  a74=(a57*a74);
  a209=(a43*a74);
  a81=(a81/a77);
  a79=(a61*a79);
  a72=(a72/a73);
  a202=(a79*a72);
  a202=(a49*a202);
  a202=(a39*a202);
  a202=(a71?a202:0);
  a81=(a81+a202);
  a202=(a79/a73);
  a202=(a49*a202);
  a202=(a39*a202);
  a202=(a32*a202);
  a202=(-a202);
  a202=(a71?a202:0);
  a81=(a81+a202);
  a68=(a68/a69);
  a202=(a79*a68);
  a202=(a39*a202);
  a208=casadi_sign(a66);
  a202=(a202*a208);
  a202=(-a202);
  a202=(a67?a202:0);
  a81=(a81+a202);
  a79=(a79/a69);
  a79=(a39*a79);
  a66=casadi_sign(a66);
  a79=(a79*a66);
  a79=(-a79);
  a79=(a67?a79:0);
  a81=(a81+a79);
  a79=(a23*a81);
  a209=(a209+a79);
  a79=(a47*a209);
  a202=(a65+a65);
  a79=(a79/a202);
  a135=(a54*a79);
  a83=(a83-a135);
  a209=(a209/a65);
  a135=(a14*a209);
  a83=(a83-a135);
  a83=(a1*a83);
  a135=(a62*a209);
  a135=(a1*a135);
  a83=(a83+a135);
  a83=(a203*a83);
  a183=(a183-a83);
  a83=cos(a2);
  a11=(a11+a11);
  a82=(a11*a82);
  a0=(a0+a0);
  a79=(a0*a79);
  a82=(a82-a79);
  a79=(a63*a209);
  a82=(a82+a79);
  a82=(a1*a82);
  a209=(a55*a209);
  a209=(a1*a209);
  a82=(a82-a209);
  a82=(a83*a82);
  a183=(a183+a82);
  a82=sin(a2);
  a48=(a48/a13);
  a48=(a37?a48:0);
  a50=(a50/a13);
  a48=(a48-a50);
  a48=(a190*a48);
  a7=(a7+a7);
  a48=(a48/a7);
  a50=(a8*a48);
  a209=(a22/a5);
  a36=(a13*a36);
  a43=(a43*a36);
  a46=(a46/a41);
  a44=(a19*a44);
  a33=(a33/a35);
  a79=(a44*a33);
  a79=(a49*a79);
  a79=(a39*a79);
  a79=(a31?a79:0);
  a46=(a46+a79);
  a79=(a44/a35);
  a79=(a49*a79);
  a79=(a39*a79);
  a79=(a32*a79);
  a79=(-a79);
  a79=(a31?a79:0);
  a46=(a46+a79);
  a27=(a27/a29);
  a79=(a44*a27);
  a79=(a39*a79);
  a135=casadi_sign(a24);
  a79=(a79*a135);
  a79=(-a79);
  a79=(a26?a79:0);
  a46=(a46+a79);
  a44=(a44/a29);
  a44=(a39*a44);
  a24=casadi_sign(a24);
  a44=(a44*a24);
  a44=(-a44);
  a44=(a26?a44:0);
  a46=(a46+a44);
  a44=(a23*a46);
  a43=(a43+a44);
  a44=(a209*a43);
  a79=(a5+a5);
  a44=(a44/a79);
  a132=(a8*a44);
  a50=(a50-a132);
  a43=(a43/a5);
  a132=(a3*a43);
  a50=(a50-a132);
  a50=(a1*a50);
  a132=(a20*a43);
  a132=(a1*a132);
  a50=(a50+a132);
  a50=(a82*a50);
  a183=(a183-a50);
  a50=cos(a2);
  a6=(a6+a6);
  a48=(a6*a48);
  a4=(a4+a4);
  a44=(a4*a44);
  a48=(a48-a44);
  a44=(a21*a43);
  a48=(a48+a44);
  a48=(a1*a48);
  a43=(a9*a43);
  a43=(a1*a43);
  a48=(a48-a43);
  a48=(a50*a48);
  a183=(a183+a48);
  if (res[0]!=0) res[0][8]=a183;
  a183=(a133*a34);
  a182=(a182/a175);
  a18=(a18*a177);
  a172=(a18*a172);
  a172=(a49*a172);
  a172=(a39*a172);
  a172=(a171?a172:0);
  a182=(a182+a172);
  a173=(a18/a173);
  a173=(a49*a173);
  a173=(a39*a173);
  a173=(a32*a173);
  a173=(-a173);
  a171=(a171?a173:0);
  a182=(a182+a171);
  a161=(a18*a161);
  a161=(a39*a161);
  a161=(a161*a152);
  a161=(-a161);
  a161=(a25?a161:0);
  a182=(a182+a161);
  a18=(a18/a169);
  a18=(a39*a18);
  a18=(a18*a150);
  a18=(-a18);
  a25=(a25?a18:0);
  a182=(a182+a25);
  a25=(a111*a182);
  a183=(a183+a25);
  a25=(a183/a144);
  a18=(a160*a25);
  a150=(a148*a18);
  a169=(a156*a25);
  a161=(a100*a169);
  a150=(a150-a161);
  a161=(a142*a150);
  a152=(a164*a169);
  a161=(a161+a152);
  a152=(a128*a161);
  a171=(a147*a150);
  a164=(a164*a18);
  a171=(a171-a164);
  a164=(a102*a171);
  a152=(a152-a164);
  a34=(a45*a34);
  a164=(a23*a182);
  a34=(a34+a164);
  a144=(a34/a144);
  a160=(a160*a144);
  a164=(a143*a160);
  a152=(a152+a164);
  a164=(a166*a18);
  a173=(a163*a169);
  a172=(a167*a18);
  a173=(a173-a172);
  a147=(a147*a173);
  a164=(a164+a147);
  a28=(a28/a12);
  a10=(a10?a28:0);
  a154=(a154/a12);
  a10=(a10-a154);
  a10=(a190*a10);
  a10=(a10/a146);
  a151=(a151*a10);
  a191=(a191*a183);
  a192=(a192*a34);
  a191=(a191+a192);
  a191=(a191/a185);
  a149=(a149*a191);
  a185=(a151-a149);
  a168=(a168*a25);
  a162=(a162*a144);
  a168=(a168+a162);
  a185=(a185+a168);
  a162=(a1*a185);
  a164=(a164-a162);
  a156=(a156*a144);
  a162=(a169+a156);
  a192=(a1*a162);
  a164=(a164+a192);
  a192=(a128*a164);
  a152=(a152+a192);
  a145=(a145*a10);
  a141=(a141*a191);
  a191=(a145-a141);
  a165=(a165*a25);
  a157=(a157*a144);
  a165=(a165+a157);
  a191=(a191+a165);
  a191=(a143*a191);
  a152=(a152+a191);
  a163=(a163*a150);
  a100=(a100*a173);
  a163=(a163+a100);
  a18=(a18+a160);
  a163=(a163-a18);
  a145=(a145-a141);
  a145=(a145+a165);
  a163=(a163-a145);
  a163=(a89*a163);
  a152=(a152+a163);
  a142=(a142*a173);
  a166=(a166*a169);
  a142=(a142-a166);
  a18=(a1*a18);
  a142=(a142+a18);
  a145=(a1*a145);
  a142=(a142+a145);
  a145=(a102*a142);
  a152=(a152+a145);
  a181=(a181*a152);
  a152=-4.8780487804877992e-01;
  a145=(a152*a174);
  a18=(a174*a145);
  a166=(a152*a186);
  a169=(a186*a166);
  a18=(a18+a169);
  a18=(a138*a18);
  a179=(a179*a18);
  a18=(a174*a166);
  a169=(a186*a145);
  a18=(a18-a169);
  a18=(a138*a18);
  a200=(a200*a18);
  a179=(a179+a200);
  a200=(a128*a171);
  a18=(a102*a161);
  a200=(a200+a18);
  a149=(a149-a151);
  a149=(a149-a168);
  a149=(a143*a149);
  a200=(a200+a149);
  a102=(a102*a164);
  a200=(a200+a102);
  a167=(a167*a150);
  a148=(a148*a173);
  a167=(a167+a148);
  a167=(a167+a185);
  a167=(a167-a162);
  a167=(a89*a167);
  a200=(a200+a167);
  a143=(a143*a156);
  a200=(a200+a143);
  a128=(a128*a142);
  a200=(a200-a128);
  a201=(a201*a200);
  a179=(a179+a201);
  a181=(a181-a179);
  a133=(a133*a126);
  a139=(a139/a129);
  a105=(a105*a137);
  a124=(a105*a124);
  a124=(a49*a124);
  a124=(a39*a124);
  a124=(a123?a124:0);
  a139=(a139+a124);
  a125=(a105/a125);
  a125=(a49*a125);
  a125=(a39*a125);
  a125=(a32*a125);
  a125=(-a125);
  a123=(a123?a125:0);
  a139=(a139+a123);
  a120=(a105*a120);
  a120=(a39*a120);
  a120=(a120*a196);
  a120=(-a120);
  a120=(a119?a120:0);
  a139=(a139+a120);
  a105=(a105/a121);
  a105=(a39*a105);
  a105=(a105*a98);
  a105=(-a105);
  a119=(a119?a105:0);
  a139=(a139+a119);
  a119=(a111*a139);
  a133=(a133+a119);
  a119=(a133/a92);
  a105=(a110*a119);
  a98=(a96*a105);
  a121=(a106*a119);
  a120=(a76*a121);
  a98=(a98-a120);
  a120=(a95*a98);
  a196=(a114*a105);
  a120=(a120-a196);
  a196=(a86*a120);
  a123=(a90*a98);
  a114=(a114*a121);
  a123=(a123+a114);
  a114=(a52*a123);
  a196=(a196+a114);
  a195=(a195*a133);
  a126=(a45*a126);
  a133=(a23*a139);
  a126=(a126+a133);
  a180=(a180*a126);
  a195=(a195+a180);
  a195=(a195/a204);
  a97=(a97*a195);
  a130=(a130/a101);
  a127=(a127?a130:0);
  a103=(a103/a101);
  a127=(a127-a103);
  a127=(a190*a127);
  a127=(a127/a94);
  a99=(a99*a127);
  a94=(a97-a99);
  a118=(a118*a119);
  a126=(a126/a92);
  a112=(a112*a126);
  a118=(a118+a112);
  a94=(a94-a118);
  a94=(a91*a94);
  a196=(a196+a94);
  a94=(a116*a105);
  a112=(a113*a121);
  a92=(a117*a105);
  a112=(a112-a92);
  a95=(a95*a112);
  a94=(a94+a95);
  a99=(a99-a97);
  a99=(a99+a118);
  a118=(a1*a99);
  a94=(a94-a118);
  a106=(a106*a126);
  a118=(a121+a106);
  a97=(a1*a118);
  a94=(a94+a97);
  a97=(a52*a94);
  a196=(a196+a97);
  a117=(a117*a98);
  a96=(a96*a112);
  a117=(a117+a96);
  a117=(a117+a99);
  a117=(a117-a118);
  a117=(a89*a117);
  a196=(a196+a117);
  a106=(a91*a106);
  a196=(a196+a106);
  a90=(a90*a112);
  a116=(a116*a121);
  a90=(a90-a116);
  a110=(a110*a126);
  a105=(a105+a110);
  a116=(a1*a105);
  a90=(a90+a116);
  a93=(a93*a127);
  a88=(a88*a195);
  a195=(a93-a88);
  a115=(a115*a119);
  a107=(a107*a126);
  a115=(a115+a107);
  a195=(a195+a115);
  a107=(a1*a195);
  a90=(a90+a107);
  a107=(a86*a90);
  a196=(a196-a107);
  a187=(a187*a196);
  a181=(a181-a187);
  a187=(a86*a123);
  a196=(a52*a120);
  a187=(a187-a196);
  a110=(a91*a110);
  a187=(a187+a110);
  a86=(a86*a94);
  a187=(a187+a86);
  a93=(a93-a88);
  a93=(a93+a115);
  a91=(a91*a93);
  a187=(a187+a91);
  a113=(a113*a98);
  a76=(a76*a112);
  a113=(a113+a76);
  a113=(a113-a105);
  a113=(a113-a195);
  a113=(a89*a113);
  a187=(a187+a113);
  a52=(a52*a90);
  a187=(a187+a52);
  a189=(a189*a187);
  a181=(a181+a189);
  a78=(a78/a57);
  a75=(a75?a78:0);
  a59=(a59/a57);
  a75=(a75-a59);
  a75=(a190*a75);
  a75=(a75/a38);
  a38=(a54*a75);
  a74=(a45*a74);
  a85=(a85/a77);
  a61=(a61*a80);
  a72=(a61*a72);
  a72=(a49*a72);
  a72=(a39*a72);
  a72=(a71?a72:0);
  a85=(a85+a72);
  a73=(a61/a73);
  a73=(a49*a73);
  a73=(a39*a73);
  a73=(a32*a73);
  a73=(-a73);
  a71=(a71?a73:0);
  a85=(a85+a71);
  a68=(a61*a68);
  a68=(a39*a68);
  a68=(a68*a208);
  a68=(-a68);
  a68=(a67?a68:0);
  a85=(a85+a68);
  a61=(a61/a69);
  a61=(a39*a61);
  a61=(a61*a66);
  a61=(-a61);
  a67=(a67?a61:0);
  a85=(a85+a67);
  a67=(a23*a85);
  a74=(a74+a67);
  a47=(a47*a74);
  a47=(a47/a202);
  a54=(a54*a47);
  a38=(a38-a54);
  a74=(a74/a65);
  a14=(a14*a74);
  a38=(a38-a14);
  a38=(a1*a38);
  a62=(a62*a74);
  a62=(a1*a62);
  a38=(a38+a62);
  a203=(a203*a38);
  a181=(a181-a203);
  a11=(a11*a75);
  a0=(a0*a47);
  a11=(a11-a0);
  a63=(a63*a74);
  a11=(a11+a63);
  a11=(a1*a11);
  a55=(a55*a74);
  a55=(a1*a55);
  a11=(a11-a55);
  a83=(a83*a11);
  a181=(a181+a83);
  a42=(a42/a13);
  a37=(a37?a42:0);
  a16=(a16/a13);
  a37=(a37-a16);
  a190=(a190*a37);
  a190=(a190/a7);
  a7=(a8*a190);
  a45=(a45*a36);
  a53=(a53/a41);
  a19=(a19*a51);
  a33=(a19*a33);
  a33=(a49*a33);
  a33=(a39*a33);
  a33=(a31?a33:0);
  a53=(a53+a33);
  a35=(a19/a35);
  a49=(a49*a35);
  a49=(a39*a49);
  a32=(a32*a49);
  a32=(-a32);
  a31=(a31?a32:0);
  a53=(a53+a31);
  a27=(a19*a27);
  a27=(a39*a27);
  a27=(a27*a135);
  a27=(-a27);
  a27=(a26?a27:0);
  a53=(a53+a27);
  a19=(a19/a29);
  a39=(a39*a19);
  a39=(a39*a24);
  a39=(-a39);
  a26=(a26?a39:0);
  a53=(a53+a26);
  a26=(a23*a53);
  a45=(a45+a26);
  a209=(a209*a45);
  a209=(a209/a79);
  a8=(a8*a209);
  a7=(a7-a8);
  a45=(a45/a5);
  a3=(a3*a45);
  a7=(a7-a3);
  a7=(a1*a7);
  a20=(a20*a45);
  a20=(a1*a20);
  a7=(a7+a20);
  a82=(a82*a7);
  a181=(a181-a82);
  a6=(a6*a190);
  a4=(a4*a209);
  a6=(a6-a4);
  a21=(a21*a45);
  a6=(a6+a21);
  a6=(a1*a6);
  a9=(a9*a45);
  a1=(a1*a9);
  a6=(a6-a1);
  a50=(a50*a6);
  a181=(a181+a50);
  if (res[0]!=0) res[0][9]=a181;
  a181=cos(a58);
  a50=(a89*a23);
  a23=(a23+a111);
  a6=(a23*a153);
  a1=(a111*a153);
  a6=(a6-a1);
  a1=(a50*a6);
  a9=cos(a2);
  a9=(a138*a9);
  a45=(a9*a153);
  a1=(a1-a45);
  a45=(a174*a9);
  a2=sin(a2);
  a138=(a138*a2);
  a2=(a186*a138);
  a45=(a45-a2);
  a2=(a174*a50);
  a21=(a2*a111);
  a45=(a45+a21);
  a21=(a23*a2);
  a45=(a45-a21);
  a21=(a176*a45);
  a1=(a1+a21);
  a21=(a138*a199);
  a1=(a1-a21);
  a1=(a181*a1);
  a21=sin(a58);
  a4=(a186*a50);
  a209=(a23*a4);
  a190=(a174*a138);
  a82=(a186*a9);
  a190=(a190+a82);
  a82=(a4*a111);
  a190=(a190+a82);
  a209=(a209-a190);
  a176=(a176*a209);
  a190=(a138*a153);
  a176=(a176-a190);
  a190=(a111*a199);
  a82=(a23*a199);
  a190=(a190-a82);
  a82=(a50*a190);
  a176=(a176+a82);
  a82=(a9*a199);
  a176=(a176+a82);
  a176=(a21*a176);
  a1=(a1-a176);
  a176=sin(a58);
  a82=(a140*a184);
  a7=(a84*a40);
  a82=(a82+a7);
  a7=(a84*a188);
  a82=(a82+a7);
  a7=(a140*a198);
  a82=(a82-a7);
  a82=(a176*a82);
  a1=(a1-a82);
  a82=cos(a58);
  a40=(a140*a40);
  a184=(a84*a184);
  a40=(a40-a184);
  a188=(a140*a188);
  a40=(a40+a188);
  a198=(a84*a198);
  a40=(a40+a198);
  a40=(a82*a40);
  a1=(a1+a40);
  a40=sin(a58);
  a198=(a87*a194);
  a188=(a56*a193);
  a198=(a198+a188);
  a188=(a56*a206);
  a198=(a198+a188);
  a188=(a87*a207);
  a198=(a198-a188);
  a198=(a40*a198);
  a1=(a1-a198);
  a58=cos(a58);
  a193=(a87*a193);
  a194=(a56*a194);
  a193=(a193-a194);
  a206=(a87*a206);
  a193=(a193+a206);
  a207=(a56*a207);
  a193=(a193+a207);
  a193=(a58*a193);
  a1=(a1+a193);
  if (res[0]!=0) res[0][10]=a1;
  a1=(a23*a145);
  a193=(a111*a145);
  a1=(a1-a193);
  a193=(a50*a1);
  a207=(a9*a145);
  a193=(a193-a207);
  a45=(a152*a45);
  a193=(a193+a45);
  a45=(a138*a166);
  a193=(a193-a45);
  a181=(a181*a193);
  a152=(a152*a209);
  a138=(a138*a145);
  a152=(a152-a138);
  a111=(a111*a166);
  a23=(a23*a166);
  a111=(a111-a23);
  a50=(a50*a111);
  a152=(a152+a50);
  a9=(a9*a166);
  a152=(a152+a9);
  a21=(a21*a152);
  a181=(a181-a21);
  a21=(a140*a171);
  a152=(a84*a161);
  a21=(a21+a152);
  a152=(a84*a164);
  a21=(a21+a152);
  a152=(a140*a142);
  a21=(a21-a152);
  a176=(a176*a21);
  a181=(a181-a176);
  a161=(a140*a161);
  a171=(a84*a171);
  a161=(a161-a171);
  a140=(a140*a164);
  a161=(a161+a140);
  a84=(a84*a142);
  a161=(a161+a84);
  a82=(a82*a161);
  a181=(a181+a82);
  a82=(a87*a120);
  a161=(a56*a123);
  a82=(a82+a161);
  a161=(a56*a94);
  a82=(a82+a161);
  a161=(a87*a90);
  a82=(a82-a161);
  a40=(a40*a82);
  a181=(a181-a40);
  a123=(a87*a123);
  a120=(a56*a120);
  a123=(a123-a120);
  a87=(a87*a94);
  a123=(a123+a87);
  a56=(a56*a90);
  a123=(a123+a56);
  a58=(a58*a123);
  a181=(a181+a58);
  if (res[0]!=0) res[0][11]=a181;
  a181=-1.;
  if (res[0]!=0) res[0][12]=a181;
  a58=(a4*a153);
  a123=(a2*a199);
  a58=(a58-a123);
  a6=(a186*a6);
  a190=(a174*a190);
  a6=(a6+a190);
  a6=(a89*a6);
  a6=(a58+a6);
  a190=(a159*a178);
  a6=(a6+a190);
  a190=(a109*a134);
  a6=(a6+a190);
  a81=(a64*a81);
  a6=(a6+a81);
  a46=(a22*a46);
  a6=(a6+a46);
  if (res[0]!=0) res[0][13]=a6;
  a6=(a4*a145);
  a46=(a2*a166);
  a6=(a6-a46);
  a186=(a186*a1);
  a174=(a174*a111);
  a186=(a186+a174);
  a89=(a89*a186);
  a89=(a6+a89);
  a186=(a159*a182);
  a89=(a89+a186);
  a186=(a109*a139);
  a89=(a89+a186);
  a85=(a64*a85);
  a89=(a89+a85);
  a53=(a22*a53);
  a89=(a89+a53);
  if (res[0]!=0) res[0][14]=a89;
  if (res[0]!=0) res[0][15]=a181;
  a153=(a4*a153);
  a58=(a58-a153);
  a199=(a2*a199);
  a58=(a58+a199);
  a178=(a158*a178);
  a58=(a58+a178);
  a134=(a108*a134);
  a58=(a58+a134);
  if (res[0]!=0) res[0][16]=a58;
  a4=(a4*a145);
  a6=(a6-a4);
  a2=(a2*a166);
  a6=(a6+a2);
  a182=(a158*a182);
  a6=(a6+a182);
  a139=(a108*a139);
  a6=(a6+a139);
  if (res[0]!=0) res[0][17]=a6;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a15;
  if (res[1]!=0) res[1][2]=a15;
  if (res[1]!=0) res[1][3]=a15;
  if (res[1]!=0) res[1][4]=a15;
  if (res[1]!=0) res[1][5]=a15;
  if (res[1]!=0) res[1][6]=a15;
  if (res[1]!=0) res[1][7]=a15;
  if (res[1]!=0) res[1][8]=a15;
  if (res[1]!=0) res[1][9]=a15;
  if (res[1]!=0) res[1][10]=a15;
  if (res[1]!=0) res[1][11]=a15;
  a15=2.7025639012821789e-01;
  a6=1.2330447799599942e+00;
  a139=1.4439765966454325e+00;
  a182=-2.7025639012821762e-01;
  a30=(a30*a17);
  a13=(a13*a30);
  a22=(a22*a13);
  a13=(a182*a22);
  a13=(a139*a13);
  a30=(a6*a13);
  a17=9.6278838983177628e-01;
  a22=(a17*a22);
  a30=(a30-a22);
  a30=(a15*a30);
  a30=(-a30);
  if (res[2]!=0) res[2][0]=a30;
  if (res[2]!=0) res[2][1]=a13;
  a70=(a70*a60);
  a57=(a57*a70);
  a64=(a64*a57);
  a57=(a182*a64);
  a57=(a139*a57);
  a70=(a6*a57);
  a64=(a17*a64);
  a70=(a70-a64);
  a70=(a15*a70);
  a70=(-a70);
  if (res[2]!=0) res[2][2]=a70;
  if (res[2]!=0) res[2][3]=a57;
  a122=(a122*a104);
  a101=(a101*a122);
  a109=(a109*a101);
  a122=(a182*a109);
  a104=9.6278838983177639e-01;
  a108=(a108*a101);
  a101=(a104*a108);
  a122=(a122+a101);
  a122=(a139*a122);
  a101=(a6*a122);
  a109=(a17*a109);
  a108=(a15*a108);
  a109=(a109+a108);
  a101=(a101-a109);
  a101=(a15*a101);
  a101=(-a101);
  if (res[2]!=0) res[2][4]=a101;
  if (res[2]!=0) res[2][5]=a122;
  a170=(a170*a155);
  a12=(a12*a170);
  a159=(a159*a12);
  a182=(a182*a159);
  a158=(a158*a12);
  a104=(a104*a158);
  a182=(a182+a104);
  a139=(a139*a182);
  a6=(a6*a139);
  a17=(a17*a159);
  a158=(a15*a158);
  a17=(a17+a158);
  a6=(a6-a17);
  a15=(a15*a6);
  a15=(-a15);
  if (res[2]!=0) res[2][6]=a15;
  if (res[2]!=0) res[2][7]=a139;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09521449_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
