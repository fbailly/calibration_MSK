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
  #define CASADI_PREFIX(ID) model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_ ## ID
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

/* model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8x8,18nz],o1[8x8,8nz],o2[8x4,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[2]? arg[2][0] : 0;
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
  a10=4.0000000000000001e-02;
  a11=(a7-a10);
  a12=8.7758256189037276e-01;
  a11=(a11/a12);
  a13=6.9999999999999996e-01;
  a14=(a11/a13);
  a15=1.;
  a14=(a14-a15);
  a16=casadi_sq(a14);
  a17=4.5000000000000001e-01;
  a16=(a16/a17);
  a16=(-a16);
  a16=exp(a16);
  a18=(a0*a16);
  a19=(a4+a5);
  a20=(a1*a8);
  a21=(a19*a20);
  a3=(a1*a3);
  a22=(a9*a3);
  a21=(a21-a22);
  a4=(a4+a5);
  a5=casadi_sq(a4);
  a22=casadi_sq(a9);
  a5=(a5+a22);
  a5=sqrt(a5);
  a21=(a21/a5);
  a22=arg[0]? arg[0][6] : 0;
  a23=(a21*a22);
  a24=0.;
  a25=(a23<=a24);
  a26=fabs(a23);
  a27=10.;
  a26=(a26/a27);
  a26=(a15-a26);
  a28=fabs(a23);
  a28=(a28/a27);
  a28=(a15+a28);
  a26=(a26/a28);
  a29=(a25?a26:0);
  a30=(!a25);
  a31=1.3300000000000001e+00;
  a32=(a31*a23);
  a32=(a32/a27);
  a33=-8.2500000000000004e-02;
  a32=(a32/a33);
  a32=(a15-a32);
  a34=(a23/a27);
  a34=(a34/a33);
  a34=(a15-a34);
  a32=(a32/a34);
  a35=(a30?a32:0);
  a29=(a29+a35);
  a35=(a18*a29);
  a36=(a10<a11);
  a11=(a11/a13);
  a11=(a11-a15);
  a11=(a27*a11);
  a11=exp(a11);
  a37=(a11-a15);
  a38=1.4741315910257660e+02;
  a37=(a37/a38);
  a37=(a36?a37:0);
  a35=(a35+a37);
  a37=1.0000000000000001e-01;
  a39=7.;
  a40=(a23/a39);
  a40=(a37*a40);
  a35=(a35+a40);
  a40=3.9024390243902418e-01;
  a41=(a40*a21);
  a42=(a35*a41);
  if (res[0]!=0) res[0][0]=a42;
  a42=-3.9024390243902396e-01;
  a43=(a42*a21);
  a44=(a35*a43);
  if (res[0]!=0) res[0][1]=a44;
  a44=arg[2]? arg[2][1] : 0;
  a45=sin(a2);
  a46=(a1*a45);
  a47=(a46+a1);
  a48=casadi_sq(a47);
  a49=cos(a2);
  a50=(a1*a49);
  a51=casadi_sq(a50);
  a48=(a48+a51);
  a48=sqrt(a48);
  a51=(a48-a10);
  a51=(a51/a12);
  a52=(a51/a13);
  a52=(a52-a15);
  a53=casadi_sq(a52);
  a53=(a53/a17);
  a53=(-a53);
  a53=exp(a53);
  a54=(a44*a53);
  a55=(a46+a1);
  a56=(a1*a49);
  a57=(a55*a56);
  a45=(a1*a45);
  a58=(a50*a45);
  a57=(a57-a58);
  a46=(a46+a1);
  a58=casadi_sq(a46);
  a59=casadi_sq(a50);
  a58=(a58+a59);
  a58=sqrt(a58);
  a57=(a57/a58);
  a59=(a57*a22);
  a60=(a59<=a24);
  a61=fabs(a59);
  a61=(a61/a27);
  a61=(a15-a61);
  a62=fabs(a59);
  a62=(a62/a27);
  a62=(a15+a62);
  a61=(a61/a62);
  a63=(a60?a61:0);
  a64=(!a60);
  a65=(a31*a59);
  a65=(a65/a27);
  a65=(a65/a33);
  a65=(a15-a65);
  a66=(a59/a27);
  a66=(a66/a33);
  a66=(a15-a66);
  a65=(a65/a66);
  a67=(a64?a65:0);
  a63=(a63+a67);
  a67=(a54*a63);
  a68=(a10<a51);
  a51=(a51/a13);
  a51=(a51-a15);
  a51=(a27*a51);
  a51=exp(a51);
  a69=(a51-a15);
  a69=(a69/a38);
  a69=(a68?a69:0);
  a67=(a67+a69);
  a69=(a59/a39);
  a69=(a37*a69);
  a67=(a67+a69);
  a69=(a40*a57);
  a70=(a67*a69);
  if (res[0]!=0) res[0][2]=a70;
  a70=(a42*a57);
  a71=(a67*a70);
  if (res[0]!=0) res[0][3]=a71;
  a71=arg[2]? arg[2][2] : 0;
  a72=arg[0]? arg[0][5] : 0;
  a73=sin(a72);
  a74=sin(a2);
  a75=(a73*a74);
  a76=cos(a72);
  a77=cos(a2);
  a78=(a76*a77);
  a75=(a75-a78);
  a78=(a1*a75);
  a79=1.2500000000000000e+00;
  a80=(a79*a74);
  a78=(a78-a80);
  a81=7.5000000000000000e-01;
  a82=(a81*a74);
  a83=(a78+a82);
  a84=casadi_sq(a83);
  a85=(a79*a77);
  a86=(a76*a74);
  a87=(a73*a77);
  a86=(a86+a87);
  a87=(a1*a86);
  a87=(a85-a87);
  a88=(a81*a77);
  a89=(a87-a88);
  a90=casadi_sq(a89);
  a84=(a84+a90);
  a84=sqrt(a84);
  a90=(a84-a10);
  a90=(a90/a12);
  a91=(a90/a13);
  a91=(a91-a15);
  a92=casadi_sq(a91);
  a92=(a92/a17);
  a92=(-a92);
  a92=exp(a92);
  a93=(a71*a92);
  a94=(a78+a82);
  a95=(a81*a77);
  a96=(a1*a86);
  a96=(a85-a96);
  a95=(a95-a96);
  a97=(a94*a95);
  a98=(a87-a88);
  a99=(a1*a75);
  a99=(a99-a80);
  a100=(a81*a74);
  a100=(a99+a100);
  a101=(a98*a100);
  a97=(a97+a101);
  a78=(a78+a82);
  a82=casadi_sq(a78);
  a87=(a87-a88);
  a88=casadi_sq(a87);
  a82=(a82+a88);
  a82=sqrt(a82);
  a97=(a97/a82);
  a88=(a97*a22);
  a101=(a73*a77);
  a102=(a76*a74);
  a101=(a101+a102);
  a102=(a75*a80);
  a103=(a86*a85);
  a102=(a102+a103);
  a103=(a101*a102);
  a104=(a101*a80);
  a105=(a76*a77);
  a106=(a73*a74);
  a105=(a105-a106);
  a106=(a105*a85);
  a104=(a104+a106);
  a106=(a75*a104);
  a103=(a103-a106);
  a103=(a103-a96);
  a96=(a94*a103);
  a106=(a86*a104);
  a107=(a105*a102);
  a106=(a106-a107);
  a106=(a106+a99);
  a99=(a98*a106);
  a96=(a96+a99);
  a96=(a96/a82);
  a99=arg[0]? arg[0][7] : 0;
  a107=(a96*a99);
  a88=(a88+a107);
  a107=(a88<=a24);
  a108=fabs(a88);
  a108=(a108/a27);
  a108=(a15-a108);
  a109=fabs(a88);
  a109=(a109/a27);
  a109=(a15+a109);
  a108=(a108/a109);
  a110=(a107?a108:0);
  a111=(!a107);
  a112=(a31*a88);
  a112=(a112/a27);
  a112=(a112/a33);
  a112=(a15-a112);
  a113=(a88/a27);
  a113=(a113/a33);
  a113=(a15-a113);
  a112=(a112/a113);
  a114=(a111?a112:0);
  a110=(a110+a114);
  a114=(a93*a110);
  a115=(a10<a90);
  a90=(a90/a13);
  a90=(a90-a15);
  a90=(a27*a90);
  a90=exp(a90);
  a116=(a90-a15);
  a116=(a116/a38);
  a116=(a115?a116:0);
  a114=(a114+a116);
  a116=(a88/a39);
  a116=(a37*a116);
  a114=(a114+a116);
  a116=-3.9024390243902440e-01;
  a117=(a116*a96);
  a118=(a40*a97);
  a117=(a117+a118);
  a118=(a114*a117);
  if (res[0]!=0) res[0][4]=a118;
  a118=1.3902439024390245e+00;
  a119=(a118*a96);
  a120=(a42*a97);
  a119=(a119+a120);
  a120=(a114*a119);
  if (res[0]!=0) res[0][5]=a120;
  a120=arg[2]? arg[2][3] : 0;
  a121=sin(a72);
  a122=sin(a2);
  a123=(a121*a122);
  a124=cos(a72);
  a125=cos(a2);
  a126=(a124*a125);
  a123=(a123-a126);
  a126=(a1*a123);
  a127=(a79*a122);
  a126=(a126-a127);
  a128=1.7500000000000000e+00;
  a129=(a128*a122);
  a130=(a126+a129);
  a131=casadi_sq(a130);
  a132=(a79*a125);
  a133=(a124*a122);
  a134=(a121*a125);
  a133=(a133+a134);
  a134=(a1*a133);
  a134=(a132-a134);
  a135=(a128*a125);
  a136=(a134-a135);
  a137=casadi_sq(a136);
  a131=(a131+a137);
  a131=sqrt(a131);
  a137=(a131-a10);
  a137=(a137/a12);
  a12=(a137/a13);
  a12=(a12-a15);
  a138=casadi_sq(a12);
  a138=(a138/a17);
  a138=(-a138);
  a138=exp(a138);
  a17=(a120*a138);
  a139=(a126+a129);
  a140=(a128*a125);
  a141=(a1*a133);
  a141=(a132-a141);
  a140=(a140-a141);
  a142=(a139*a140);
  a143=(a134-a135);
  a144=(a1*a123);
  a144=(a144-a127);
  a145=(a128*a122);
  a145=(a144+a145);
  a146=(a143*a145);
  a142=(a142+a146);
  a126=(a126+a129);
  a129=casadi_sq(a126);
  a134=(a134-a135);
  a135=casadi_sq(a134);
  a129=(a129+a135);
  a129=sqrt(a129);
  a142=(a142/a129);
  a135=(a142*a22);
  a146=(a121*a125);
  a147=(a124*a122);
  a146=(a146+a147);
  a147=(a123*a127);
  a148=(a133*a132);
  a147=(a147+a148);
  a148=(a146*a147);
  a149=(a146*a127);
  a150=(a124*a125);
  a151=(a121*a122);
  a150=(a150-a151);
  a151=(a150*a132);
  a149=(a149+a151);
  a151=(a123*a149);
  a148=(a148-a151);
  a148=(a148-a141);
  a141=(a139*a148);
  a151=(a133*a149);
  a152=(a150*a147);
  a151=(a151-a152);
  a151=(a151+a144);
  a144=(a143*a151);
  a141=(a141+a144);
  a141=(a141/a129);
  a144=(a141*a99);
  a135=(a135+a144);
  a24=(a135<=a24);
  a144=fabs(a135);
  a144=(a144/a27);
  a144=(a15-a144);
  a152=fabs(a135);
  a152=(a152/a27);
  a152=(a15+a152);
  a144=(a144/a152);
  a153=(a24?a144:0);
  a154=(!a24);
  a155=(a31*a135);
  a155=(a155/a27);
  a155=(a155/a33);
  a155=(a15-a155);
  a156=(a135/a27);
  a156=(a156/a33);
  a156=(a15-a156);
  a155=(a155/a156);
  a33=(a154?a155:0);
  a153=(a153+a33);
  a33=(a17*a153);
  a10=(a10<a137);
  a137=(a137/a13);
  a137=(a137-a15);
  a137=(a27*a137);
  a137=exp(a137);
  a13=(a137-a15);
  a13=(a13/a38);
  a13=(a10?a13:0);
  a33=(a33+a13);
  a39=(a135/a39);
  a39=(a37*a39);
  a33=(a33+a39);
  a39=(a116*a141);
  a13=(a40*a142);
  a39=(a39+a13);
  a13=(a33*a39);
  if (res[0]!=0) res[0][6]=a13;
  a13=(a118*a141);
  a38=(a42*a142);
  a13=(a13+a38);
  a38=(a33*a13);
  if (res[0]!=0) res[0][7]=a38;
  a38=cos(a2);
  a157=arg[0]? arg[0][3] : 0;
  a33=(a157*a33);
  a158=(a116*a33);
  a159=1.4285714285714285e-01;
  a39=(a157*a39);
  a160=(a37*a39);
  a160=(a159*a160);
  a161=-1.2121212121212121e+01;
  a162=(a17*a39);
  a155=(a155/a156);
  a163=(a162*a155);
  a163=(a161*a163);
  a163=(a37*a163);
  a163=(a154?a163:0);
  a160=(a160+a163);
  a163=(a162/a156);
  a163=(a161*a163);
  a163=(a37*a163);
  a163=(a31*a163);
  a163=(-a163);
  a163=(a154?a163:0);
  a160=(a160+a163);
  a144=(a144/a152);
  a163=(a162*a144);
  a163=(a37*a163);
  a164=casadi_sign(a135);
  a163=(a163*a164);
  a163=(-a163);
  a163=(a24?a163:0);
  a160=(a160+a163);
  a162=(a162/a152);
  a162=(a37*a162);
  a135=casadi_sign(a135);
  a162=(a162*a135);
  a162=(-a162);
  a162=(a24?a162:0);
  a160=(a160+a162);
  a162=(a99*a160);
  a158=(a158+a162);
  a162=(a158/a129);
  a163=(a143*a162);
  a165=(a133*a163);
  a166=(a139*a162);
  a167=(a123*a166);
  a165=(a165-a167);
  a167=(a127*a165);
  a168=(a147*a166);
  a167=(a167+a168);
  a168=(a124*a167);
  a169=(a132*a165);
  a170=(a147*a163);
  a169=(a169-a170);
  a170=(a121*a169);
  a168=(a168-a170);
  a170=(a40*a33);
  a171=(a22*a160);
  a170=(a170+a171);
  a171=(a170/a129);
  a172=(a143*a171);
  a173=(a128*a172);
  a168=(a168+a173);
  a173=(a149*a163);
  a174=(a146*a166);
  a175=(a150*a163);
  a174=(a174-a175);
  a175=(a132*a174);
  a173=(a173+a175);
  a136=(a136+a136);
  a175=1.1394939273245490e+00;
  a176=1.4285714285714286e+00;
  a177=6.7836549063042314e-03;
  a178=(a177*a39);
  a178=(a178*a137);
  a178=(a27*a178);
  a178=(a176*a178);
  a178=(a10?a178:0);
  a12=(a12+a12);
  a179=2.2222222222222223e+00;
  a39=(a153*a39);
  a39=(a120*a39);
  a39=(a138*a39);
  a39=(a179*a39);
  a39=(a12*a39);
  a39=(a176*a39);
  a178=(a178-a39);
  a178=(a175*a178);
  a131=(a131+a131);
  a178=(a178/a131);
  a39=(a136*a178);
  a134=(a134+a134);
  a180=(a141/a129);
  a158=(a180*a158);
  a181=(a142/a129);
  a170=(a181*a170);
  a158=(a158+a170);
  a170=(a129+a129);
  a158=(a158/a170);
  a182=(a134*a158);
  a183=(a39-a182);
  a184=(a151*a162);
  a185=(a145*a171);
  a184=(a184+a185);
  a183=(a183+a184);
  a185=(a1*a183);
  a173=(a173-a185);
  a185=(a139*a171);
  a186=(a166+a185);
  a187=(a1*a186);
  a173=(a173+a187);
  a187=(a124*a173);
  a168=(a168+a187);
  a130=(a130+a130);
  a178=(a130*a178);
  a126=(a126+a126);
  a158=(a126*a158);
  a187=(a178-a158);
  a162=(a148*a162);
  a171=(a140*a171);
  a162=(a162+a171);
  a187=(a187+a162);
  a187=(a128*a187);
  a168=(a168+a187);
  a187=(a146*a165);
  a171=(a123*a174);
  a187=(a187+a171);
  a163=(a163+a172);
  a187=(a187-a163);
  a178=(a178-a158);
  a178=(a178+a162);
  a187=(a187-a178);
  a187=(a79*a187);
  a168=(a168+a187);
  a187=(a127*a174);
  a166=(a149*a166);
  a187=(a187-a166);
  a163=(a1*a163);
  a187=(a187+a163);
  a178=(a1*a178);
  a187=(a187+a178);
  a178=(a121*a187);
  a168=(a168+a178);
  a168=(a38*a168);
  a178=cos(a2);
  a163=9.8100000000000005e+00;
  a166=cos(a72);
  a162=4.8780487804878025e-01;
  a158=(a162*a166);
  a172=(a166*a158);
  a171=sin(a72);
  a188=(a162*a171);
  a189=(a171*a188);
  a172=(a172+a189);
  a172=(a163*a172);
  a172=(a178*a172);
  a189=sin(a2);
  a190=(a166*a188);
  a191=(a171*a158);
  a190=(a190-a191);
  a190=(a163*a190);
  a190=(a189*a190);
  a172=(a172+a190);
  a190=sin(a2);
  a191=(a124*a169);
  a192=(a121*a167);
  a191=(a191+a192);
  a182=(a182-a39);
  a182=(a182-a184);
  a182=(a128*a182);
  a191=(a191+a182);
  a182=(a121*a173);
  a191=(a191+a182);
  a165=(a150*a165);
  a174=(a133*a174);
  a165=(a165+a174);
  a165=(a165+a183);
  a165=(a165-a186);
  a165=(a79*a165);
  a191=(a191+a165);
  a185=(a128*a185);
  a191=(a191+a185);
  a185=(a124*a187);
  a191=(a191-a185);
  a191=(a190*a191);
  a172=(a172+a191);
  a168=(a168-a172);
  a172=sin(a2);
  a191=arg[0]? arg[0][2] : 0;
  a114=(a191*a114);
  a116=(a116*a114);
  a117=(a191*a117);
  a185=(a37*a117);
  a185=(a159*a185);
  a165=(a93*a117);
  a112=(a112/a113);
  a186=(a165*a112);
  a186=(a161*a186);
  a186=(a37*a186);
  a186=(a111?a186:0);
  a185=(a185+a186);
  a186=(a165/a113);
  a186=(a161*a186);
  a186=(a37*a186);
  a186=(a31*a186);
  a186=(-a186);
  a186=(a111?a186:0);
  a185=(a185+a186);
  a108=(a108/a109);
  a186=(a165*a108);
  a186=(a37*a186);
  a183=casadi_sign(a88);
  a186=(a186*a183);
  a186=(-a186);
  a186=(a107?a186:0);
  a185=(a185+a186);
  a165=(a165/a109);
  a165=(a37*a165);
  a88=casadi_sign(a88);
  a165=(a165*a88);
  a165=(-a165);
  a165=(a107?a165:0);
  a185=(a185+a165);
  a165=(a99*a185);
  a116=(a116+a165);
  a165=(a116/a82);
  a186=(a98*a165);
  a174=(a86*a186);
  a182=(a94*a165);
  a184=(a75*a182);
  a174=(a174-a184);
  a184=(a85*a174);
  a39=(a102*a186);
  a184=(a184-a39);
  a39=(a76*a184);
  a192=(a80*a174);
  a193=(a102*a182);
  a192=(a192+a193);
  a193=(a73*a192);
  a39=(a39+a193);
  a87=(a87+a87);
  a193=(a96/a82);
  a116=(a193*a116);
  a194=(a97/a82);
  a195=(a40*a114);
  a196=(a22*a185);
  a195=(a195+a196);
  a196=(a194*a195);
  a116=(a116+a196);
  a196=(a82+a82);
  a116=(a116/a196);
  a197=(a87*a116);
  a89=(a89+a89);
  a198=(a177*a117);
  a198=(a198*a90);
  a198=(a27*a198);
  a198=(a176*a198);
  a198=(a115?a198:0);
  a91=(a91+a91);
  a117=(a110*a117);
  a117=(a71*a117);
  a117=(a92*a117);
  a117=(a179*a117);
  a117=(a91*a117);
  a117=(a176*a117);
  a198=(a198-a117);
  a198=(a175*a198);
  a84=(a84+a84);
  a198=(a198/a84);
  a117=(a89*a198);
  a199=(a197-a117);
  a200=(a106*a165);
  a195=(a195/a82);
  a201=(a100*a195);
  a200=(a200+a201);
  a199=(a199-a200);
  a199=(a81*a199);
  a39=(a39+a199);
  a199=(a104*a186);
  a201=(a101*a182);
  a202=(a105*a186);
  a201=(a201-a202);
  a202=(a85*a201);
  a199=(a199+a202);
  a117=(a117-a197);
  a117=(a117+a200);
  a200=(a1*a117);
  a199=(a199-a200);
  a200=(a94*a195);
  a197=(a182+a200);
  a202=(a1*a197);
  a199=(a199+a202);
  a202=(a73*a199);
  a39=(a39+a202);
  a202=(a105*a174);
  a203=(a86*a201);
  a202=(a202+a203);
  a202=(a202+a117);
  a202=(a202-a197);
  a202=(a79*a202);
  a39=(a39+a202);
  a200=(a81*a200);
  a39=(a39+a200);
  a200=(a80*a201);
  a182=(a104*a182);
  a200=(a200-a182);
  a182=(a98*a195);
  a186=(a186+a182);
  a202=(a1*a186);
  a200=(a200+a202);
  a83=(a83+a83);
  a198=(a83*a198);
  a78=(a78+a78);
  a116=(a78*a116);
  a202=(a198-a116);
  a165=(a103*a165);
  a195=(a95*a195);
  a165=(a165+a195);
  a202=(a202+a165);
  a195=(a1*a202);
  a200=(a200+a195);
  a195=(a76*a200);
  a39=(a39-a195);
  a39=(a172*a39);
  a168=(a168-a39);
  a39=cos(a2);
  a195=(a76*a192);
  a197=(a73*a184);
  a195=(a195-a197);
  a182=(a81*a182);
  a195=(a195+a182);
  a182=(a76*a199);
  a195=(a195+a182);
  a198=(a198-a116);
  a198=(a198+a165);
  a198=(a81*a198);
  a195=(a195+a198);
  a174=(a101*a174);
  a201=(a75*a201);
  a174=(a174+a201);
  a174=(a174-a186);
  a174=(a174-a202);
  a174=(a79*a174);
  a195=(a195+a174);
  a174=(a73*a200);
  a195=(a195+a174);
  a195=(a39*a195);
  a168=(a168+a195);
  a195=sin(a2);
  a174=arg[0]? arg[0][1] : 0;
  a69=(a174*a69);
  a202=(a177*a69);
  a202=(a202*a51);
  a202=(a27*a202);
  a202=(a176*a202);
  a202=(a68?a202:0);
  a52=(a52+a52);
  a186=(a63*a69);
  a186=(a44*a186);
  a186=(a53*a186);
  a186=(a179*a186);
  a186=(a52*a186);
  a186=(a176*a186);
  a202=(a202-a186);
  a202=(a175*a202);
  a48=(a48+a48);
  a202=(a202/a48);
  a186=(a49*a202);
  a201=(a57/a58);
  a67=(a174*a67);
  a198=(a40*a67);
  a165=(a37*a69);
  a165=(a159*a165);
  a69=(a54*a69);
  a65=(a65/a66);
  a116=(a69*a65);
  a116=(a161*a116);
  a116=(a37*a116);
  a116=(a64?a116:0);
  a165=(a165+a116);
  a116=(a69/a66);
  a116=(a161*a116);
  a116=(a37*a116);
  a116=(a31*a116);
  a116=(-a116);
  a116=(a64?a116:0);
  a165=(a165+a116);
  a61=(a61/a62);
  a116=(a69*a61);
  a116=(a37*a116);
  a182=casadi_sign(a59);
  a116=(a116*a182);
  a116=(-a116);
  a116=(a60?a116:0);
  a165=(a165+a116);
  a69=(a69/a62);
  a69=(a37*a69);
  a59=casadi_sign(a59);
  a69=(a69*a59);
  a69=(-a69);
  a69=(a60?a69:0);
  a165=(a165+a69);
  a69=(a22*a165);
  a198=(a198+a69);
  a69=(a201*a198);
  a116=(a58+a58);
  a69=(a69/a116);
  a197=(a49*a69);
  a186=(a186-a197);
  a198=(a198/a58);
  a197=(a45*a198);
  a186=(a186-a197);
  a186=(a1*a186);
  a197=(a55*a198);
  a197=(a1*a197);
  a186=(a186+a197);
  a186=(a195*a186);
  a168=(a168-a186);
  a186=cos(a2);
  a47=(a47+a47);
  a202=(a47*a202);
  a46=(a46+a46);
  a69=(a46*a69);
  a202=(a202-a69);
  a69=(a56*a198);
  a202=(a202+a69);
  a202=(a1*a202);
  a198=(a50*a198);
  a198=(a1*a198);
  a202=(a202-a198);
  a202=(a186*a202);
  a168=(a168+a202);
  a202=sin(a2);
  a198=arg[0]? arg[0][0] : 0;
  a41=(a198*a41);
  a69=(a177*a41);
  a69=(a69*a11);
  a69=(a27*a69);
  a69=(a176*a69);
  a69=(a36?a69:0);
  a14=(a14+a14);
  a197=(a29*a41);
  a197=(a0*a197);
  a197=(a16*a197);
  a197=(a179*a197);
  a197=(a14*a197);
  a197=(a176*a197);
  a69=(a69-a197);
  a69=(a175*a69);
  a7=(a7+a7);
  a69=(a69/a7);
  a197=(a8*a69);
  a117=(a21/a5);
  a35=(a198*a35);
  a40=(a40*a35);
  a203=(a37*a41);
  a203=(a159*a203);
  a41=(a18*a41);
  a32=(a32/a34);
  a204=(a41*a32);
  a204=(a161*a204);
  a204=(a37*a204);
  a204=(a30?a204:0);
  a203=(a203+a204);
  a204=(a41/a34);
  a204=(a161*a204);
  a204=(a37*a204);
  a204=(a31*a204);
  a204=(-a204);
  a204=(a30?a204:0);
  a203=(a203+a204);
  a26=(a26/a28);
  a204=(a41*a26);
  a204=(a37*a204);
  a205=casadi_sign(a23);
  a204=(a204*a205);
  a204=(-a204);
  a204=(a25?a204:0);
  a203=(a203+a204);
  a41=(a41/a28);
  a41=(a37*a41);
  a23=casadi_sign(a23);
  a41=(a41*a23);
  a41=(-a41);
  a41=(a25?a41:0);
  a203=(a203+a41);
  a41=(a22*a203);
  a40=(a40+a41);
  a41=(a117*a40);
  a204=(a5+a5);
  a41=(a41/a204);
  a206=(a8*a41);
  a197=(a197-a206);
  a40=(a40/a5);
  a206=(a3*a40);
  a197=(a197-a206);
  a197=(a1*a197);
  a206=(a19*a40);
  a206=(a1*a206);
  a197=(a197+a206);
  a197=(a202*a197);
  a168=(a168-a197);
  a197=cos(a2);
  a6=(a6+a6);
  a69=(a6*a69);
  a4=(a4+a4);
  a41=(a4*a41);
  a69=(a69-a41);
  a41=(a20*a40);
  a69=(a69+a41);
  a69=(a1*a69);
  a40=(a9*a40);
  a40=(a1*a40);
  a69=(a69-a40);
  a69=(a197*a69);
  a168=(a168+a69);
  if (res[0]!=0) res[0][8]=a168;
  a168=(a118*a33);
  a13=(a157*a13);
  a69=(a37*a13);
  a69=(a159*a69);
  a17=(a17*a13);
  a155=(a17*a155);
  a155=(a161*a155);
  a155=(a37*a155);
  a155=(a154?a155:0);
  a69=(a69+a155);
  a156=(a17/a156);
  a156=(a161*a156);
  a156=(a37*a156);
  a156=(a31*a156);
  a156=(-a156);
  a154=(a154?a156:0);
  a69=(a69+a154);
  a144=(a17*a144);
  a144=(a37*a144);
  a144=(a144*a164);
  a144=(-a144);
  a144=(a24?a144:0);
  a69=(a69+a144);
  a17=(a17/a152);
  a17=(a37*a17);
  a17=(a17*a135);
  a17=(-a17);
  a24=(a24?a17:0);
  a69=(a69+a24);
  a24=(a99*a69);
  a168=(a168+a24);
  a24=(a168/a129);
  a17=(a143*a24);
  a135=(a133*a17);
  a152=(a139*a24);
  a144=(a123*a152);
  a135=(a135-a144);
  a144=(a127*a135);
  a164=(a147*a152);
  a144=(a144+a164);
  a164=(a124*a144);
  a154=(a132*a135);
  a147=(a147*a17);
  a154=(a154-a147);
  a147=(a121*a154);
  a164=(a164-a147);
  a33=(a42*a33);
  a147=(a22*a69);
  a33=(a33+a147);
  a129=(a33/a129);
  a143=(a143*a129);
  a147=(a128*a143);
  a164=(a164+a147);
  a147=(a149*a17);
  a156=(a146*a152);
  a155=(a150*a17);
  a156=(a156-a155);
  a132=(a132*a156);
  a147=(a147+a132);
  a132=(a177*a13);
  a132=(a132*a137);
  a132=(a27*a132);
  a132=(a176*a132);
  a10=(a10?a132:0);
  a13=(a153*a13);
  a120=(a120*a13);
  a120=(a138*a120);
  a120=(a179*a120);
  a12=(a12*a120);
  a12=(a176*a12);
  a10=(a10-a12);
  a10=(a175*a10);
  a10=(a10/a131);
  a136=(a136*a10);
  a180=(a180*a168);
  a181=(a181*a33);
  a180=(a180+a181);
  a180=(a180/a170);
  a134=(a134*a180);
  a170=(a136-a134);
  a151=(a151*a24);
  a145=(a145*a129);
  a151=(a151+a145);
  a170=(a170+a151);
  a145=(a1*a170);
  a147=(a147-a145);
  a139=(a139*a129);
  a145=(a152+a139);
  a181=(a1*a145);
  a147=(a147+a181);
  a181=(a124*a147);
  a164=(a164+a181);
  a130=(a130*a10);
  a126=(a126*a180);
  a180=(a130-a126);
  a148=(a148*a24);
  a140=(a140*a129);
  a148=(a148+a140);
  a180=(a180+a148);
  a180=(a128*a180);
  a164=(a164+a180);
  a146=(a146*a135);
  a123=(a123*a156);
  a146=(a146+a123);
  a17=(a17+a143);
  a146=(a146-a17);
  a130=(a130-a126);
  a130=(a130+a148);
  a146=(a146-a130);
  a146=(a79*a146);
  a164=(a164+a146);
  a127=(a127*a156);
  a149=(a149*a152);
  a127=(a127-a149);
  a17=(a1*a17);
  a127=(a127+a17);
  a130=(a1*a130);
  a127=(a127+a130);
  a130=(a121*a127);
  a164=(a164+a130);
  a38=(a38*a164);
  a164=-4.8780487804877992e-01;
  a130=(a164*a166);
  a17=(a166*a130);
  a149=(a164*a171);
  a152=(a171*a149);
  a17=(a17+a152);
  a17=(a163*a17);
  a178=(a178*a17);
  a17=(a166*a149);
  a152=(a171*a130);
  a17=(a17-a152);
  a17=(a163*a17);
  a189=(a189*a17);
  a178=(a178+a189);
  a189=(a124*a154);
  a17=(a121*a144);
  a189=(a189+a17);
  a134=(a134-a136);
  a134=(a134-a151);
  a134=(a128*a134);
  a189=(a189+a134);
  a121=(a121*a147);
  a189=(a189+a121);
  a150=(a150*a135);
  a133=(a133*a156);
  a150=(a150+a133);
  a150=(a150+a170);
  a150=(a150-a145);
  a150=(a79*a150);
  a189=(a189+a150);
  a128=(a128*a139);
  a189=(a189+a128);
  a124=(a124*a127);
  a189=(a189-a124);
  a190=(a190*a189);
  a178=(a178+a190);
  a38=(a38-a178);
  a118=(a118*a114);
  a119=(a191*a119);
  a178=(a37*a119);
  a178=(a159*a178);
  a93=(a93*a119);
  a112=(a93*a112);
  a112=(a161*a112);
  a112=(a37*a112);
  a112=(a111?a112:0);
  a178=(a178+a112);
  a113=(a93/a113);
  a113=(a161*a113);
  a113=(a37*a113);
  a113=(a31*a113);
  a113=(-a113);
  a111=(a111?a113:0);
  a178=(a178+a111);
  a108=(a93*a108);
  a108=(a37*a108);
  a108=(a108*a183);
  a108=(-a108);
  a108=(a107?a108:0);
  a178=(a178+a108);
  a93=(a93/a109);
  a93=(a37*a93);
  a93=(a93*a88);
  a93=(-a93);
  a107=(a107?a93:0);
  a178=(a178+a107);
  a107=(a99*a178);
  a118=(a118+a107);
  a107=(a118/a82);
  a93=(a98*a107);
  a88=(a86*a93);
  a109=(a94*a107);
  a108=(a75*a109);
  a88=(a88-a108);
  a108=(a85*a88);
  a183=(a102*a93);
  a108=(a108-a183);
  a183=(a76*a108);
  a111=(a80*a88);
  a102=(a102*a109);
  a111=(a111+a102);
  a102=(a73*a111);
  a183=(a183+a102);
  a193=(a193*a118);
  a114=(a42*a114);
  a118=(a22*a178);
  a114=(a114+a118);
  a194=(a194*a114);
  a193=(a193+a194);
  a193=(a193/a196);
  a87=(a87*a193);
  a196=(a177*a119);
  a196=(a196*a90);
  a196=(a27*a196);
  a196=(a176*a196);
  a115=(a115?a196:0);
  a119=(a110*a119);
  a71=(a71*a119);
  a71=(a92*a71);
  a71=(a179*a71);
  a91=(a91*a71);
  a91=(a176*a91);
  a115=(a115-a91);
  a115=(a175*a115);
  a115=(a115/a84);
  a89=(a89*a115);
  a84=(a87-a89);
  a106=(a106*a107);
  a114=(a114/a82);
  a100=(a100*a114);
  a106=(a106+a100);
  a84=(a84-a106);
  a84=(a81*a84);
  a183=(a183+a84);
  a84=(a104*a93);
  a100=(a101*a109);
  a82=(a105*a93);
  a100=(a100-a82);
  a85=(a85*a100);
  a84=(a84+a85);
  a89=(a89-a87);
  a89=(a89+a106);
  a106=(a1*a89);
  a84=(a84-a106);
  a94=(a94*a114);
  a106=(a109+a94);
  a87=(a1*a106);
  a84=(a84+a87);
  a87=(a73*a84);
  a183=(a183+a87);
  a105=(a105*a88);
  a86=(a86*a100);
  a105=(a105+a86);
  a105=(a105+a89);
  a105=(a105-a106);
  a105=(a79*a105);
  a183=(a183+a105);
  a94=(a81*a94);
  a183=(a183+a94);
  a80=(a80*a100);
  a104=(a104*a109);
  a80=(a80-a104);
  a98=(a98*a114);
  a93=(a93+a98);
  a104=(a1*a93);
  a80=(a80+a104);
  a83=(a83*a115);
  a78=(a78*a193);
  a193=(a83-a78);
  a103=(a103*a107);
  a95=(a95*a114);
  a103=(a103+a95);
  a193=(a193+a103);
  a95=(a1*a193);
  a80=(a80+a95);
  a95=(a76*a80);
  a183=(a183-a95);
  a172=(a172*a183);
  a38=(a38-a172);
  a172=(a76*a111);
  a183=(a73*a108);
  a172=(a172-a183);
  a98=(a81*a98);
  a172=(a172+a98);
  a76=(a76*a84);
  a172=(a172+a76);
  a83=(a83-a78);
  a83=(a83+a103);
  a81=(a81*a83);
  a172=(a172+a81);
  a101=(a101*a88);
  a75=(a75*a100);
  a101=(a101+a75);
  a101=(a101-a93);
  a101=(a101-a193);
  a101=(a79*a101);
  a172=(a172+a101);
  a73=(a73*a80);
  a172=(a172+a73);
  a39=(a39*a172);
  a38=(a38+a39);
  a70=(a174*a70);
  a39=(a177*a70);
  a39=(a39*a51);
  a39=(a27*a39);
  a39=(a176*a39);
  a68=(a68?a39:0);
  a39=(a63*a70);
  a44=(a44*a39);
  a44=(a53*a44);
  a44=(a179*a44);
  a52=(a52*a44);
  a52=(a176*a52);
  a68=(a68-a52);
  a68=(a175*a68);
  a68=(a68/a48);
  a48=(a49*a68);
  a67=(a42*a67);
  a52=(a37*a70);
  a52=(a159*a52);
  a54=(a54*a70);
  a65=(a54*a65);
  a65=(a161*a65);
  a65=(a37*a65);
  a65=(a64?a65:0);
  a52=(a52+a65);
  a66=(a54/a66);
  a66=(a161*a66);
  a66=(a37*a66);
  a66=(a31*a66);
  a66=(-a66);
  a64=(a64?a66:0);
  a52=(a52+a64);
  a61=(a54*a61);
  a61=(a37*a61);
  a61=(a61*a182);
  a61=(-a61);
  a61=(a60?a61:0);
  a52=(a52+a61);
  a54=(a54/a62);
  a54=(a37*a54);
  a54=(a54*a59);
  a54=(-a54);
  a60=(a60?a54:0);
  a52=(a52+a60);
  a60=(a22*a52);
  a67=(a67+a60);
  a201=(a201*a67);
  a201=(a201/a116);
  a49=(a49*a201);
  a48=(a48-a49);
  a67=(a67/a58);
  a45=(a45*a67);
  a48=(a48-a45);
  a48=(a1*a48);
  a55=(a55*a67);
  a55=(a1*a55);
  a48=(a48+a55);
  a195=(a195*a48);
  a38=(a38-a195);
  a47=(a47*a68);
  a46=(a46*a201);
  a47=(a47-a46);
  a56=(a56*a67);
  a47=(a47+a56);
  a47=(a1*a47);
  a50=(a50*a67);
  a50=(a1*a50);
  a47=(a47-a50);
  a186=(a186*a47);
  a38=(a38+a186);
  a43=(a198*a43);
  a177=(a177*a43);
  a177=(a177*a11);
  a27=(a27*a177);
  a27=(a176*a27);
  a36=(a36?a27:0);
  a27=(a29*a43);
  a0=(a0*a27);
  a0=(a16*a0);
  a179=(a179*a0);
  a14=(a14*a179);
  a176=(a176*a14);
  a36=(a36-a176);
  a175=(a175*a36);
  a175=(a175/a7);
  a7=(a8*a175);
  a42=(a42*a35);
  a35=(a37*a43);
  a159=(a159*a35);
  a18=(a18*a43);
  a32=(a18*a32);
  a32=(a161*a32);
  a32=(a37*a32);
  a32=(a30?a32:0);
  a159=(a159+a32);
  a34=(a18/a34);
  a161=(a161*a34);
  a161=(a37*a161);
  a31=(a31*a161);
  a31=(-a31);
  a30=(a30?a31:0);
  a159=(a159+a30);
  a26=(a18*a26);
  a26=(a37*a26);
  a26=(a26*a205);
  a26=(-a26);
  a26=(a25?a26:0);
  a159=(a159+a26);
  a18=(a18/a28);
  a37=(a37*a18);
  a37=(a37*a23);
  a37=(-a37);
  a25=(a25?a37:0);
  a159=(a159+a25);
  a25=(a22*a159);
  a42=(a42+a25);
  a117=(a117*a42);
  a117=(a117/a204);
  a8=(a8*a117);
  a7=(a7-a8);
  a42=(a42/a5);
  a3=(a3*a42);
  a7=(a7-a3);
  a7=(a1*a7);
  a19=(a19*a42);
  a19=(a1*a19);
  a7=(a7+a19);
  a202=(a202*a7);
  a38=(a38-a202);
  a6=(a6*a175);
  a4=(a4*a117);
  a6=(a6-a4);
  a20=(a20*a42);
  a6=(a6+a20);
  a6=(a1*a6);
  a9=(a9*a42);
  a1=(a1*a9);
  a6=(a6-a1);
  a197=(a197*a6);
  a38=(a38+a197);
  if (res[0]!=0) res[0][9]=a38;
  a38=cos(a72);
  a197=(a79*a22);
  a22=(a22+a99);
  a6=(a22*a158);
  a1=(a99*a158);
  a6=(a6-a1);
  a1=(a197*a6);
  a9=cos(a2);
  a9=(a163*a9);
  a42=(a9*a158);
  a1=(a1-a42);
  a42=(a166*a9);
  a2=sin(a2);
  a163=(a163*a2);
  a2=(a171*a163);
  a42=(a42-a2);
  a2=(a166*a197);
  a20=(a2*a99);
  a42=(a42+a20);
  a20=(a22*a2);
  a42=(a42-a20);
  a20=(a162*a42);
  a1=(a1+a20);
  a20=(a163*a188);
  a1=(a1-a20);
  a1=(a38*a1);
  a20=sin(a72);
  a4=(a171*a197);
  a117=(a22*a4);
  a175=(a166*a163);
  a202=(a171*a9);
  a175=(a175+a202);
  a202=(a4*a99);
  a175=(a175+a202);
  a117=(a117-a175);
  a162=(a162*a117);
  a175=(a163*a158);
  a162=(a162-a175);
  a175=(a99*a188);
  a202=(a22*a188);
  a175=(a175-a202);
  a202=(a197*a175);
  a162=(a162+a202);
  a202=(a9*a188);
  a162=(a162+a202);
  a162=(a20*a162);
  a1=(a1-a162);
  a162=sin(a72);
  a202=(a125*a169);
  a7=(a122*a167);
  a202=(a202+a7);
  a7=(a122*a173);
  a202=(a202+a7);
  a7=(a125*a187);
  a202=(a202-a7);
  a202=(a162*a202);
  a1=(a1-a202);
  a202=cos(a72);
  a167=(a125*a167);
  a169=(a122*a169);
  a167=(a167-a169);
  a173=(a125*a173);
  a167=(a167+a173);
  a187=(a122*a187);
  a167=(a167+a187);
  a167=(a202*a167);
  a1=(a1+a167);
  a167=sin(a72);
  a187=(a77*a184);
  a173=(a74*a192);
  a187=(a187+a173);
  a173=(a74*a199);
  a187=(a187+a173);
  a173=(a77*a200);
  a187=(a187-a173);
  a187=(a167*a187);
  a1=(a1-a187);
  a72=cos(a72);
  a192=(a77*a192);
  a184=(a74*a184);
  a192=(a192-a184);
  a199=(a77*a199);
  a192=(a192+a199);
  a200=(a74*a200);
  a192=(a192+a200);
  a192=(a72*a192);
  a1=(a1+a192);
  if (res[0]!=0) res[0][10]=a1;
  a1=(a22*a130);
  a192=(a99*a130);
  a1=(a1-a192);
  a192=(a197*a1);
  a200=(a9*a130);
  a192=(a192-a200);
  a42=(a164*a42);
  a192=(a192+a42);
  a42=(a163*a149);
  a192=(a192-a42);
  a38=(a38*a192);
  a164=(a164*a117);
  a163=(a163*a130);
  a164=(a164-a163);
  a99=(a99*a149);
  a22=(a22*a149);
  a99=(a99-a22);
  a197=(a197*a99);
  a164=(a164+a197);
  a9=(a9*a149);
  a164=(a164+a9);
  a20=(a20*a164);
  a38=(a38-a20);
  a20=(a125*a154);
  a164=(a122*a144);
  a20=(a20+a164);
  a164=(a122*a147);
  a20=(a20+a164);
  a164=(a125*a127);
  a20=(a20-a164);
  a162=(a162*a20);
  a38=(a38-a162);
  a144=(a125*a144);
  a154=(a122*a154);
  a144=(a144-a154);
  a125=(a125*a147);
  a144=(a144+a125);
  a122=(a122*a127);
  a144=(a144+a122);
  a202=(a202*a144);
  a38=(a38+a202);
  a202=(a77*a108);
  a144=(a74*a111);
  a202=(a202+a144);
  a144=(a74*a84);
  a202=(a202+a144);
  a144=(a77*a80);
  a202=(a202-a144);
  a167=(a167*a202);
  a38=(a38-a167);
  a111=(a77*a111);
  a108=(a74*a108);
  a111=(a111-a108);
  a77=(a77*a84);
  a111=(a111+a77);
  a74=(a74*a80);
  a111=(a111+a74);
  a72=(a72*a111);
  a38=(a38+a72);
  if (res[0]!=0) res[0][11]=a38;
  a38=-1.;
  if (res[0]!=0) res[0][12]=a38;
  a72=(a4*a158);
  a111=(a2*a188);
  a72=(a72-a111);
  a6=(a171*a6);
  a175=(a166*a175);
  a6=(a6+a175);
  a6=(a79*a6);
  a6=(a72+a6);
  a175=(a142*a160);
  a6=(a6+a175);
  a175=(a97*a185);
  a6=(a6+a175);
  a165=(a57*a165);
  a6=(a6+a165);
  a203=(a21*a203);
  a6=(a6+a203);
  if (res[0]!=0) res[0][13]=a6;
  a6=(a4*a130);
  a203=(a2*a149);
  a6=(a6-a203);
  a171=(a171*a1);
  a166=(a166*a99);
  a171=(a171+a166);
  a79=(a79*a171);
  a79=(a6+a79);
  a171=(a142*a69);
  a79=(a79+a171);
  a171=(a97*a178);
  a79=(a79+a171);
  a52=(a57*a52);
  a79=(a79+a52);
  a159=(a21*a159);
  a79=(a79+a159);
  if (res[0]!=0) res[0][14]=a79;
  if (res[0]!=0) res[0][15]=a38;
  a158=(a4*a158);
  a72=(a72-a158);
  a188=(a2*a188);
  a72=(a72+a188);
  a160=(a141*a160);
  a72=(a72+a160);
  a185=(a96*a185);
  a72=(a72+a185);
  if (res[0]!=0) res[0][16]=a72;
  a4=(a4*a130);
  a6=(a6-a4);
  a2=(a2*a149);
  a6=(a6+a2);
  a69=(a141*a69);
  a6=(a6+a69);
  a178=(a96*a178);
  a6=(a6+a178);
  if (res[0]!=0) res[0][17]=a6;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a15;
  if (res[1]!=0) res[1][2]=a15;
  if (res[1]!=0) res[1][3]=a15;
  if (res[1]!=0) res[1][4]=a15;
  if (res[1]!=0) res[1][5]=a15;
  if (res[1]!=0) res[1][6]=a15;
  if (res[1]!=0) res[1][7]=a15;
  a15=2.7025639012821789e-01;
  a6=1.2330447799599942e+00;
  a178=1.4439765966454325e+00;
  a69=-2.7025639012821762e-01;
  a29=(a29*a16);
  a198=(a198*a29);
  a21=(a21*a198);
  a198=(a69*a21);
  a198=(a178*a198);
  a29=(a6*a198);
  a16=9.6278838983177628e-01;
  a21=(a16*a21);
  a29=(a29-a21);
  a29=(a15*a29);
  a29=(-a29);
  if (res[2]!=0) res[2][0]=a29;
  if (res[2]!=0) res[2][1]=a198;
  a63=(a63*a53);
  a174=(a174*a63);
  a57=(a57*a174);
  a174=(a69*a57);
  a174=(a178*a174);
  a63=(a6*a174);
  a57=(a16*a57);
  a63=(a63-a57);
  a63=(a15*a63);
  a63=(-a63);
  if (res[2]!=0) res[2][2]=a63;
  if (res[2]!=0) res[2][3]=a174;
  a110=(a110*a92);
  a191=(a191*a110);
  a97=(a97*a191);
  a110=(a69*a97);
  a92=9.6278838983177639e-01;
  a96=(a96*a191);
  a191=(a92*a96);
  a110=(a110+a191);
  a110=(a178*a110);
  a191=(a6*a110);
  a97=(a16*a97);
  a96=(a15*a96);
  a97=(a97+a96);
  a191=(a191-a97);
  a191=(a15*a191);
  a191=(-a191);
  if (res[2]!=0) res[2][4]=a191;
  if (res[2]!=0) res[2][5]=a110;
  a153=(a153*a138);
  a157=(a157*a153);
  a142=(a142*a157);
  a69=(a69*a142);
  a141=(a141*a157);
  a92=(a92*a141);
  a69=(a69+a92);
  a178=(a178*a69);
  a6=(a6*a178);
  a16=(a16*a142);
  a141=(a15*a141);
  a16=(a16+a141);
  a6=(a6-a16);
  a15=(a15*a6);
  a15=(-a15);
  if (res[2]!=0) res[2][6]=a15;
  if (res[2]!=0) res[2][7]=a178;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
