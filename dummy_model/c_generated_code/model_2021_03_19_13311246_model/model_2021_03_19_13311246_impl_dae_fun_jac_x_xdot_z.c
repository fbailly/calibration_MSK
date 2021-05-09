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
  #define CASADI_PREFIX(ID) model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a17=arg[0]? arg[0][0] : 0;
  a18=arg[2]? arg[2][0] : 0;
  a19=1.;
  a20=(a18<a19);
  a18=(a20?a18:0);
  a20=(!a20);
  a20=(a20?a19:0);
  a18=(a18+a20);
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a20=casadi_sq(a14);
  a9=(a9+a20);
  a9=sqrt(a9);
  a20=(a9-a19);
  a21=9.9995000041666526e-01;
  a20=(a20/a21);
  a22=6.9999999999999996e-01;
  a23=(a20/a22);
  a23=(a23-a19);
  a24=casadi_sq(a23);
  a25=4.5000000000000001e-01;
  a24=(a24/a25);
  a24=(-a24);
  a24=exp(a24);
  a26=(a18*a24);
  a27=(a13*a1);
  a28=0.;
  a29=(a27<=a28);
  a30=fabs(a27);
  a31=10.;
  a30=(a30/a31);
  a30=(a19-a30);
  a32=fabs(a27);
  a32=(a32/a31);
  a32=(a19+a32);
  a30=(a30/a32);
  a33=(a29?a30:0);
  a34=(!a29);
  a35=1.3300000000000001e+00;
  a36=(a35*a27);
  a36=(a36/a31);
  a37=-8.2500000000000004e-02;
  a36=(a36/a37);
  a36=(a19-a36);
  a38=(a27/a31);
  a38=(a38/a37);
  a38=(a19-a38);
  a36=(a36/a38);
  a39=(a34?a36:0);
  a33=(a33+a39);
  a39=(a26*a33);
  a40=(a19<a20);
  a20=(a20/a22);
  a20=(a20-a19);
  a20=(a31*a20);
  a20=exp(a20);
  a41=(a20-a19);
  a42=1.4741315910257660e+02;
  a41=(a41/a42);
  a41=(a40?a41:0);
  a39=(a39+a41);
  a41=1.0000000000000001e-01;
  a43=7.;
  a44=(a27/a43);
  a44=(a41*a44);
  a39=(a39+a44);
  a44=(a17*a39);
  a45=(a13*a44);
  a46=sin(a6);
  a47=(a5*a46);
  a48=(a47+a5);
  a49=cos(a6);
  a50=(a5*a49);
  a51=(a48*a50);
  a52=(a5*a49);
  a46=(a5*a46);
  a53=(a52*a46);
  a51=(a51-a53);
  a53=(a47+a5);
  a54=casadi_sq(a53);
  a55=casadi_sq(a52);
  a54=(a54+a55);
  a54=sqrt(a54);
  a51=(a51/a54);
  a55=arg[0]? arg[0][1] : 0;
  a56=arg[2]? arg[2][1] : 0;
  a57=(a56<a19);
  a56=(a57?a56:0);
  a57=(!a57);
  a57=(a57?a19:0);
  a56=(a56+a57);
  a47=(a47+a5);
  a57=casadi_sq(a47);
  a58=casadi_sq(a52);
  a57=(a57+a58);
  a57=sqrt(a57);
  a58=(a57-a19);
  a58=(a58/a21);
  a59=(a58/a22);
  a59=(a59-a19);
  a60=casadi_sq(a59);
  a60=(a60/a25);
  a60=(-a60);
  a60=exp(a60);
  a61=(a56*a60);
  a62=(a51*a1);
  a63=(a62<=a28);
  a64=fabs(a62);
  a64=(a64/a31);
  a64=(a19-a64);
  a65=fabs(a62);
  a65=(a65/a31);
  a65=(a19+a65);
  a64=(a64/a65);
  a66=(a63?a64:0);
  a67=(!a63);
  a68=(a35*a62);
  a68=(a68/a31);
  a68=(a68/a37);
  a68=(a19-a68);
  a69=(a62/a31);
  a69=(a69/a37);
  a69=(a19-a69);
  a68=(a68/a69);
  a70=(a67?a68:0);
  a66=(a66+a70);
  a70=(a61*a66);
  a71=(a19<a58);
  a58=(a58/a22);
  a58=(a58-a19);
  a58=(a31*a58);
  a58=exp(a58);
  a72=(a58-a19);
  a72=(a72/a42);
  a72=(a71?a72:0);
  a70=(a70+a72);
  a72=(a62/a43);
  a72=(a41*a72);
  a70=(a70+a72);
  a72=(a55*a70);
  a73=(a51*a72);
  a45=(a45+a73);
  a73=arg[0]? arg[0][5] : 0;
  a74=sin(a73);
  a75=sin(a6);
  a76=(a74*a75);
  a77=cos(a73);
  a78=cos(a6);
  a79=(a77*a78);
  a76=(a76-a79);
  a79=(a5*a76);
  a80=1.2500000000000000e+00;
  a81=(a80*a75);
  a79=(a79-a81);
  a82=7.5000000000000000e-01;
  a83=(a82*a75);
  a84=(a79+a83);
  a85=(a82*a78);
  a86=(a80*a78);
  a87=(a77*a75);
  a88=(a74*a78);
  a87=(a87+a88);
  a88=(a5*a87);
  a88=(a86-a88);
  a85=(a85-a88);
  a89=(a84*a85);
  a90=(a5*a87);
  a90=(a86-a90);
  a91=(a82*a78);
  a92=(a90-a91);
  a93=(a5*a76);
  a93=(a93-a81);
  a94=(a82*a75);
  a94=(a93+a94);
  a95=(a92*a94);
  a89=(a89+a95);
  a95=(a79+a83);
  a96=casadi_sq(a95);
  a97=(a90-a91);
  a98=casadi_sq(a97);
  a96=(a96+a98);
  a96=sqrt(a96);
  a89=(a89/a96);
  a98=arg[0]? arg[0][2] : 0;
  a99=arg[2]? arg[2][2] : 0;
  a100=(a99<a19);
  a99=(a100?a99:0);
  a100=(!a100);
  a100=(a100?a19:0);
  a99=(a99+a100);
  a79=(a79+a83);
  a83=casadi_sq(a79);
  a90=(a90-a91);
  a91=casadi_sq(a90);
  a83=(a83+a91);
  a83=sqrt(a83);
  a91=(a83-a19);
  a91=(a91/a21);
  a100=(a91/a22);
  a100=(a100-a19);
  a101=casadi_sq(a100);
  a101=(a101/a25);
  a101=(-a101);
  a101=exp(a101);
  a102=(a99*a101);
  a103=(a89*a1);
  a104=(a74*a78);
  a105=(a77*a75);
  a104=(a104+a105);
  a105=(a76*a81);
  a106=(a87*a86);
  a105=(a105+a106);
  a106=(a104*a105);
  a107=(a104*a81);
  a108=(a77*a78);
  a109=(a74*a75);
  a108=(a108-a109);
  a109=(a108*a86);
  a107=(a107+a109);
  a109=(a76*a107);
  a106=(a106-a109);
  a106=(a106-a88);
  a88=(a84*a106);
  a109=(a87*a107);
  a110=(a108*a105);
  a109=(a109-a110);
  a109=(a109+a93);
  a93=(a92*a109);
  a88=(a88+a93);
  a88=(a88/a96);
  a93=(a88*a2);
  a103=(a103+a93);
  a93=(a103<=a28);
  a110=fabs(a103);
  a110=(a110/a31);
  a110=(a19-a110);
  a111=fabs(a103);
  a111=(a111/a31);
  a111=(a19+a111);
  a110=(a110/a111);
  a112=(a93?a110:0);
  a113=(!a93);
  a114=(a35*a103);
  a114=(a114/a31);
  a114=(a114/a37);
  a114=(a19-a114);
  a115=(a103/a31);
  a115=(a115/a37);
  a115=(a19-a115);
  a114=(a114/a115);
  a116=(a113?a114:0);
  a112=(a112+a116);
  a116=(a102*a112);
  a117=(a19<a91);
  a91=(a91/a22);
  a91=(a91-a19);
  a91=(a31*a91);
  a91=exp(a91);
  a118=(a91-a19);
  a118=(a118/a42);
  a118=(a117?a118:0);
  a116=(a116+a118);
  a118=(a103/a43);
  a118=(a41*a118);
  a116=(a116+a118);
  a118=(a98*a116);
  a119=(a89*a118);
  a45=(a45+a119);
  a119=sin(a73);
  a120=sin(a6);
  a121=(a119*a120);
  a122=cos(a73);
  a123=cos(a6);
  a124=(a122*a123);
  a121=(a121-a124);
  a124=(a5*a121);
  a125=(a80*a120);
  a124=(a124-a125);
  a126=1.7500000000000000e+00;
  a127=(a126*a120);
  a128=(a124+a127);
  a129=(a126*a123);
  a130=(a80*a123);
  a131=(a122*a120);
  a132=(a119*a123);
  a131=(a131+a132);
  a132=(a5*a131);
  a132=(a130-a132);
  a129=(a129-a132);
  a133=(a128*a129);
  a134=(a5*a131);
  a134=(a130-a134);
  a135=(a126*a123);
  a136=(a134-a135);
  a137=(a5*a121);
  a137=(a137-a125);
  a138=(a126*a120);
  a138=(a137+a138);
  a139=(a136*a138);
  a133=(a133+a139);
  a139=(a124+a127);
  a140=casadi_sq(a139);
  a141=(a134-a135);
  a142=casadi_sq(a141);
  a140=(a140+a142);
  a140=sqrt(a140);
  a133=(a133/a140);
  a142=arg[0]? arg[0][3] : 0;
  a143=arg[2]? arg[2][3] : 0;
  a144=(a143<a19);
  a143=(a144?a143:0);
  a144=(!a144);
  a144=(a144?a19:0);
  a143=(a143+a144);
  a124=(a124+a127);
  a127=casadi_sq(a124);
  a134=(a134-a135);
  a135=casadi_sq(a134);
  a127=(a127+a135);
  a127=sqrt(a127);
  a135=(a127-a19);
  a135=(a135/a21);
  a21=(a135/a22);
  a21=(a21-a19);
  a144=casadi_sq(a21);
  a144=(a144/a25);
  a144=(-a144);
  a144=exp(a144);
  a25=(a143*a144);
  a145=(a133*a1);
  a146=(a119*a123);
  a147=(a122*a120);
  a146=(a146+a147);
  a147=(a121*a125);
  a148=(a131*a130);
  a147=(a147+a148);
  a148=(a146*a147);
  a149=(a146*a125);
  a150=(a122*a123);
  a151=(a119*a120);
  a150=(a150-a151);
  a151=(a150*a130);
  a149=(a149+a151);
  a151=(a121*a149);
  a148=(a148-a151);
  a148=(a148-a132);
  a132=(a128*a148);
  a151=(a131*a149);
  a152=(a150*a147);
  a151=(a151-a152);
  a151=(a151+a137);
  a137=(a136*a151);
  a132=(a132+a137);
  a132=(a132/a140);
  a137=(a132*a2);
  a145=(a145+a137);
  a28=(a145<=a28);
  a137=fabs(a145);
  a137=(a137/a31);
  a137=(a19-a137);
  a152=fabs(a145);
  a152=(a152/a31);
  a152=(a19+a152);
  a137=(a137/a152);
  a153=(a28?a137:0);
  a154=(!a28);
  a155=(a35*a145);
  a155=(a155/a31);
  a155=(a155/a37);
  a155=(a19-a155);
  a156=(a145/a31);
  a156=(a156/a37);
  a156=(a19-a156);
  a155=(a155/a156);
  a37=(a154?a155:0);
  a153=(a153+a37);
  a37=(a25*a153);
  a157=(a19<a135);
  a135=(a135/a22);
  a135=(a135-a19);
  a135=(a31*a135);
  a135=exp(a135);
  a22=(a135-a19);
  a22=(a22/a42);
  a22=(a157?a22:0);
  a37=(a37+a22);
  a43=(a145/a43);
  a43=(a41*a43);
  a37=(a37+a43);
  a43=(a142*a37);
  a22=(a133*a43);
  a45=(a45+a22);
  a22=sin(a73);
  a42=cos(a73);
  a158=9.8100000000000005e+00;
  a159=cos(a6);
  a159=(a158*a159);
  a160=(a42*a159);
  a161=sin(a6);
  a161=(a158*a161);
  a162=(a22*a161);
  a160=(a160-a162);
  a162=(a80*a1);
  a163=(a42*a162);
  a164=(a163*a2);
  a160=(a160+a164);
  a164=(a1+a2);
  a165=(a164*a163);
  a160=(a160-a165);
  a165=(a22*a160);
  a166=(a22*a162);
  a167=(a164*a166);
  a168=(a42*a161);
  a169=(a22*a159);
  a168=(a168+a169);
  a169=(a166*a2);
  a168=(a168+a169);
  a167=(a167-a168);
  a168=(a42*a167);
  a165=(a165+a168);
  a165=(a80*a165);
  a45=(a45+a165);
  a4=(a4*a45);
  a165=9.6278838983177639e-01;
  a168=(a88*a118);
  a169=(a132*a43);
  a168=(a168+a169);
  a165=(a165*a168);
  a4=(a4+a165);
  a165=6.9253199970355839e-01;
  a4=(a4/a165);
  a3=(a3*a4);
  a165=9.6278838983177628e-01;
  a165=(a165*a45);
  a45=2.7025639012821789e-01;
  a45=(a45*a168);
  a165=(a165+a45);
  a3=(a3-a165);
  a165=3.7001900289039211e+00;
  a3=(a3/a165);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  a0=3.9024390243902418e-01;
  a4=(a0*a13);
  a3=(a39*a4);
  if (res[1]!=0) res[1][0]=a3;
  a3=-3.9024390243902396e-01;
  a165=(a3*a13);
  a39=(a39*a165);
  if (res[1]!=0) res[1][1]=a39;
  a39=(a0*a51);
  a45=(a70*a39);
  if (res[1]!=0) res[1][2]=a45;
  a45=(a3*a51);
  a70=(a70*a45);
  if (res[1]!=0) res[1][3]=a70;
  a70=-3.9024390243902440e-01;
  a168=(a70*a88);
  a169=(a0*a89);
  a168=(a168+a169);
  a169=(a116*a168);
  if (res[1]!=0) res[1][4]=a169;
  a169=1.3902439024390245e+00;
  a170=(a169*a88);
  a171=(a3*a89);
  a170=(a170+a171);
  a116=(a116*a170);
  if (res[1]!=0) res[1][5]=a116;
  a116=(a70*a132);
  a171=(a0*a133);
  a116=(a116+a171);
  a171=(a37*a116);
  if (res[1]!=0) res[1][6]=a171;
  a171=(a169*a132);
  a172=(a3*a133);
  a171=(a171+a172);
  a37=(a37*a171);
  if (res[1]!=0) res[1][7]=a37;
  a37=cos(a6);
  a172=(a70*a43);
  a173=1.4285714285714285e-01;
  a116=(a142*a116);
  a174=(a41*a116);
  a174=(a173*a174);
  a175=-1.2121212121212121e+01;
  a176=(a25*a116);
  a155=(a155/a156);
  a177=(a176*a155);
  a177=(a175*a177);
  a177=(a41*a177);
  a177=(a154?a177:0);
  a174=(a174+a177);
  a177=(a176/a156);
  a177=(a175*a177);
  a177=(a41*a177);
  a177=(a35*a177);
  a177=(-a177);
  a177=(a154?a177:0);
  a174=(a174+a177);
  a137=(a137/a152);
  a177=(a176*a137);
  a177=(a41*a177);
  a178=casadi_sign(a145);
  a177=(a177*a178);
  a177=(-a177);
  a177=(a28?a177:0);
  a174=(a174+a177);
  a176=(a176/a152);
  a176=(a41*a176);
  a145=casadi_sign(a145);
  a176=(a176*a145);
  a176=(-a176);
  a176=(a28?a176:0);
  a174=(a174+a176);
  a176=(a2*a174);
  a172=(a172+a176);
  a176=(a172/a140);
  a177=(a136*a176);
  a179=(a131*a177);
  a180=(a128*a176);
  a181=(a121*a180);
  a179=(a179-a181);
  a181=(a125*a179);
  a182=(a147*a180);
  a181=(a181+a182);
  a182=(a122*a181);
  a183=(a130*a179);
  a184=(a147*a177);
  a183=(a183-a184);
  a184=(a119*a183);
  a182=(a182-a184);
  a184=(a0*a43);
  a185=(a1*a174);
  a184=(a184+a185);
  a185=(a184/a140);
  a186=(a136*a185);
  a187=(a126*a186);
  a182=(a182+a187);
  a187=(a149*a177);
  a188=(a146*a180);
  a189=(a150*a177);
  a188=(a188-a189);
  a189=(a130*a188);
  a187=(a187+a189);
  a134=(a134+a134);
  a189=1.0000500020834180e+00;
  a190=1.4285714285714286e+00;
  a191=6.7836549063042314e-03;
  a192=(a191*a116);
  a192=(a192*a135);
  a192=(a31*a192);
  a192=(a190*a192);
  a192=(a157?a192:0);
  a21=(a21+a21);
  a193=2.2222222222222223e+00;
  a116=(a153*a116);
  a116=(a143*a116);
  a116=(a144*a116);
  a116=(a193*a116);
  a116=(a21*a116);
  a116=(a190*a116);
  a192=(a192-a116);
  a192=(a189*a192);
  a127=(a127+a127);
  a192=(a192/a127);
  a116=(a134*a192);
  a141=(a141+a141);
  a194=(a132/a140);
  a172=(a194*a172);
  a195=(a133/a140);
  a184=(a195*a184);
  a172=(a172+a184);
  a184=(a140+a140);
  a172=(a172/a184);
  a196=(a141*a172);
  a197=(a116-a196);
  a198=(a151*a176);
  a199=(a138*a185);
  a198=(a198+a199);
  a197=(a197+a198);
  a199=(a5*a197);
  a187=(a187-a199);
  a199=(a128*a185);
  a200=(a180+a199);
  a201=(a5*a200);
  a187=(a187+a201);
  a201=(a122*a187);
  a182=(a182+a201);
  a124=(a124+a124);
  a192=(a124*a192);
  a139=(a139+a139);
  a172=(a139*a172);
  a201=(a192-a172);
  a176=(a148*a176);
  a185=(a129*a185);
  a176=(a176+a185);
  a201=(a201+a176);
  a201=(a126*a201);
  a182=(a182+a201);
  a201=(a146*a179);
  a185=(a121*a188);
  a201=(a201+a185);
  a177=(a177+a186);
  a201=(a201-a177);
  a192=(a192-a172);
  a192=(a192+a176);
  a201=(a201-a192);
  a201=(a80*a201);
  a182=(a182+a201);
  a201=(a125*a188);
  a180=(a149*a180);
  a201=(a201-a180);
  a177=(a5*a177);
  a201=(a201+a177);
  a192=(a5*a192);
  a201=(a201+a192);
  a192=(a119*a201);
  a182=(a182+a192);
  a182=(a37*a182);
  a192=cos(a6);
  a177=4.8780487804878025e-01;
  a180=(a177*a42);
  a176=(a42*a180);
  a172=(a177*a22);
  a186=(a22*a172);
  a176=(a176+a186);
  a176=(a158*a176);
  a176=(a192*a176);
  a186=sin(a6);
  a185=(a42*a172);
  a202=(a22*a180);
  a185=(a185-a202);
  a185=(a158*a185);
  a185=(a186*a185);
  a176=(a176+a185);
  a185=sin(a6);
  a202=(a122*a183);
  a203=(a119*a181);
  a202=(a202+a203);
  a196=(a196-a116);
  a196=(a196-a198);
  a196=(a126*a196);
  a202=(a202+a196);
  a196=(a119*a187);
  a202=(a202+a196);
  a179=(a150*a179);
  a188=(a131*a188);
  a179=(a179+a188);
  a179=(a179+a197);
  a179=(a179-a200);
  a179=(a80*a179);
  a202=(a202+a179);
  a199=(a126*a199);
  a202=(a202+a199);
  a199=(a122*a201);
  a202=(a202-a199);
  a202=(a185*a202);
  a176=(a176+a202);
  a182=(a182-a176);
  a176=sin(a6);
  a70=(a70*a118);
  a168=(a98*a168);
  a202=(a41*a168);
  a202=(a173*a202);
  a199=(a102*a168);
  a114=(a114/a115);
  a179=(a199*a114);
  a179=(a175*a179);
  a179=(a41*a179);
  a179=(a113?a179:0);
  a202=(a202+a179);
  a179=(a199/a115);
  a179=(a175*a179);
  a179=(a41*a179);
  a179=(a35*a179);
  a179=(-a179);
  a179=(a113?a179:0);
  a202=(a202+a179);
  a110=(a110/a111);
  a179=(a199*a110);
  a179=(a41*a179);
  a200=casadi_sign(a103);
  a179=(a179*a200);
  a179=(-a179);
  a179=(a93?a179:0);
  a202=(a202+a179);
  a199=(a199/a111);
  a199=(a41*a199);
  a103=casadi_sign(a103);
  a199=(a199*a103);
  a199=(-a199);
  a199=(a93?a199:0);
  a202=(a202+a199);
  a199=(a2*a202);
  a70=(a70+a199);
  a199=(a70/a96);
  a179=(a92*a199);
  a197=(a87*a179);
  a188=(a84*a199);
  a196=(a76*a188);
  a197=(a197-a196);
  a196=(a86*a197);
  a198=(a105*a179);
  a196=(a196-a198);
  a198=(a77*a196);
  a116=(a81*a197);
  a203=(a105*a188);
  a116=(a116+a203);
  a203=(a74*a116);
  a198=(a198+a203);
  a97=(a97+a97);
  a203=(a88/a96);
  a70=(a203*a70);
  a204=(a89/a96);
  a205=(a0*a118);
  a206=(a1*a202);
  a205=(a205+a206);
  a206=(a204*a205);
  a70=(a70+a206);
  a206=(a96+a96);
  a70=(a70/a206);
  a207=(a97*a70);
  a90=(a90+a90);
  a208=(a191*a168);
  a208=(a208*a91);
  a208=(a31*a208);
  a208=(a190*a208);
  a208=(a117?a208:0);
  a100=(a100+a100);
  a168=(a112*a168);
  a168=(a99*a168);
  a168=(a101*a168);
  a168=(a193*a168);
  a168=(a100*a168);
  a168=(a190*a168);
  a208=(a208-a168);
  a208=(a189*a208);
  a83=(a83+a83);
  a208=(a208/a83);
  a168=(a90*a208);
  a209=(a207-a168);
  a210=(a109*a199);
  a205=(a205/a96);
  a211=(a94*a205);
  a210=(a210+a211);
  a209=(a209-a210);
  a209=(a82*a209);
  a198=(a198+a209);
  a209=(a107*a179);
  a211=(a104*a188);
  a212=(a108*a179);
  a211=(a211-a212);
  a212=(a86*a211);
  a209=(a209+a212);
  a168=(a168-a207);
  a168=(a168+a210);
  a210=(a5*a168);
  a209=(a209-a210);
  a210=(a84*a205);
  a207=(a188+a210);
  a212=(a5*a207);
  a209=(a209+a212);
  a212=(a74*a209);
  a198=(a198+a212);
  a212=(a108*a197);
  a213=(a87*a211);
  a212=(a212+a213);
  a212=(a212+a168);
  a212=(a212-a207);
  a212=(a80*a212);
  a198=(a198+a212);
  a210=(a82*a210);
  a198=(a198+a210);
  a210=(a81*a211);
  a188=(a107*a188);
  a210=(a210-a188);
  a188=(a92*a205);
  a179=(a179+a188);
  a212=(a5*a179);
  a210=(a210+a212);
  a79=(a79+a79);
  a208=(a79*a208);
  a95=(a95+a95);
  a70=(a95*a70);
  a212=(a208-a70);
  a199=(a106*a199);
  a205=(a85*a205);
  a199=(a199+a205);
  a212=(a212+a199);
  a205=(a5*a212);
  a210=(a210+a205);
  a205=(a77*a210);
  a198=(a198-a205);
  a198=(a176*a198);
  a182=(a182-a198);
  a198=cos(a6);
  a205=(a77*a116);
  a207=(a74*a196);
  a205=(a205-a207);
  a188=(a82*a188);
  a205=(a205+a188);
  a188=(a77*a209);
  a205=(a205+a188);
  a208=(a208-a70);
  a208=(a208+a199);
  a208=(a82*a208);
  a205=(a205+a208);
  a197=(a104*a197);
  a211=(a76*a211);
  a197=(a197+a211);
  a197=(a197-a179);
  a197=(a197-a212);
  a197=(a80*a197);
  a205=(a205+a197);
  a197=(a74*a210);
  a205=(a205+a197);
  a205=(a198*a205);
  a182=(a182+a205);
  a205=sin(a6);
  a39=(a55*a39);
  a197=(a191*a39);
  a197=(a197*a58);
  a197=(a31*a197);
  a197=(a190*a197);
  a197=(a71?a197:0);
  a59=(a59+a59);
  a212=(a66*a39);
  a212=(a56*a212);
  a212=(a60*a212);
  a212=(a193*a212);
  a212=(a59*a212);
  a212=(a190*a212);
  a197=(a197-a212);
  a197=(a189*a197);
  a57=(a57+a57);
  a197=(a197/a57);
  a212=(a49*a197);
  a179=(a51/a54);
  a211=(a0*a72);
  a208=(a41*a39);
  a208=(a173*a208);
  a39=(a61*a39);
  a68=(a68/a69);
  a199=(a39*a68);
  a199=(a175*a199);
  a199=(a41*a199);
  a199=(a67?a199:0);
  a208=(a208+a199);
  a199=(a39/a69);
  a199=(a175*a199);
  a199=(a41*a199);
  a199=(a35*a199);
  a199=(-a199);
  a199=(a67?a199:0);
  a208=(a208+a199);
  a64=(a64/a65);
  a199=(a39*a64);
  a199=(a41*a199);
  a70=casadi_sign(a62);
  a199=(a199*a70);
  a199=(-a199);
  a199=(a63?a199:0);
  a208=(a208+a199);
  a39=(a39/a65);
  a39=(a41*a39);
  a62=casadi_sign(a62);
  a39=(a39*a62);
  a39=(-a39);
  a39=(a63?a39:0);
  a208=(a208+a39);
  a39=(a1*a208);
  a211=(a211+a39);
  a39=(a179*a211);
  a199=(a54+a54);
  a39=(a39/a199);
  a188=(a49*a39);
  a212=(a212-a188);
  a211=(a211/a54);
  a188=(a46*a211);
  a212=(a212-a188);
  a212=(a5*a212);
  a188=(a48*a211);
  a188=(a5*a188);
  a212=(a212+a188);
  a212=(a205*a212);
  a182=(a182-a212);
  a212=cos(a6);
  a47=(a47+a47);
  a197=(a47*a197);
  a53=(a53+a53);
  a39=(a53*a39);
  a197=(a197-a39);
  a39=(a50*a211);
  a197=(a197+a39);
  a197=(a5*a197);
  a211=(a52*a211);
  a211=(a5*a211);
  a197=(a197-a211);
  a197=(a212*a197);
  a182=(a182+a197);
  a197=sin(a6);
  a4=(a17*a4);
  a211=(a191*a4);
  a211=(a211*a20);
  a211=(a31*a211);
  a211=(a190*a211);
  a211=(a40?a211:0);
  a23=(a23+a23);
  a39=(a33*a4);
  a39=(a18*a39);
  a39=(a24*a39);
  a39=(a193*a39);
  a39=(a23*a39);
  a39=(a190*a39);
  a211=(a211-a39);
  a211=(a189*a211);
  a9=(a9+a9);
  a211=(a211/a9);
  a39=(a11*a211);
  a188=(a13/a16);
  a0=(a0*a44);
  a207=(a41*a4);
  a207=(a173*a207);
  a4=(a26*a4);
  a36=(a36/a38);
  a168=(a4*a36);
  a168=(a175*a168);
  a168=(a41*a168);
  a168=(a34?a168:0);
  a207=(a207+a168);
  a168=(a4/a38);
  a168=(a175*a168);
  a168=(a41*a168);
  a168=(a35*a168);
  a168=(-a168);
  a168=(a34?a168:0);
  a207=(a207+a168);
  a30=(a30/a32);
  a168=(a4*a30);
  a168=(a41*a168);
  a213=casadi_sign(a27);
  a168=(a168*a213);
  a168=(-a168);
  a168=(a29?a168:0);
  a207=(a207+a168);
  a4=(a4/a32);
  a4=(a41*a4);
  a27=casadi_sign(a27);
  a4=(a4*a27);
  a4=(-a4);
  a4=(a29?a4:0);
  a207=(a207+a4);
  a4=(a1*a207);
  a0=(a0+a4);
  a4=(a188*a0);
  a168=(a16+a16);
  a4=(a4/a168);
  a214=(a11*a4);
  a39=(a39-a214);
  a0=(a0/a16);
  a214=(a7*a0);
  a39=(a39-a214);
  a39=(a5*a39);
  a214=(a10*a0);
  a214=(a5*a214);
  a39=(a39+a214);
  a39=(a197*a39);
  a182=(a182-a39);
  a6=cos(a6);
  a8=(a8+a8);
  a211=(a8*a211);
  a15=(a15+a15);
  a4=(a15*a4);
  a211=(a211-a4);
  a4=(a12*a0);
  a211=(a211+a4);
  a211=(a5*a211);
  a0=(a14*a0);
  a0=(a5*a0);
  a211=(a211-a0);
  a211=(a6*a211);
  a182=(a182+a211);
  if (res[1]!=0) res[1][8]=a182;
  a182=(a169*a43);
  a142=(a142*a171);
  a171=(a41*a142);
  a171=(a173*a171);
  a25=(a25*a142);
  a155=(a25*a155);
  a155=(a175*a155);
  a155=(a41*a155);
  a155=(a154?a155:0);
  a171=(a171+a155);
  a156=(a25/a156);
  a156=(a175*a156);
  a156=(a41*a156);
  a156=(a35*a156);
  a156=(-a156);
  a154=(a154?a156:0);
  a171=(a171+a154);
  a137=(a25*a137);
  a137=(a41*a137);
  a137=(a137*a178);
  a137=(-a137);
  a137=(a28?a137:0);
  a171=(a171+a137);
  a25=(a25/a152);
  a25=(a41*a25);
  a25=(a25*a145);
  a25=(-a25);
  a28=(a28?a25:0);
  a171=(a171+a28);
  a28=(a2*a171);
  a182=(a182+a28);
  a28=(a182/a140);
  a25=(a136*a28);
  a145=(a131*a25);
  a152=(a128*a28);
  a137=(a121*a152);
  a145=(a145-a137);
  a137=(a125*a145);
  a178=(a147*a152);
  a137=(a137+a178);
  a178=(a122*a137);
  a154=(a130*a145);
  a147=(a147*a25);
  a154=(a154-a147);
  a147=(a119*a154);
  a178=(a178-a147);
  a43=(a3*a43);
  a147=(a1*a171);
  a43=(a43+a147);
  a140=(a43/a140);
  a136=(a136*a140);
  a147=(a126*a136);
  a178=(a178+a147);
  a147=(a149*a25);
  a156=(a146*a152);
  a155=(a150*a25);
  a156=(a156-a155);
  a130=(a130*a156);
  a147=(a147+a130);
  a130=(a191*a142);
  a130=(a130*a135);
  a130=(a31*a130);
  a130=(a190*a130);
  a157=(a157?a130:0);
  a153=(a153*a142);
  a143=(a143*a153);
  a144=(a144*a143);
  a144=(a193*a144);
  a21=(a21*a144);
  a21=(a190*a21);
  a157=(a157-a21);
  a157=(a189*a157);
  a157=(a157/a127);
  a134=(a134*a157);
  a194=(a194*a182);
  a195=(a195*a43);
  a194=(a194+a195);
  a194=(a194/a184);
  a141=(a141*a194);
  a184=(a134-a141);
  a151=(a151*a28);
  a138=(a138*a140);
  a151=(a151+a138);
  a184=(a184+a151);
  a138=(a5*a184);
  a147=(a147-a138);
  a128=(a128*a140);
  a138=(a152+a128);
  a195=(a5*a138);
  a147=(a147+a195);
  a195=(a122*a147);
  a178=(a178+a195);
  a124=(a124*a157);
  a139=(a139*a194);
  a194=(a124-a139);
  a148=(a148*a28);
  a129=(a129*a140);
  a148=(a148+a129);
  a194=(a194+a148);
  a194=(a126*a194);
  a178=(a178+a194);
  a146=(a146*a145);
  a121=(a121*a156);
  a146=(a146+a121);
  a25=(a25+a136);
  a146=(a146-a25);
  a124=(a124-a139);
  a124=(a124+a148);
  a146=(a146-a124);
  a146=(a80*a146);
  a178=(a178+a146);
  a125=(a125*a156);
  a149=(a149*a152);
  a125=(a125-a149);
  a25=(a5*a25);
  a125=(a125+a25);
  a124=(a5*a124);
  a125=(a125+a124);
  a124=(a119*a125);
  a178=(a178+a124);
  a37=(a37*a178);
  a178=-4.8780487804877992e-01;
  a124=(a178*a42);
  a25=(a42*a124);
  a149=(a178*a22);
  a152=(a22*a149);
  a25=(a25+a152);
  a25=(a158*a25);
  a192=(a192*a25);
  a25=(a42*a149);
  a152=(a22*a124);
  a25=(a25-a152);
  a158=(a158*a25);
  a186=(a186*a158);
  a192=(a192+a186);
  a186=(a122*a154);
  a158=(a119*a137);
  a186=(a186+a158);
  a141=(a141-a134);
  a141=(a141-a151);
  a141=(a126*a141);
  a186=(a186+a141);
  a119=(a119*a147);
  a186=(a186+a119);
  a150=(a150*a145);
  a131=(a131*a156);
  a150=(a150+a131);
  a150=(a150+a184);
  a150=(a150-a138);
  a150=(a80*a150);
  a186=(a186+a150);
  a126=(a126*a128);
  a186=(a186+a126);
  a122=(a122*a125);
  a186=(a186-a122);
  a185=(a185*a186);
  a192=(a192+a185);
  a37=(a37-a192);
  a169=(a169*a118);
  a98=(a98*a170);
  a170=(a41*a98);
  a170=(a173*a170);
  a102=(a102*a98);
  a114=(a102*a114);
  a114=(a175*a114);
  a114=(a41*a114);
  a114=(a113?a114:0);
  a170=(a170+a114);
  a115=(a102/a115);
  a115=(a175*a115);
  a115=(a41*a115);
  a115=(a35*a115);
  a115=(-a115);
  a113=(a113?a115:0);
  a170=(a170+a113);
  a110=(a102*a110);
  a110=(a41*a110);
  a110=(a110*a200);
  a110=(-a110);
  a110=(a93?a110:0);
  a170=(a170+a110);
  a102=(a102/a111);
  a102=(a41*a102);
  a102=(a102*a103);
  a102=(-a102);
  a93=(a93?a102:0);
  a170=(a170+a93);
  a93=(a2*a170);
  a169=(a169+a93);
  a93=(a169/a96);
  a102=(a92*a93);
  a103=(a87*a102);
  a111=(a84*a93);
  a110=(a76*a111);
  a103=(a103-a110);
  a110=(a86*a103);
  a200=(a105*a102);
  a110=(a110-a200);
  a200=(a77*a110);
  a113=(a81*a103);
  a105=(a105*a111);
  a113=(a113+a105);
  a105=(a74*a113);
  a200=(a200+a105);
  a203=(a203*a169);
  a118=(a3*a118);
  a169=(a1*a170);
  a118=(a118+a169);
  a204=(a204*a118);
  a203=(a203+a204);
  a203=(a203/a206);
  a97=(a97*a203);
  a206=(a191*a98);
  a206=(a206*a91);
  a206=(a31*a206);
  a206=(a190*a206);
  a117=(a117?a206:0);
  a112=(a112*a98);
  a99=(a99*a112);
  a101=(a101*a99);
  a101=(a193*a101);
  a100=(a100*a101);
  a100=(a190*a100);
  a117=(a117-a100);
  a117=(a189*a117);
  a117=(a117/a83);
  a90=(a90*a117);
  a83=(a97-a90);
  a109=(a109*a93);
  a118=(a118/a96);
  a94=(a94*a118);
  a109=(a109+a94);
  a83=(a83-a109);
  a83=(a82*a83);
  a200=(a200+a83);
  a83=(a107*a102);
  a94=(a104*a111);
  a96=(a108*a102);
  a94=(a94-a96);
  a86=(a86*a94);
  a83=(a83+a86);
  a90=(a90-a97);
  a90=(a90+a109);
  a109=(a5*a90);
  a83=(a83-a109);
  a84=(a84*a118);
  a109=(a111+a84);
  a97=(a5*a109);
  a83=(a83+a97);
  a97=(a74*a83);
  a200=(a200+a97);
  a108=(a108*a103);
  a87=(a87*a94);
  a108=(a108+a87);
  a108=(a108+a90);
  a108=(a108-a109);
  a108=(a80*a108);
  a200=(a200+a108);
  a84=(a82*a84);
  a200=(a200+a84);
  a81=(a81*a94);
  a107=(a107*a111);
  a81=(a81-a107);
  a92=(a92*a118);
  a102=(a102+a92);
  a107=(a5*a102);
  a81=(a81+a107);
  a79=(a79*a117);
  a95=(a95*a203);
  a203=(a79-a95);
  a106=(a106*a93);
  a85=(a85*a118);
  a106=(a106+a85);
  a203=(a203+a106);
  a85=(a5*a203);
  a81=(a81+a85);
  a85=(a77*a81);
  a200=(a200-a85);
  a176=(a176*a200);
  a37=(a37-a176);
  a176=(a77*a113);
  a200=(a74*a110);
  a176=(a176-a200);
  a92=(a82*a92);
  a176=(a176+a92);
  a77=(a77*a83);
  a176=(a176+a77);
  a79=(a79-a95);
  a79=(a79+a106);
  a82=(a82*a79);
  a176=(a176+a82);
  a104=(a104*a103);
  a76=(a76*a94);
  a104=(a104+a76);
  a104=(a104-a102);
  a104=(a104-a203);
  a104=(a80*a104);
  a176=(a176+a104);
  a74=(a74*a81);
  a176=(a176+a74);
  a198=(a198*a176);
  a37=(a37+a198);
  a55=(a55*a45);
  a45=(a191*a55);
  a45=(a45*a58);
  a45=(a31*a45);
  a45=(a190*a45);
  a71=(a71?a45:0);
  a66=(a66*a55);
  a56=(a56*a66);
  a60=(a60*a56);
  a60=(a193*a60);
  a59=(a59*a60);
  a59=(a190*a59);
  a71=(a71-a59);
  a71=(a189*a71);
  a71=(a71/a57);
  a57=(a49*a71);
  a72=(a3*a72);
  a59=(a41*a55);
  a59=(a173*a59);
  a61=(a61*a55);
  a68=(a61*a68);
  a68=(a175*a68);
  a68=(a41*a68);
  a68=(a67?a68:0);
  a59=(a59+a68);
  a69=(a61/a69);
  a69=(a175*a69);
  a69=(a41*a69);
  a69=(a35*a69);
  a69=(-a69);
  a67=(a67?a69:0);
  a59=(a59+a67);
  a64=(a61*a64);
  a64=(a41*a64);
  a64=(a64*a70);
  a64=(-a64);
  a64=(a63?a64:0);
  a59=(a59+a64);
  a61=(a61/a65);
  a61=(a41*a61);
  a61=(a61*a62);
  a61=(-a61);
  a63=(a63?a61:0);
  a59=(a59+a63);
  a63=(a1*a59);
  a72=(a72+a63);
  a179=(a179*a72);
  a179=(a179/a199);
  a49=(a49*a179);
  a57=(a57-a49);
  a72=(a72/a54);
  a46=(a46*a72);
  a57=(a57-a46);
  a57=(a5*a57);
  a48=(a48*a72);
  a48=(a5*a48);
  a57=(a57+a48);
  a205=(a205*a57);
  a37=(a37-a205);
  a47=(a47*a71);
  a53=(a53*a179);
  a47=(a47-a53);
  a50=(a50*a72);
  a47=(a47+a50);
  a47=(a5*a47);
  a52=(a52*a72);
  a52=(a5*a52);
  a47=(a47-a52);
  a212=(a212*a47);
  a37=(a37+a212);
  a17=(a17*a165);
  a191=(a191*a17);
  a191=(a191*a20);
  a31=(a31*a191);
  a31=(a190*a31);
  a40=(a40?a31:0);
  a33=(a33*a17);
  a18=(a18*a33);
  a24=(a24*a18);
  a193=(a193*a24);
  a23=(a23*a193);
  a190=(a190*a23);
  a40=(a40-a190);
  a189=(a189*a40);
  a189=(a189/a9);
  a9=(a11*a189);
  a3=(a3*a44);
  a44=(a41*a17);
  a173=(a173*a44);
  a26=(a26*a17);
  a36=(a26*a36);
  a36=(a175*a36);
  a36=(a41*a36);
  a36=(a34?a36:0);
  a173=(a173+a36);
  a38=(a26/a38);
  a175=(a175*a38);
  a175=(a41*a175);
  a35=(a35*a175);
  a35=(-a35);
  a34=(a34?a35:0);
  a173=(a173+a34);
  a30=(a26*a30);
  a30=(a41*a30);
  a30=(a30*a213);
  a30=(-a30);
  a30=(a29?a30:0);
  a173=(a173+a30);
  a26=(a26/a32);
  a41=(a41*a26);
  a41=(a41*a27);
  a41=(-a41);
  a29=(a29?a41:0);
  a173=(a173+a29);
  a1=(a1*a173);
  a3=(a3+a1);
  a188=(a188*a3);
  a188=(a188/a168);
  a11=(a11*a188);
  a9=(a9-a11);
  a3=(a3/a16);
  a7=(a7*a3);
  a9=(a9-a7);
  a9=(a5*a9);
  a10=(a10*a3);
  a10=(a5*a10);
  a9=(a9+a10);
  a197=(a197*a9);
  a37=(a37-a197);
  a8=(a8*a189);
  a15=(a15*a188);
  a8=(a8-a15);
  a12=(a12*a3);
  a8=(a8+a12);
  a8=(a5*a8);
  a14=(a14*a3);
  a5=(a5*a14);
  a8=(a8-a5);
  a6=(a6*a8);
  a37=(a37+a6);
  if (res[1]!=0) res[1][9]=a37;
  a37=cos(a73);
  a6=(a164*a180);
  a8=(a2*a180);
  a6=(a6-a8);
  a8=(a162*a6);
  a5=(a159*a180);
  a8=(a8-a5);
  a5=(a177*a160);
  a8=(a8+a5);
  a5=(a161*a172);
  a8=(a8-a5);
  a8=(a37*a8);
  a5=sin(a73);
  a177=(a177*a167);
  a14=(a161*a180);
  a177=(a177-a14);
  a14=(a2*a172);
  a3=(a164*a172);
  a14=(a14-a3);
  a3=(a162*a14);
  a177=(a177+a3);
  a3=(a159*a172);
  a177=(a177+a3);
  a177=(a5*a177);
  a8=(a8-a177);
  a177=sin(a73);
  a3=(a123*a183);
  a12=(a120*a181);
  a3=(a3+a12);
  a12=(a120*a187);
  a3=(a3+a12);
  a12=(a123*a201);
  a3=(a3-a12);
  a3=(a177*a3);
  a8=(a8-a3);
  a3=cos(a73);
  a181=(a123*a181);
  a183=(a120*a183);
  a181=(a181-a183);
  a187=(a123*a187);
  a181=(a181+a187);
  a201=(a120*a201);
  a181=(a181+a201);
  a181=(a3*a181);
  a8=(a8+a181);
  a181=sin(a73);
  a201=(a78*a196);
  a187=(a75*a116);
  a201=(a201+a187);
  a187=(a75*a209);
  a201=(a201+a187);
  a187=(a78*a210);
  a201=(a201-a187);
  a201=(a181*a201);
  a8=(a8-a201);
  a73=cos(a73);
  a116=(a78*a116);
  a196=(a75*a196);
  a116=(a116-a196);
  a209=(a78*a209);
  a116=(a116+a209);
  a210=(a75*a210);
  a116=(a116+a210);
  a116=(a73*a116);
  a8=(a8+a116);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a164*a124);
  a116=(a2*a124);
  a8=(a8-a116);
  a116=(a162*a8);
  a210=(a159*a124);
  a116=(a116-a210);
  a160=(a178*a160);
  a116=(a116+a160);
  a160=(a161*a149);
  a116=(a116-a160);
  a37=(a37*a116);
  a178=(a178*a167);
  a161=(a161*a124);
  a178=(a178-a161);
  a2=(a2*a149);
  a164=(a164*a149);
  a2=(a2-a164);
  a162=(a162*a2);
  a178=(a178+a162);
  a159=(a159*a149);
  a178=(a178+a159);
  a5=(a5*a178);
  a37=(a37-a5);
  a5=(a123*a154);
  a178=(a120*a137);
  a5=(a5+a178);
  a178=(a120*a147);
  a5=(a5+a178);
  a178=(a123*a125);
  a5=(a5-a178);
  a177=(a177*a5);
  a37=(a37-a177);
  a137=(a123*a137);
  a154=(a120*a154);
  a137=(a137-a154);
  a123=(a123*a147);
  a137=(a137+a123);
  a120=(a120*a125);
  a137=(a137+a120);
  a3=(a3*a137);
  a37=(a37+a3);
  a3=(a78*a110);
  a137=(a75*a113);
  a3=(a3+a137);
  a137=(a75*a83);
  a3=(a3+a137);
  a137=(a78*a81);
  a3=(a3-a137);
  a181=(a181*a3);
  a37=(a37-a181);
  a113=(a78*a113);
  a110=(a75*a110);
  a113=(a113-a110);
  a78=(a78*a83);
  a113=(a113+a78);
  a75=(a75*a81);
  a113=(a113+a75);
  a73=(a73*a113);
  a37=(a37+a73);
  if (res[1]!=0) res[1][11]=a37;
  a37=-1.;
  if (res[1]!=0) res[1][12]=a37;
  a73=(a166*a180);
  a113=(a163*a172);
  a73=(a73-a113);
  a6=(a22*a6);
  a14=(a42*a14);
  a6=(a6+a14);
  a6=(a80*a6);
  a6=(a73+a6);
  a14=(a133*a174);
  a6=(a6+a14);
  a14=(a89*a202);
  a6=(a6+a14);
  a208=(a51*a208);
  a6=(a6+a208);
  a207=(a13*a207);
  a6=(a6+a207);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a166*a124);
  a207=(a163*a149);
  a6=(a6-a207);
  a22=(a22*a8);
  a42=(a42*a2);
  a22=(a22+a42);
  a80=(a80*a22);
  a80=(a6+a80);
  a133=(a133*a171);
  a80=(a80+a133);
  a89=(a89*a170);
  a80=(a80+a89);
  a51=(a51*a59);
  a80=(a80+a51);
  a13=(a13*a173);
  a80=(a80+a13);
  if (res[1]!=0) res[1][14]=a80;
  if (res[1]!=0) res[1][15]=a37;
  a180=(a166*a180);
  a73=(a73-a180);
  a172=(a163*a172);
  a73=(a73+a172);
  a174=(a132*a174);
  a73=(a73+a174);
  a202=(a88*a202);
  a73=(a73+a202);
  if (res[1]!=0) res[1][16]=a73;
  a166=(a166*a124);
  a6=(a6-a166);
  a163=(a163*a149);
  a6=(a6+a163);
  a132=(a132*a171);
  a6=(a6+a132);
  a88=(a88*a170);
  a6=(a6+a88);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a19;
  if (res[2]!=0) res[2][1]=a19;
  if (res[2]!=0) res[2][2]=a19;
  if (res[2]!=0) res[2][3]=a19;
  if (res[2]!=0) res[2][4]=a19;
  if (res[2]!=0) res[2][5]=a19;
  if (res[2]!=0) res[2][6]=a19;
  if (res[2]!=0) res[2][7]=a19;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_19_13311246_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
