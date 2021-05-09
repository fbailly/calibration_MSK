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
  #define CASADI_PREFIX(ID) model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a215, a216, a217, a218, a219, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a5=4.3693674051011216e-01;
  a6=arg[0]? arg[0][4] : 0;
  a7=sin(a6);
  a8=(a5*a7);
  a9=-5.0000000000000000e-01;
  a10=(a8+a9);
  a11=cos(a6);
  a12=(a5*a11);
  a13=(a10*a12);
  a11=(a5*a11);
  a7=(a5*a7);
  a14=(a11*a7);
  a13=(a13-a14);
  a14=(a8+a9);
  a15=casadi_sq(a14);
  a16=casadi_sq(a11);
  a15=(a15+a16);
  a15=sqrt(a15);
  a13=(a13/a15);
  a16=arg[0]? arg[0][0] : 0;
  a17=arg[2]? arg[2][0] : 0;
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a18=casadi_sq(a11);
  a9=(a9+a18);
  a9=sqrt(a9);
  a18=4.0000000000000001e-02;
  a19=(a9-a18);
  a20=8.7758256189037276e-01;
  a19=(a19/a20);
  a21=6.9999999999999996e-01;
  a22=(a19/a21);
  a23=1.;
  a22=(a22-a23);
  a24=casadi_sq(a22);
  a25=4.5000000000000001e-01;
  a24=(a24/a25);
  a24=(-a24);
  a24=exp(a24);
  a26=(a17*a24);
  a27=(a13*a1);
  a28=0.;
  a29=(a27<=a28);
  a30=fabs(a27);
  a31=10.;
  a30=(a30/a31);
  a30=(a23-a30);
  a32=fabs(a27);
  a32=(a32/a31);
  a32=(a23+a32);
  a30=(a30/a32);
  a33=(a29?a30:0);
  a34=(!a29);
  a35=1.3300000000000001e+00;
  a36=(a35*a27);
  a36=(a36/a31);
  a37=-8.2500000000000004e-02;
  a36=(a36/a37);
  a36=(a23-a36);
  a38=(a27/a31);
  a38=(a38/a37);
  a38=(a23-a38);
  a36=(a36/a38);
  a39=(a34?a36:0);
  a33=(a33+a39);
  a39=(a26*a33);
  a40=(a18<a19);
  a19=(a19/a21);
  a19=(a19-a23);
  a19=(a31*a19);
  a19=exp(a19);
  a41=(a19-a23);
  a42=1.4741315910257660e+02;
  a41=(a41/a42);
  a41=(a40?a41:0);
  a39=(a39+a41);
  a41=1.0000000000000001e-01;
  a43=7.;
  a44=(a27/a43);
  a44=(a41*a44);
  a39=(a39+a44);
  a44=(a16*a39);
  a45=(a13*a44);
  a46=5.9458846032218293e-01;
  a47=sin(a6);
  a48=(a46*a47);
  a49=5.0000000000000000e-01;
  a50=(a48+a49);
  a51=cos(a6);
  a52=(a46*a51);
  a53=(a50*a52);
  a51=(a46*a51);
  a47=(a46*a47);
  a54=(a51*a47);
  a53=(a53-a54);
  a54=(a48+a49);
  a55=casadi_sq(a54);
  a56=casadi_sq(a51);
  a55=(a55+a56);
  a55=sqrt(a55);
  a53=(a53/a55);
  a56=arg[0]? arg[0][1] : 0;
  a57=arg[2]? arg[2][1] : 0;
  a48=(a48+a49);
  a49=casadi_sq(a48);
  a58=casadi_sq(a51);
  a49=(a49+a58);
  a49=sqrt(a49);
  a58=(a49-a18);
  a58=(a58/a20);
  a59=(a58/a21);
  a59=(a59-a23);
  a60=casadi_sq(a59);
  a60=(a60/a25);
  a60=(-a60);
  a60=exp(a60);
  a61=(a57*a60);
  a62=(a53*a1);
  a63=(a62<=a28);
  a64=fabs(a62);
  a64=(a64/a31);
  a64=(a23-a64);
  a65=fabs(a62);
  a65=(a65/a31);
  a65=(a23+a65);
  a64=(a64/a65);
  a66=(a63?a64:0);
  a67=(!a63);
  a68=(a35*a62);
  a68=(a68/a31);
  a68=(a68/a37);
  a68=(a23-a68);
  a69=(a62/a31);
  a69=(a69/a37);
  a69=(a23-a69);
  a68=(a68/a69);
  a70=(a67?a68:0);
  a66=(a66+a70);
  a70=(a61*a66);
  a71=(a18<a58);
  a58=(a58/a21);
  a58=(a58-a23);
  a58=(a31*a58);
  a58=exp(a58);
  a72=(a58-a23);
  a72=(a72/a42);
  a72=(a71?a72:0);
  a70=(a70+a72);
  a72=(a62/a43);
  a72=(a41*a72);
  a70=(a70+a72);
  a72=(a56*a70);
  a73=(a53*a72);
  a45=(a45+a73);
  a73=3.5028313960910029e-01;
  a74=arg[0]? arg[0][5] : 0;
  a75=sin(a74);
  a76=sin(a6);
  a77=(a75*a76);
  a78=cos(a74);
  a79=cos(a6);
  a80=(a78*a79);
  a77=(a77-a80);
  a80=(a73*a77);
  a81=1.2500000000000000e+00;
  a82=(a81*a76);
  a80=(a80-a82);
  a83=7.5000000000000000e-01;
  a84=(a83*a76);
  a85=(a80+a84);
  a86=(a83*a79);
  a87=(a81*a79);
  a88=(a78*a76);
  a89=(a75*a79);
  a88=(a88+a89);
  a89=(a73*a88);
  a89=(a87-a89);
  a86=(a86-a89);
  a90=(a85*a86);
  a91=(a73*a88);
  a91=(a87-a91);
  a92=(a83*a79);
  a93=(a91-a92);
  a94=(a73*a77);
  a94=(a94-a82);
  a95=(a83*a76);
  a95=(a94+a95);
  a96=(a93*a95);
  a90=(a90+a96);
  a96=(a80+a84);
  a97=casadi_sq(a96);
  a98=(a91-a92);
  a99=casadi_sq(a98);
  a97=(a97+a99);
  a97=sqrt(a97);
  a90=(a90/a97);
  a99=arg[0]? arg[0][2] : 0;
  a100=arg[2]? arg[2][2] : 0;
  a80=(a80+a84);
  a84=casadi_sq(a80);
  a91=(a91-a92);
  a92=casadi_sq(a91);
  a84=(a84+a92);
  a84=sqrt(a84);
  a92=(a84-a18);
  a92=(a92/a20);
  a101=(a92/a21);
  a101=(a101-a23);
  a102=casadi_sq(a101);
  a102=(a102/a25);
  a102=(-a102);
  a102=exp(a102);
  a103=(a100*a102);
  a104=(a90*a1);
  a105=(a75*a79);
  a106=(a78*a76);
  a105=(a105+a106);
  a106=(a77*a82);
  a107=(a88*a87);
  a106=(a106+a107);
  a107=(a105*a106);
  a108=(a105*a82);
  a109=(a78*a79);
  a110=(a75*a76);
  a109=(a109-a110);
  a110=(a109*a87);
  a108=(a108+a110);
  a110=(a77*a108);
  a107=(a107-a110);
  a107=(a107-a89);
  a89=(a85*a107);
  a110=(a88*a108);
  a111=(a109*a106);
  a110=(a110-a111);
  a110=(a110+a94);
  a94=(a93*a110);
  a89=(a89+a94);
  a89=(a89/a97);
  a94=(a89*a2);
  a104=(a104+a94);
  a94=(a104<=a28);
  a111=fabs(a104);
  a111=(a111/a31);
  a111=(a23-a111);
  a112=fabs(a104);
  a112=(a112/a31);
  a112=(a23+a112);
  a111=(a111/a112);
  a113=(a94?a111:0);
  a114=(!a94);
  a115=(a35*a104);
  a115=(a115/a31);
  a115=(a115/a37);
  a115=(a23-a115);
  a116=(a104/a31);
  a116=(a116/a37);
  a116=(a23-a116);
  a115=(a115/a116);
  a117=(a114?a115:0);
  a113=(a113+a117);
  a117=(a103*a113);
  a118=(a18<a92);
  a92=(a92/a21);
  a92=(a92-a23);
  a92=(a31*a92);
  a92=exp(a92);
  a119=(a92-a23);
  a119=(a119/a42);
  a119=(a118?a119:0);
  a117=(a117+a119);
  a119=(a104/a43);
  a119=(a41*a119);
  a117=(a117+a119);
  a119=(a99*a117);
  a120=(a90*a119);
  a45=(a45+a120);
  a120=5.5628333693748588e-01;
  a121=sin(a74);
  a122=sin(a6);
  a123=(a121*a122);
  a124=cos(a74);
  a125=cos(a6);
  a126=(a124*a125);
  a123=(a123-a126);
  a126=(a120*a123);
  a127=(a81*a122);
  a126=(a126-a127);
  a128=1.7500000000000000e+00;
  a129=(a128*a122);
  a130=(a126+a129);
  a131=(a128*a125);
  a132=(a81*a125);
  a133=(a124*a122);
  a134=(a121*a125);
  a133=(a133+a134);
  a134=(a120*a133);
  a134=(a132-a134);
  a131=(a131-a134);
  a135=(a130*a131);
  a136=(a120*a133);
  a136=(a132-a136);
  a137=(a128*a125);
  a138=(a136-a137);
  a139=(a120*a123);
  a139=(a139-a127);
  a140=(a128*a122);
  a140=(a139+a140);
  a141=(a138*a140);
  a135=(a135+a141);
  a141=(a126+a129);
  a142=casadi_sq(a141);
  a143=(a136-a137);
  a144=casadi_sq(a143);
  a142=(a142+a144);
  a142=sqrt(a142);
  a135=(a135/a142);
  a144=arg[0]? arg[0][3] : 0;
  a145=arg[2]? arg[2][3] : 0;
  a126=(a126+a129);
  a129=casadi_sq(a126);
  a136=(a136-a137);
  a137=casadi_sq(a136);
  a129=(a129+a137);
  a129=sqrt(a129);
  a137=(a129-a18);
  a137=(a137/a20);
  a20=(a137/a21);
  a20=(a20-a23);
  a146=casadi_sq(a20);
  a146=(a146/a25);
  a146=(-a146);
  a146=exp(a146);
  a25=(a145*a146);
  a147=(a135*a1);
  a148=(a121*a125);
  a149=(a124*a122);
  a148=(a148+a149);
  a149=(a123*a127);
  a150=(a133*a132);
  a149=(a149+a150);
  a150=(a148*a149);
  a151=(a148*a127);
  a152=(a124*a125);
  a153=(a121*a122);
  a152=(a152-a153);
  a153=(a152*a132);
  a151=(a151+a153);
  a153=(a123*a151);
  a150=(a150-a153);
  a150=(a150-a134);
  a134=(a130*a150);
  a153=(a133*a151);
  a154=(a152*a149);
  a153=(a153-a154);
  a153=(a153+a139);
  a139=(a138*a153);
  a134=(a134+a139);
  a134=(a134/a142);
  a139=(a134*a2);
  a147=(a147+a139);
  a28=(a147<=a28);
  a139=fabs(a147);
  a139=(a139/a31);
  a139=(a23-a139);
  a154=fabs(a147);
  a154=(a154/a31);
  a154=(a23+a154);
  a139=(a139/a154);
  a155=(a28?a139:0);
  a156=(!a28);
  a157=(a35*a147);
  a157=(a157/a31);
  a157=(a157/a37);
  a157=(a23-a157);
  a158=(a147/a31);
  a158=(a158/a37);
  a158=(a23-a158);
  a157=(a157/a158);
  a37=(a156?a157:0);
  a155=(a155+a37);
  a37=(a25*a155);
  a18=(a18<a137);
  a137=(a137/a21);
  a137=(a137-a23);
  a137=(a31*a137);
  a137=exp(a137);
  a21=(a137-a23);
  a21=(a21/a42);
  a21=(a18?a21:0);
  a37=(a37+a21);
  a43=(a147/a43);
  a43=(a41*a43);
  a37=(a37+a43);
  a43=(a144*a37);
  a21=(a135*a43);
  a45=(a45+a21);
  a21=sin(a74);
  a42=cos(a74);
  a159=9.8100000000000005e+00;
  a160=cos(a6);
  a160=(a159*a160);
  a161=(a42*a160);
  a162=sin(a6);
  a162=(a159*a162);
  a163=(a21*a162);
  a161=(a161-a163);
  a163=(a81*a1);
  a164=(a42*a163);
  a165=(a164*a2);
  a161=(a161+a165);
  a165=(a1+a2);
  a166=(a165*a164);
  a161=(a161-a166);
  a166=(a21*a161);
  a167=(a21*a163);
  a168=(a165*a167);
  a169=(a42*a162);
  a170=(a21*a160);
  a169=(a169+a170);
  a170=(a167*a2);
  a169=(a169+a170);
  a168=(a168-a169);
  a169=(a42*a168);
  a166=(a166+a169);
  a166=(a81*a166);
  a45=(a45+a166);
  a4=(a4*a45);
  a166=9.6278838983177639e-01;
  a169=(a89*a119);
  a170=(a134*a43);
  a169=(a169+a170);
  a166=(a166*a169);
  a4=(a4+a166);
  a166=6.9253199970355839e-01;
  a4=(a4/a166);
  a3=(a3*a4);
  a166=9.6278838983177628e-01;
  a166=(a166*a45);
  a45=2.7025639012821789e-01;
  a45=(a45*a169);
  a166=(a166+a45);
  a3=(a3-a166);
  a166=3.7001900289039211e+00;
  a3=(a3/a166);
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
  a166=(a3*a13);
  a39=(a39*a166);
  if (res[1]!=0) res[1][1]=a39;
  a39=(a0*a53);
  a45=(a70*a39);
  if (res[1]!=0) res[1][2]=a45;
  a45=(a3*a53);
  a70=(a70*a45);
  if (res[1]!=0) res[1][3]=a70;
  a70=-3.9024390243902440e-01;
  a169=(a70*a89);
  a170=(a0*a90);
  a169=(a169+a170);
  a170=(a117*a169);
  if (res[1]!=0) res[1][4]=a170;
  a170=1.3902439024390245e+00;
  a171=(a170*a89);
  a172=(a3*a90);
  a171=(a171+a172);
  a117=(a117*a171);
  if (res[1]!=0) res[1][5]=a117;
  a117=(a70*a134);
  a172=(a0*a135);
  a117=(a117+a172);
  a172=(a37*a117);
  if (res[1]!=0) res[1][6]=a172;
  a172=(a170*a134);
  a173=(a3*a135);
  a172=(a172+a173);
  a37=(a37*a172);
  if (res[1]!=0) res[1][7]=a37;
  a37=cos(a6);
  a173=(a70*a43);
  a174=1.4285714285714285e-01;
  a117=(a144*a117);
  a175=(a41*a117);
  a175=(a174*a175);
  a176=-1.2121212121212121e+01;
  a177=(a25*a117);
  a157=(a157/a158);
  a178=(a177*a157);
  a178=(a176*a178);
  a178=(a41*a178);
  a178=(a156?a178:0);
  a175=(a175+a178);
  a178=(a177/a158);
  a178=(a176*a178);
  a178=(a41*a178);
  a178=(a35*a178);
  a178=(-a178);
  a178=(a156?a178:0);
  a175=(a175+a178);
  a139=(a139/a154);
  a178=(a177*a139);
  a178=(a41*a178);
  a179=casadi_sign(a147);
  a178=(a178*a179);
  a178=(-a178);
  a178=(a28?a178:0);
  a175=(a175+a178);
  a177=(a177/a154);
  a177=(a41*a177);
  a147=casadi_sign(a147);
  a177=(a177*a147);
  a177=(-a177);
  a177=(a28?a177:0);
  a175=(a175+a177);
  a177=(a2*a175);
  a173=(a173+a177);
  a177=(a173/a142);
  a178=(a138*a177);
  a180=(a133*a178);
  a181=(a130*a177);
  a182=(a123*a181);
  a180=(a180-a182);
  a182=(a127*a180);
  a183=(a149*a181);
  a182=(a182+a183);
  a183=(a124*a182);
  a184=(a132*a180);
  a185=(a149*a178);
  a184=(a184-a185);
  a185=(a121*a184);
  a183=(a183-a185);
  a185=(a0*a43);
  a186=(a1*a175);
  a185=(a185+a186);
  a186=(a185/a142);
  a187=(a138*a186);
  a188=(a128*a187);
  a183=(a183+a188);
  a188=(a151*a178);
  a189=(a148*a181);
  a190=(a152*a178);
  a189=(a189-a190);
  a190=(a132*a189);
  a188=(a188+a190);
  a136=(a136+a136);
  a190=1.1394939273245490e+00;
  a191=1.4285714285714286e+00;
  a192=6.7836549063042314e-03;
  a193=(a192*a117);
  a193=(a193*a137);
  a193=(a31*a193);
  a193=(a191*a193);
  a193=(a18?a193:0);
  a20=(a20+a20);
  a194=2.2222222222222223e+00;
  a117=(a155*a117);
  a117=(a145*a117);
  a117=(a146*a117);
  a117=(a194*a117);
  a117=(a20*a117);
  a117=(a191*a117);
  a193=(a193-a117);
  a193=(a190*a193);
  a129=(a129+a129);
  a193=(a193/a129);
  a117=(a136*a193);
  a143=(a143+a143);
  a195=(a134/a142);
  a173=(a195*a173);
  a196=(a135/a142);
  a185=(a196*a185);
  a173=(a173+a185);
  a185=(a142+a142);
  a173=(a173/a185);
  a197=(a143*a173);
  a198=(a117-a197);
  a199=(a153*a177);
  a200=(a140*a186);
  a199=(a199+a200);
  a198=(a198+a199);
  a200=(a120*a198);
  a188=(a188-a200);
  a200=(a130*a186);
  a201=(a181+a200);
  a202=(a120*a201);
  a188=(a188+a202);
  a202=(a124*a188);
  a183=(a183+a202);
  a126=(a126+a126);
  a193=(a126*a193);
  a141=(a141+a141);
  a173=(a141*a173);
  a202=(a193-a173);
  a177=(a150*a177);
  a186=(a131*a186);
  a177=(a177+a186);
  a202=(a202+a177);
  a202=(a128*a202);
  a183=(a183+a202);
  a202=(a148*a180);
  a186=(a123*a189);
  a202=(a202+a186);
  a178=(a178+a187);
  a202=(a202-a178);
  a193=(a193-a173);
  a193=(a193+a177);
  a202=(a202-a193);
  a202=(a81*a202);
  a183=(a183+a202);
  a202=(a127*a189);
  a181=(a151*a181);
  a202=(a202-a181);
  a178=(a120*a178);
  a202=(a202+a178);
  a193=(a120*a193);
  a202=(a202+a193);
  a193=(a121*a202);
  a183=(a183+a193);
  a183=(a37*a183);
  a193=cos(a6);
  a178=4.8780487804878025e-01;
  a181=(a178*a42);
  a177=(a42*a181);
  a173=(a178*a21);
  a187=(a21*a173);
  a177=(a177+a187);
  a177=(a159*a177);
  a177=(a193*a177);
  a187=sin(a6);
  a186=(a42*a173);
  a203=(a21*a181);
  a186=(a186-a203);
  a186=(a159*a186);
  a186=(a187*a186);
  a177=(a177+a186);
  a186=sin(a6);
  a203=(a124*a184);
  a204=(a121*a182);
  a203=(a203+a204);
  a197=(a197-a117);
  a197=(a197-a199);
  a197=(a128*a197);
  a203=(a203+a197);
  a197=(a121*a188);
  a203=(a203+a197);
  a180=(a152*a180);
  a189=(a133*a189);
  a180=(a180+a189);
  a180=(a180+a198);
  a180=(a180-a201);
  a180=(a81*a180);
  a203=(a203+a180);
  a200=(a128*a200);
  a203=(a203+a200);
  a200=(a124*a202);
  a203=(a203-a200);
  a203=(a186*a203);
  a177=(a177+a203);
  a183=(a183-a177);
  a177=sin(a6);
  a70=(a70*a119);
  a169=(a99*a169);
  a203=(a41*a169);
  a203=(a174*a203);
  a200=(a103*a169);
  a115=(a115/a116);
  a180=(a200*a115);
  a180=(a176*a180);
  a180=(a41*a180);
  a180=(a114?a180:0);
  a203=(a203+a180);
  a180=(a200/a116);
  a180=(a176*a180);
  a180=(a41*a180);
  a180=(a35*a180);
  a180=(-a180);
  a180=(a114?a180:0);
  a203=(a203+a180);
  a111=(a111/a112);
  a180=(a200*a111);
  a180=(a41*a180);
  a201=casadi_sign(a104);
  a180=(a180*a201);
  a180=(-a180);
  a180=(a94?a180:0);
  a203=(a203+a180);
  a200=(a200/a112);
  a200=(a41*a200);
  a104=casadi_sign(a104);
  a200=(a200*a104);
  a200=(-a200);
  a200=(a94?a200:0);
  a203=(a203+a200);
  a200=(a2*a203);
  a70=(a70+a200);
  a200=(a70/a97);
  a180=(a93*a200);
  a198=(a88*a180);
  a189=(a85*a200);
  a197=(a77*a189);
  a198=(a198-a197);
  a197=(a87*a198);
  a199=(a106*a180);
  a197=(a197-a199);
  a199=(a78*a197);
  a117=(a82*a198);
  a204=(a106*a189);
  a117=(a117+a204);
  a204=(a75*a117);
  a199=(a199+a204);
  a98=(a98+a98);
  a204=(a89/a97);
  a70=(a204*a70);
  a205=(a90/a97);
  a206=(a0*a119);
  a207=(a1*a203);
  a206=(a206+a207);
  a207=(a205*a206);
  a70=(a70+a207);
  a207=(a97+a97);
  a70=(a70/a207);
  a208=(a98*a70);
  a91=(a91+a91);
  a209=(a192*a169);
  a209=(a209*a92);
  a209=(a31*a209);
  a209=(a191*a209);
  a209=(a118?a209:0);
  a101=(a101+a101);
  a169=(a113*a169);
  a169=(a100*a169);
  a169=(a102*a169);
  a169=(a194*a169);
  a169=(a101*a169);
  a169=(a191*a169);
  a209=(a209-a169);
  a209=(a190*a209);
  a84=(a84+a84);
  a209=(a209/a84);
  a169=(a91*a209);
  a210=(a208-a169);
  a211=(a110*a200);
  a206=(a206/a97);
  a212=(a95*a206);
  a211=(a211+a212);
  a210=(a210-a211);
  a210=(a83*a210);
  a199=(a199+a210);
  a210=(a108*a180);
  a212=(a105*a189);
  a213=(a109*a180);
  a212=(a212-a213);
  a213=(a87*a212);
  a210=(a210+a213);
  a169=(a169-a208);
  a169=(a169+a211);
  a211=(a73*a169);
  a210=(a210-a211);
  a211=(a85*a206);
  a208=(a189+a211);
  a213=(a73*a208);
  a210=(a210+a213);
  a213=(a75*a210);
  a199=(a199+a213);
  a213=(a109*a198);
  a214=(a88*a212);
  a213=(a213+a214);
  a213=(a213+a169);
  a213=(a213-a208);
  a213=(a81*a213);
  a199=(a199+a213);
  a211=(a83*a211);
  a199=(a199+a211);
  a211=(a82*a212);
  a189=(a108*a189);
  a211=(a211-a189);
  a189=(a93*a206);
  a180=(a180+a189);
  a213=(a73*a180);
  a211=(a211+a213);
  a80=(a80+a80);
  a209=(a80*a209);
  a96=(a96+a96);
  a70=(a96*a70);
  a213=(a209-a70);
  a200=(a107*a200);
  a206=(a86*a206);
  a200=(a200+a206);
  a213=(a213+a200);
  a206=(a73*a213);
  a211=(a211+a206);
  a206=(a78*a211);
  a199=(a199-a206);
  a199=(a177*a199);
  a183=(a183-a199);
  a199=cos(a6);
  a206=(a78*a117);
  a208=(a75*a197);
  a206=(a206-a208);
  a189=(a83*a189);
  a206=(a206+a189);
  a189=(a78*a210);
  a206=(a206+a189);
  a209=(a209-a70);
  a209=(a209+a200);
  a209=(a83*a209);
  a206=(a206+a209);
  a198=(a105*a198);
  a212=(a77*a212);
  a198=(a198+a212);
  a198=(a198-a180);
  a198=(a198-a213);
  a198=(a81*a198);
  a206=(a206+a198);
  a198=(a75*a211);
  a206=(a206+a198);
  a206=(a199*a206);
  a183=(a183+a206);
  a206=sin(a6);
  a198=(a51+a51);
  a39=(a56*a39);
  a213=(a192*a39);
  a213=(a213*a58);
  a213=(a31*a213);
  a213=(a191*a213);
  a213=(a71?a213:0);
  a59=(a59+a59);
  a180=(a66*a39);
  a180=(a57*a180);
  a180=(a60*a180);
  a180=(a194*a180);
  a180=(a59*a180);
  a180=(a191*a180);
  a213=(a213-a180);
  a213=(a190*a213);
  a49=(a49+a49);
  a213=(a213/a49);
  a180=(a198*a213);
  a212=(a51+a51);
  a209=(a53/a55);
  a200=(a0*a72);
  a70=(a41*a39);
  a70=(a174*a70);
  a39=(a61*a39);
  a68=(a68/a69);
  a189=(a39*a68);
  a189=(a176*a189);
  a189=(a41*a189);
  a189=(a67?a189:0);
  a70=(a70+a189);
  a189=(a39/a69);
  a189=(a176*a189);
  a189=(a41*a189);
  a189=(a35*a189);
  a189=(-a189);
  a189=(a67?a189:0);
  a70=(a70+a189);
  a64=(a64/a65);
  a189=(a39*a64);
  a189=(a41*a189);
  a208=casadi_sign(a62);
  a189=(a189*a208);
  a189=(-a189);
  a189=(a63?a189:0);
  a70=(a70+a189);
  a39=(a39/a65);
  a39=(a41*a39);
  a62=casadi_sign(a62);
  a39=(a39*a62);
  a39=(-a39);
  a39=(a63?a39:0);
  a70=(a70+a39);
  a39=(a1*a70);
  a200=(a200+a39);
  a39=(a209*a200);
  a189=(a55+a55);
  a39=(a39/a189);
  a169=(a212*a39);
  a180=(a180-a169);
  a200=(a200/a55);
  a169=(a47*a200);
  a180=(a180-a169);
  a180=(a46*a180);
  a169=(a50*a200);
  a169=(a46*a169);
  a180=(a180+a169);
  a180=(a206*a180);
  a183=(a183-a180);
  a180=cos(a6);
  a48=(a48+a48);
  a213=(a48*a213);
  a54=(a54+a54);
  a39=(a54*a39);
  a213=(a213-a39);
  a39=(a52*a200);
  a213=(a213+a39);
  a213=(a46*a213);
  a200=(a51*a200);
  a200=(a46*a200);
  a213=(a213-a200);
  a213=(a180*a213);
  a183=(a183+a213);
  a213=sin(a6);
  a200=(a11+a11);
  a4=(a16*a4);
  a39=(a192*a4);
  a39=(a39*a19);
  a39=(a31*a39);
  a39=(a191*a39);
  a39=(a40?a39:0);
  a22=(a22+a22);
  a169=(a33*a4);
  a169=(a17*a169);
  a169=(a24*a169);
  a169=(a194*a169);
  a169=(a22*a169);
  a169=(a191*a169);
  a39=(a39-a169);
  a39=(a190*a39);
  a9=(a9+a9);
  a39=(a39/a9);
  a169=(a200*a39);
  a214=(a11+a11);
  a215=(a13/a15);
  a0=(a0*a44);
  a216=(a41*a4);
  a216=(a174*a216);
  a4=(a26*a4);
  a36=(a36/a38);
  a217=(a4*a36);
  a217=(a176*a217);
  a217=(a41*a217);
  a217=(a34?a217:0);
  a216=(a216+a217);
  a217=(a4/a38);
  a217=(a176*a217);
  a217=(a41*a217);
  a217=(a35*a217);
  a217=(-a217);
  a217=(a34?a217:0);
  a216=(a216+a217);
  a30=(a30/a32);
  a217=(a4*a30);
  a217=(a41*a217);
  a218=casadi_sign(a27);
  a217=(a217*a218);
  a217=(-a217);
  a217=(a29?a217:0);
  a216=(a216+a217);
  a4=(a4/a32);
  a4=(a41*a4);
  a27=casadi_sign(a27);
  a4=(a4*a27);
  a4=(-a4);
  a4=(a29?a4:0);
  a216=(a216+a4);
  a4=(a1*a216);
  a0=(a0+a4);
  a4=(a215*a0);
  a217=(a15+a15);
  a4=(a4/a217);
  a219=(a214*a4);
  a169=(a169-a219);
  a0=(a0/a15);
  a219=(a7*a0);
  a169=(a169-a219);
  a169=(a5*a169);
  a219=(a10*a0);
  a219=(a5*a219);
  a169=(a169+a219);
  a169=(a213*a169);
  a183=(a183-a169);
  a6=cos(a6);
  a8=(a8+a8);
  a39=(a8*a39);
  a14=(a14+a14);
  a4=(a14*a4);
  a39=(a39-a4);
  a4=(a12*a0);
  a39=(a39+a4);
  a39=(a5*a39);
  a0=(a11*a0);
  a0=(a5*a0);
  a39=(a39-a0);
  a39=(a6*a39);
  a183=(a183+a39);
  if (res[1]!=0) res[1][8]=a183;
  a183=(a170*a43);
  a144=(a144*a172);
  a172=(a41*a144);
  a172=(a174*a172);
  a25=(a25*a144);
  a157=(a25*a157);
  a157=(a176*a157);
  a157=(a41*a157);
  a157=(a156?a157:0);
  a172=(a172+a157);
  a158=(a25/a158);
  a158=(a176*a158);
  a158=(a41*a158);
  a158=(a35*a158);
  a158=(-a158);
  a156=(a156?a158:0);
  a172=(a172+a156);
  a139=(a25*a139);
  a139=(a41*a139);
  a139=(a139*a179);
  a139=(-a139);
  a139=(a28?a139:0);
  a172=(a172+a139);
  a25=(a25/a154);
  a25=(a41*a25);
  a25=(a25*a147);
  a25=(-a25);
  a28=(a28?a25:0);
  a172=(a172+a28);
  a28=(a2*a172);
  a183=(a183+a28);
  a28=(a183/a142);
  a25=(a138*a28);
  a147=(a133*a25);
  a154=(a130*a28);
  a139=(a123*a154);
  a147=(a147-a139);
  a139=(a127*a147);
  a179=(a149*a154);
  a139=(a139+a179);
  a179=(a124*a139);
  a156=(a132*a147);
  a149=(a149*a25);
  a156=(a156-a149);
  a149=(a121*a156);
  a179=(a179-a149);
  a43=(a3*a43);
  a149=(a1*a172);
  a43=(a43+a149);
  a142=(a43/a142);
  a138=(a138*a142);
  a149=(a128*a138);
  a179=(a179+a149);
  a149=(a151*a25);
  a158=(a148*a154);
  a157=(a152*a25);
  a158=(a158-a157);
  a132=(a132*a158);
  a149=(a149+a132);
  a132=(a192*a144);
  a132=(a132*a137);
  a132=(a31*a132);
  a132=(a191*a132);
  a18=(a18?a132:0);
  a155=(a155*a144);
  a145=(a145*a155);
  a146=(a146*a145);
  a146=(a194*a146);
  a20=(a20*a146);
  a20=(a191*a20);
  a18=(a18-a20);
  a18=(a190*a18);
  a18=(a18/a129);
  a136=(a136*a18);
  a195=(a195*a183);
  a196=(a196*a43);
  a195=(a195+a196);
  a195=(a195/a185);
  a143=(a143*a195);
  a185=(a136-a143);
  a153=(a153*a28);
  a140=(a140*a142);
  a153=(a153+a140);
  a185=(a185+a153);
  a140=(a120*a185);
  a149=(a149-a140);
  a130=(a130*a142);
  a140=(a154+a130);
  a196=(a120*a140);
  a149=(a149+a196);
  a196=(a124*a149);
  a179=(a179+a196);
  a126=(a126*a18);
  a141=(a141*a195);
  a195=(a126-a141);
  a150=(a150*a28);
  a131=(a131*a142);
  a150=(a150+a131);
  a195=(a195+a150);
  a195=(a128*a195);
  a179=(a179+a195);
  a148=(a148*a147);
  a123=(a123*a158);
  a148=(a148+a123);
  a25=(a25+a138);
  a148=(a148-a25);
  a126=(a126-a141);
  a126=(a126+a150);
  a148=(a148-a126);
  a148=(a81*a148);
  a179=(a179+a148);
  a127=(a127*a158);
  a151=(a151*a154);
  a127=(a127-a151);
  a25=(a120*a25);
  a127=(a127+a25);
  a120=(a120*a126);
  a127=(a127+a120);
  a120=(a121*a127);
  a179=(a179+a120);
  a37=(a37*a179);
  a179=-4.8780487804877992e-01;
  a120=(a179*a42);
  a126=(a42*a120);
  a25=(a179*a21);
  a151=(a21*a25);
  a126=(a126+a151);
  a126=(a159*a126);
  a193=(a193*a126);
  a126=(a42*a25);
  a151=(a21*a120);
  a126=(a126-a151);
  a159=(a159*a126);
  a187=(a187*a159);
  a193=(a193+a187);
  a187=(a124*a156);
  a159=(a121*a139);
  a187=(a187+a159);
  a143=(a143-a136);
  a143=(a143-a153);
  a143=(a128*a143);
  a187=(a187+a143);
  a121=(a121*a149);
  a187=(a187+a121);
  a152=(a152*a147);
  a133=(a133*a158);
  a152=(a152+a133);
  a152=(a152+a185);
  a152=(a152-a140);
  a152=(a81*a152);
  a187=(a187+a152);
  a128=(a128*a130);
  a187=(a187+a128);
  a124=(a124*a127);
  a187=(a187-a124);
  a186=(a186*a187);
  a193=(a193+a186);
  a37=(a37-a193);
  a170=(a170*a119);
  a99=(a99*a171);
  a171=(a41*a99);
  a171=(a174*a171);
  a103=(a103*a99);
  a115=(a103*a115);
  a115=(a176*a115);
  a115=(a41*a115);
  a115=(a114?a115:0);
  a171=(a171+a115);
  a116=(a103/a116);
  a116=(a176*a116);
  a116=(a41*a116);
  a116=(a35*a116);
  a116=(-a116);
  a114=(a114?a116:0);
  a171=(a171+a114);
  a111=(a103*a111);
  a111=(a41*a111);
  a111=(a111*a201);
  a111=(-a111);
  a111=(a94?a111:0);
  a171=(a171+a111);
  a103=(a103/a112);
  a103=(a41*a103);
  a103=(a103*a104);
  a103=(-a103);
  a94=(a94?a103:0);
  a171=(a171+a94);
  a94=(a2*a171);
  a170=(a170+a94);
  a94=(a170/a97);
  a103=(a93*a94);
  a104=(a88*a103);
  a112=(a85*a94);
  a111=(a77*a112);
  a104=(a104-a111);
  a111=(a87*a104);
  a201=(a106*a103);
  a111=(a111-a201);
  a201=(a78*a111);
  a114=(a82*a104);
  a106=(a106*a112);
  a114=(a114+a106);
  a106=(a75*a114);
  a201=(a201+a106);
  a204=(a204*a170);
  a119=(a3*a119);
  a170=(a1*a171);
  a119=(a119+a170);
  a205=(a205*a119);
  a204=(a204+a205);
  a204=(a204/a207);
  a98=(a98*a204);
  a207=(a192*a99);
  a207=(a207*a92);
  a207=(a31*a207);
  a207=(a191*a207);
  a118=(a118?a207:0);
  a113=(a113*a99);
  a100=(a100*a113);
  a102=(a102*a100);
  a102=(a194*a102);
  a101=(a101*a102);
  a101=(a191*a101);
  a118=(a118-a101);
  a118=(a190*a118);
  a118=(a118/a84);
  a91=(a91*a118);
  a84=(a98-a91);
  a110=(a110*a94);
  a119=(a119/a97);
  a95=(a95*a119);
  a110=(a110+a95);
  a84=(a84-a110);
  a84=(a83*a84);
  a201=(a201+a84);
  a84=(a108*a103);
  a95=(a105*a112);
  a97=(a109*a103);
  a95=(a95-a97);
  a87=(a87*a95);
  a84=(a84+a87);
  a91=(a91-a98);
  a91=(a91+a110);
  a110=(a73*a91);
  a84=(a84-a110);
  a85=(a85*a119);
  a110=(a112+a85);
  a98=(a73*a110);
  a84=(a84+a98);
  a98=(a75*a84);
  a201=(a201+a98);
  a109=(a109*a104);
  a88=(a88*a95);
  a109=(a109+a88);
  a109=(a109+a91);
  a109=(a109-a110);
  a109=(a81*a109);
  a201=(a201+a109);
  a85=(a83*a85);
  a201=(a201+a85);
  a82=(a82*a95);
  a108=(a108*a112);
  a82=(a82-a108);
  a93=(a93*a119);
  a103=(a103+a93);
  a108=(a73*a103);
  a82=(a82+a108);
  a80=(a80*a118);
  a96=(a96*a204);
  a204=(a80-a96);
  a107=(a107*a94);
  a86=(a86*a119);
  a107=(a107+a86);
  a204=(a204+a107);
  a73=(a73*a204);
  a82=(a82+a73);
  a73=(a78*a82);
  a201=(a201-a73);
  a177=(a177*a201);
  a37=(a37-a177);
  a177=(a78*a114);
  a201=(a75*a111);
  a177=(a177-a201);
  a93=(a83*a93);
  a177=(a177+a93);
  a78=(a78*a84);
  a177=(a177+a78);
  a80=(a80-a96);
  a80=(a80+a107);
  a83=(a83*a80);
  a177=(a177+a83);
  a105=(a105*a104);
  a77=(a77*a95);
  a105=(a105+a77);
  a105=(a105-a103);
  a105=(a105-a204);
  a105=(a81*a105);
  a177=(a177+a105);
  a75=(a75*a82);
  a177=(a177+a75);
  a199=(a199*a177);
  a37=(a37+a199);
  a56=(a56*a45);
  a45=(a192*a56);
  a45=(a45*a58);
  a45=(a31*a45);
  a45=(a191*a45);
  a71=(a71?a45:0);
  a66=(a66*a56);
  a57=(a57*a66);
  a60=(a60*a57);
  a60=(a194*a60);
  a59=(a59*a60);
  a59=(a191*a59);
  a71=(a71-a59);
  a71=(a190*a71);
  a71=(a71/a49);
  a198=(a198*a71);
  a72=(a3*a72);
  a49=(a41*a56);
  a49=(a174*a49);
  a61=(a61*a56);
  a68=(a61*a68);
  a68=(a176*a68);
  a68=(a41*a68);
  a68=(a67?a68:0);
  a49=(a49+a68);
  a69=(a61/a69);
  a69=(a176*a69);
  a69=(a41*a69);
  a69=(a35*a69);
  a69=(-a69);
  a67=(a67?a69:0);
  a49=(a49+a67);
  a64=(a61*a64);
  a64=(a41*a64);
  a64=(a64*a208);
  a64=(-a64);
  a64=(a63?a64:0);
  a49=(a49+a64);
  a61=(a61/a65);
  a61=(a41*a61);
  a61=(a61*a62);
  a61=(-a61);
  a63=(a63?a61:0);
  a49=(a49+a63);
  a63=(a1*a49);
  a72=(a72+a63);
  a209=(a209*a72);
  a209=(a209/a189);
  a212=(a212*a209);
  a198=(a198-a212);
  a72=(a72/a55);
  a47=(a47*a72);
  a198=(a198-a47);
  a198=(a46*a198);
  a50=(a50*a72);
  a50=(a46*a50);
  a198=(a198+a50);
  a206=(a206*a198);
  a37=(a37-a206);
  a48=(a48*a71);
  a54=(a54*a209);
  a48=(a48-a54);
  a52=(a52*a72);
  a48=(a48+a52);
  a48=(a46*a48);
  a51=(a51*a72);
  a46=(a46*a51);
  a48=(a48-a46);
  a180=(a180*a48);
  a37=(a37+a180);
  a16=(a16*a166);
  a192=(a192*a16);
  a192=(a192*a19);
  a31=(a31*a192);
  a31=(a191*a31);
  a40=(a40?a31:0);
  a33=(a33*a16);
  a17=(a17*a33);
  a24=(a24*a17);
  a194=(a194*a24);
  a22=(a22*a194);
  a191=(a191*a22);
  a40=(a40-a191);
  a190=(a190*a40);
  a190=(a190/a9);
  a200=(a200*a190);
  a3=(a3*a44);
  a44=(a41*a16);
  a174=(a174*a44);
  a26=(a26*a16);
  a36=(a26*a36);
  a36=(a176*a36);
  a36=(a41*a36);
  a36=(a34?a36:0);
  a174=(a174+a36);
  a38=(a26/a38);
  a176=(a176*a38);
  a176=(a41*a176);
  a35=(a35*a176);
  a35=(-a35);
  a34=(a34?a35:0);
  a174=(a174+a34);
  a30=(a26*a30);
  a30=(a41*a30);
  a30=(a30*a218);
  a30=(-a30);
  a30=(a29?a30:0);
  a174=(a174+a30);
  a26=(a26/a32);
  a41=(a41*a26);
  a41=(a41*a27);
  a41=(-a41);
  a29=(a29?a41:0);
  a174=(a174+a29);
  a1=(a1*a174);
  a3=(a3+a1);
  a215=(a215*a3);
  a215=(a215/a217);
  a214=(a214*a215);
  a200=(a200-a214);
  a3=(a3/a15);
  a7=(a7*a3);
  a200=(a200-a7);
  a200=(a5*a200);
  a10=(a10*a3);
  a10=(a5*a10);
  a200=(a200+a10);
  a213=(a213*a200);
  a37=(a37-a213);
  a8=(a8*a190);
  a14=(a14*a215);
  a8=(a8-a14);
  a12=(a12*a3);
  a8=(a8+a12);
  a8=(a5*a8);
  a11=(a11*a3);
  a5=(a5*a11);
  a8=(a8-a5);
  a6=(a6*a8);
  a37=(a37+a6);
  if (res[1]!=0) res[1][9]=a37;
  a37=cos(a74);
  a6=(a165*a181);
  a8=(a2*a181);
  a6=(a6-a8);
  a8=(a163*a6);
  a5=(a160*a181);
  a8=(a8-a5);
  a5=(a178*a161);
  a8=(a8+a5);
  a5=(a162*a173);
  a8=(a8-a5);
  a8=(a37*a8);
  a5=sin(a74);
  a178=(a178*a168);
  a11=(a162*a181);
  a178=(a178-a11);
  a11=(a2*a173);
  a3=(a165*a173);
  a11=(a11-a3);
  a3=(a163*a11);
  a178=(a178+a3);
  a3=(a160*a173);
  a178=(a178+a3);
  a178=(a5*a178);
  a8=(a8-a178);
  a178=sin(a74);
  a3=(a125*a184);
  a12=(a122*a182);
  a3=(a3+a12);
  a12=(a122*a188);
  a3=(a3+a12);
  a12=(a125*a202);
  a3=(a3-a12);
  a3=(a178*a3);
  a8=(a8-a3);
  a3=cos(a74);
  a182=(a125*a182);
  a184=(a122*a184);
  a182=(a182-a184);
  a188=(a125*a188);
  a182=(a182+a188);
  a202=(a122*a202);
  a182=(a182+a202);
  a182=(a3*a182);
  a8=(a8+a182);
  a182=sin(a74);
  a202=(a79*a197);
  a188=(a76*a117);
  a202=(a202+a188);
  a188=(a76*a210);
  a202=(a202+a188);
  a188=(a79*a211);
  a202=(a202-a188);
  a202=(a182*a202);
  a8=(a8-a202);
  a74=cos(a74);
  a117=(a79*a117);
  a197=(a76*a197);
  a117=(a117-a197);
  a210=(a79*a210);
  a117=(a117+a210);
  a211=(a76*a211);
  a117=(a117+a211);
  a117=(a74*a117);
  a8=(a8+a117);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a165*a120);
  a117=(a2*a120);
  a8=(a8-a117);
  a117=(a163*a8);
  a211=(a160*a120);
  a117=(a117-a211);
  a161=(a179*a161);
  a117=(a117+a161);
  a161=(a162*a25);
  a117=(a117-a161);
  a37=(a37*a117);
  a179=(a179*a168);
  a162=(a162*a120);
  a179=(a179-a162);
  a2=(a2*a25);
  a165=(a165*a25);
  a2=(a2-a165);
  a163=(a163*a2);
  a179=(a179+a163);
  a160=(a160*a25);
  a179=(a179+a160);
  a5=(a5*a179);
  a37=(a37-a5);
  a5=(a125*a156);
  a179=(a122*a139);
  a5=(a5+a179);
  a179=(a122*a149);
  a5=(a5+a179);
  a179=(a125*a127);
  a5=(a5-a179);
  a178=(a178*a5);
  a37=(a37-a178);
  a139=(a125*a139);
  a156=(a122*a156);
  a139=(a139-a156);
  a125=(a125*a149);
  a139=(a139+a125);
  a122=(a122*a127);
  a139=(a139+a122);
  a3=(a3*a139);
  a37=(a37+a3);
  a3=(a79*a111);
  a139=(a76*a114);
  a3=(a3+a139);
  a139=(a76*a84);
  a3=(a3+a139);
  a139=(a79*a82);
  a3=(a3-a139);
  a182=(a182*a3);
  a37=(a37-a182);
  a114=(a79*a114);
  a111=(a76*a111);
  a114=(a114-a111);
  a79=(a79*a84);
  a114=(a114+a79);
  a76=(a76*a82);
  a114=(a114+a76);
  a74=(a74*a114);
  a37=(a37+a74);
  if (res[1]!=0) res[1][11]=a37;
  a37=-1.;
  if (res[1]!=0) res[1][12]=a37;
  a74=(a167*a181);
  a114=(a164*a173);
  a74=(a74-a114);
  a6=(a21*a6);
  a11=(a42*a11);
  a6=(a6+a11);
  a6=(a81*a6);
  a6=(a74+a6);
  a11=(a135*a175);
  a6=(a6+a11);
  a11=(a90*a203);
  a6=(a6+a11);
  a70=(a53*a70);
  a6=(a6+a70);
  a216=(a13*a216);
  a6=(a6+a216);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a167*a120);
  a216=(a164*a25);
  a6=(a6-a216);
  a21=(a21*a8);
  a42=(a42*a2);
  a21=(a21+a42);
  a81=(a81*a21);
  a81=(a6+a81);
  a135=(a135*a172);
  a81=(a81+a135);
  a90=(a90*a171);
  a81=(a81+a90);
  a53=(a53*a49);
  a81=(a81+a53);
  a13=(a13*a174);
  a81=(a81+a13);
  if (res[1]!=0) res[1][14]=a81;
  if (res[1]!=0) res[1][15]=a37;
  a181=(a167*a181);
  a74=(a74-a181);
  a173=(a164*a173);
  a74=(a74+a173);
  a175=(a134*a175);
  a74=(a74+a175);
  a203=(a89*a203);
  a74=(a74+a203);
  if (res[1]!=0) res[1][16]=a74;
  a167=(a167*a120);
  a6=(a6-a167);
  a164=(a164*a25);
  a6=(a6+a164);
  a134=(a134*a172);
  a6=(a6+a134);
  a89=(a89*a171);
  a6=(a6+a89);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a23;
  if (res[2]!=0) res[2][1]=a23;
  if (res[2]!=0) res[2][2]=a23;
  if (res[2]!=0) res[2][3]=a23;
  if (res[2]!=0) res[2][4]=a23;
  if (res[2]!=0) res[2][5]=a23;
  if (res[2]!=0) res[2][6]=a23;
  if (res[2]!=0) res[2][7]=a23;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15222092_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
