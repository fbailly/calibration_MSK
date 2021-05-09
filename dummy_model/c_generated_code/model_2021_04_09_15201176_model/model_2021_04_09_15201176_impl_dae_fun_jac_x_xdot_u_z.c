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
  #define CASADI_PREFIX(ID) model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_ ## ID
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

/* model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x4,8nz],o4[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a215, a216, a217, a218, a219, a22, a220, a221, a222, a223, a224, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a5=3.9882907436235976e-01;
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
  a46=4.2676442459727137e-01;
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
  a73=5.7464607548862467e-01;
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
  a120=4.9714582882814273e-01;
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
  a166=(a4*a45);
  a169=9.6278838983177639e-01;
  a170=(a89*a119);
  a171=(a134*a43);
  a170=(a170+a171);
  a171=(a169*a170);
  a166=(a166+a171);
  a171=6.9253199970355839e-01;
  a166=(a166/a171);
  a171=(a3*a166);
  a172=9.6278838983177628e-01;
  a45=(a172*a45);
  a173=2.7025639012821789e-01;
  a170=(a173*a170);
  a45=(a45+a170);
  a171=(a171-a45);
  a45=3.7001900289039211e+00;
  a171=(a171/a45);
  a0=(a0-a171);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a166);
  if (res[0]!=0) res[0][7]=a0;
  a0=3.9024390243902418e-01;
  a166=(a0*a13);
  a171=(a39*a166);
  if (res[1]!=0) res[1][0]=a171;
  a171=-3.9024390243902396e-01;
  a45=(a171*a13);
  a39=(a39*a45);
  if (res[1]!=0) res[1][1]=a39;
  a39=(a0*a53);
  a170=(a70*a39);
  if (res[1]!=0) res[1][2]=a170;
  a170=(a171*a53);
  a70=(a70*a170);
  if (res[1]!=0) res[1][3]=a70;
  a70=-3.9024390243902440e-01;
  a174=(a70*a89);
  a175=(a0*a90);
  a174=(a174+a175);
  a175=(a117*a174);
  if (res[1]!=0) res[1][4]=a175;
  a175=1.3902439024390245e+00;
  a176=(a175*a89);
  a177=(a171*a90);
  a176=(a176+a177);
  a117=(a117*a176);
  if (res[1]!=0) res[1][5]=a117;
  a117=(a70*a134);
  a177=(a0*a135);
  a117=(a117+a177);
  a177=(a37*a117);
  if (res[1]!=0) res[1][6]=a177;
  a177=(a175*a134);
  a178=(a171*a135);
  a177=(a177+a178);
  a37=(a37*a177);
  if (res[1]!=0) res[1][7]=a37;
  a37=cos(a6);
  a178=(a70*a43);
  a179=1.4285714285714285e-01;
  a117=(a144*a117);
  a180=(a41*a117);
  a180=(a179*a180);
  a181=-1.2121212121212121e+01;
  a182=(a25*a117);
  a157=(a157/a158);
  a183=(a182*a157);
  a183=(a181*a183);
  a183=(a41*a183);
  a183=(a156?a183:0);
  a180=(a180+a183);
  a183=(a182/a158);
  a183=(a181*a183);
  a183=(a41*a183);
  a183=(a35*a183);
  a183=(-a183);
  a183=(a156?a183:0);
  a180=(a180+a183);
  a139=(a139/a154);
  a183=(a182*a139);
  a183=(a41*a183);
  a184=casadi_sign(a147);
  a183=(a183*a184);
  a183=(-a183);
  a183=(a28?a183:0);
  a180=(a180+a183);
  a182=(a182/a154);
  a182=(a41*a182);
  a147=casadi_sign(a147);
  a182=(a182*a147);
  a182=(-a182);
  a182=(a28?a182:0);
  a180=(a180+a182);
  a182=(a2*a180);
  a178=(a178+a182);
  a182=(a178/a142);
  a183=(a138*a182);
  a185=(a133*a183);
  a186=(a130*a182);
  a187=(a123*a186);
  a185=(a185-a187);
  a187=(a127*a185);
  a188=(a149*a186);
  a187=(a187+a188);
  a188=(a124*a187);
  a189=(a132*a185);
  a190=(a149*a183);
  a189=(a189-a190);
  a190=(a121*a189);
  a188=(a188-a190);
  a190=(a0*a43);
  a191=(a1*a180);
  a190=(a190+a191);
  a191=(a190/a142);
  a192=(a138*a191);
  a193=(a128*a192);
  a188=(a188+a193);
  a193=(a151*a183);
  a194=(a148*a186);
  a195=(a152*a183);
  a194=(a194-a195);
  a195=(a132*a194);
  a193=(a193+a195);
  a136=(a136+a136);
  a195=1.1394939273245490e+00;
  a196=1.4285714285714286e+00;
  a197=6.7836549063042314e-03;
  a198=(a197*a117);
  a198=(a198*a137);
  a198=(a31*a198);
  a198=(a196*a198);
  a198=(a18?a198:0);
  a20=(a20+a20);
  a199=2.2222222222222223e+00;
  a117=(a155*a117);
  a117=(a145*a117);
  a117=(a146*a117);
  a117=(a199*a117);
  a117=(a20*a117);
  a117=(a196*a117);
  a198=(a198-a117);
  a198=(a195*a198);
  a129=(a129+a129);
  a198=(a198/a129);
  a117=(a136*a198);
  a143=(a143+a143);
  a200=(a134/a142);
  a178=(a200*a178);
  a201=(a135/a142);
  a190=(a201*a190);
  a178=(a178+a190);
  a190=(a142+a142);
  a178=(a178/a190);
  a202=(a143*a178);
  a203=(a117-a202);
  a204=(a153*a182);
  a205=(a140*a191);
  a204=(a204+a205);
  a203=(a203+a204);
  a205=(a120*a203);
  a193=(a193-a205);
  a205=(a130*a191);
  a206=(a186+a205);
  a207=(a120*a206);
  a193=(a193+a207);
  a207=(a124*a193);
  a188=(a188+a207);
  a126=(a126+a126);
  a198=(a126*a198);
  a141=(a141+a141);
  a178=(a141*a178);
  a207=(a198-a178);
  a182=(a150*a182);
  a191=(a131*a191);
  a182=(a182+a191);
  a207=(a207+a182);
  a207=(a128*a207);
  a188=(a188+a207);
  a207=(a148*a185);
  a191=(a123*a194);
  a207=(a207+a191);
  a183=(a183+a192);
  a207=(a207-a183);
  a198=(a198-a178);
  a198=(a198+a182);
  a207=(a207-a198);
  a207=(a81*a207);
  a188=(a188+a207);
  a207=(a127*a194);
  a186=(a151*a186);
  a207=(a207-a186);
  a183=(a120*a183);
  a207=(a207+a183);
  a198=(a120*a198);
  a207=(a207+a198);
  a198=(a121*a207);
  a188=(a188+a198);
  a188=(a37*a188);
  a198=cos(a6);
  a183=4.8780487804878025e-01;
  a186=(a183*a42);
  a182=(a42*a186);
  a178=(a183*a21);
  a192=(a21*a178);
  a182=(a182+a192);
  a182=(a159*a182);
  a182=(a198*a182);
  a192=sin(a6);
  a191=(a42*a178);
  a208=(a21*a186);
  a191=(a191-a208);
  a191=(a159*a191);
  a191=(a192*a191);
  a182=(a182+a191);
  a191=sin(a6);
  a208=(a124*a189);
  a209=(a121*a187);
  a208=(a208+a209);
  a202=(a202-a117);
  a202=(a202-a204);
  a202=(a128*a202);
  a208=(a208+a202);
  a202=(a121*a193);
  a208=(a208+a202);
  a185=(a152*a185);
  a194=(a133*a194);
  a185=(a185+a194);
  a185=(a185+a203);
  a185=(a185-a206);
  a185=(a81*a185);
  a208=(a208+a185);
  a205=(a128*a205);
  a208=(a208+a205);
  a205=(a124*a207);
  a208=(a208-a205);
  a208=(a191*a208);
  a182=(a182+a208);
  a188=(a188-a182);
  a182=sin(a6);
  a70=(a70*a119);
  a174=(a99*a174);
  a208=(a41*a174);
  a208=(a179*a208);
  a205=(a103*a174);
  a115=(a115/a116);
  a185=(a205*a115);
  a185=(a181*a185);
  a185=(a41*a185);
  a185=(a114?a185:0);
  a208=(a208+a185);
  a185=(a205/a116);
  a185=(a181*a185);
  a185=(a41*a185);
  a185=(a35*a185);
  a185=(-a185);
  a185=(a114?a185:0);
  a208=(a208+a185);
  a111=(a111/a112);
  a185=(a205*a111);
  a185=(a41*a185);
  a206=casadi_sign(a104);
  a185=(a185*a206);
  a185=(-a185);
  a185=(a94?a185:0);
  a208=(a208+a185);
  a205=(a205/a112);
  a205=(a41*a205);
  a104=casadi_sign(a104);
  a205=(a205*a104);
  a205=(-a205);
  a205=(a94?a205:0);
  a208=(a208+a205);
  a205=(a2*a208);
  a70=(a70+a205);
  a205=(a70/a97);
  a185=(a93*a205);
  a203=(a88*a185);
  a194=(a85*a205);
  a202=(a77*a194);
  a203=(a203-a202);
  a202=(a87*a203);
  a204=(a106*a185);
  a202=(a202-a204);
  a204=(a78*a202);
  a117=(a82*a203);
  a209=(a106*a194);
  a117=(a117+a209);
  a209=(a75*a117);
  a204=(a204+a209);
  a98=(a98+a98);
  a209=(a89/a97);
  a70=(a209*a70);
  a210=(a90/a97);
  a211=(a0*a119);
  a212=(a1*a208);
  a211=(a211+a212);
  a212=(a210*a211);
  a70=(a70+a212);
  a212=(a97+a97);
  a70=(a70/a212);
  a213=(a98*a70);
  a91=(a91+a91);
  a214=(a197*a174);
  a214=(a214*a92);
  a214=(a31*a214);
  a214=(a196*a214);
  a214=(a118?a214:0);
  a101=(a101+a101);
  a174=(a113*a174);
  a174=(a100*a174);
  a174=(a102*a174);
  a174=(a199*a174);
  a174=(a101*a174);
  a174=(a196*a174);
  a214=(a214-a174);
  a214=(a195*a214);
  a84=(a84+a84);
  a214=(a214/a84);
  a174=(a91*a214);
  a215=(a213-a174);
  a216=(a110*a205);
  a211=(a211/a97);
  a217=(a95*a211);
  a216=(a216+a217);
  a215=(a215-a216);
  a215=(a83*a215);
  a204=(a204+a215);
  a215=(a108*a185);
  a217=(a105*a194);
  a218=(a109*a185);
  a217=(a217-a218);
  a218=(a87*a217);
  a215=(a215+a218);
  a174=(a174-a213);
  a174=(a174+a216);
  a216=(a73*a174);
  a215=(a215-a216);
  a216=(a85*a211);
  a213=(a194+a216);
  a218=(a73*a213);
  a215=(a215+a218);
  a218=(a75*a215);
  a204=(a204+a218);
  a218=(a109*a203);
  a219=(a88*a217);
  a218=(a218+a219);
  a218=(a218+a174);
  a218=(a218-a213);
  a218=(a81*a218);
  a204=(a204+a218);
  a216=(a83*a216);
  a204=(a204+a216);
  a216=(a82*a217);
  a194=(a108*a194);
  a216=(a216-a194);
  a194=(a93*a211);
  a185=(a185+a194);
  a218=(a73*a185);
  a216=(a216+a218);
  a80=(a80+a80);
  a214=(a80*a214);
  a96=(a96+a96);
  a70=(a96*a70);
  a218=(a214-a70);
  a205=(a107*a205);
  a211=(a86*a211);
  a205=(a205+a211);
  a218=(a218+a205);
  a211=(a73*a218);
  a216=(a216+a211);
  a211=(a78*a216);
  a204=(a204-a211);
  a204=(a182*a204);
  a188=(a188-a204);
  a204=cos(a6);
  a211=(a78*a117);
  a213=(a75*a202);
  a211=(a211-a213);
  a194=(a83*a194);
  a211=(a211+a194);
  a194=(a78*a215);
  a211=(a211+a194);
  a214=(a214-a70);
  a214=(a214+a205);
  a214=(a83*a214);
  a211=(a211+a214);
  a203=(a105*a203);
  a217=(a77*a217);
  a203=(a203+a217);
  a203=(a203-a185);
  a203=(a203-a218);
  a203=(a81*a203);
  a211=(a211+a203);
  a203=(a75*a216);
  a211=(a211+a203);
  a211=(a204*a211);
  a188=(a188+a211);
  a211=sin(a6);
  a203=(a51+a51);
  a39=(a56*a39);
  a218=(a197*a39);
  a218=(a218*a58);
  a218=(a31*a218);
  a218=(a196*a218);
  a218=(a71?a218:0);
  a59=(a59+a59);
  a185=(a66*a39);
  a185=(a57*a185);
  a185=(a60*a185);
  a185=(a199*a185);
  a185=(a59*a185);
  a185=(a196*a185);
  a218=(a218-a185);
  a218=(a195*a218);
  a49=(a49+a49);
  a218=(a218/a49);
  a185=(a203*a218);
  a217=(a51+a51);
  a214=(a53/a55);
  a205=(a0*a72);
  a70=(a41*a39);
  a70=(a179*a70);
  a39=(a61*a39);
  a68=(a68/a69);
  a194=(a39*a68);
  a194=(a181*a194);
  a194=(a41*a194);
  a194=(a67?a194:0);
  a70=(a70+a194);
  a194=(a39/a69);
  a194=(a181*a194);
  a194=(a41*a194);
  a194=(a35*a194);
  a194=(-a194);
  a194=(a67?a194:0);
  a70=(a70+a194);
  a64=(a64/a65);
  a194=(a39*a64);
  a194=(a41*a194);
  a213=casadi_sign(a62);
  a194=(a194*a213);
  a194=(-a194);
  a194=(a63?a194:0);
  a70=(a70+a194);
  a39=(a39/a65);
  a39=(a41*a39);
  a62=casadi_sign(a62);
  a39=(a39*a62);
  a39=(-a39);
  a39=(a63?a39:0);
  a70=(a70+a39);
  a39=(a1*a70);
  a205=(a205+a39);
  a39=(a214*a205);
  a194=(a55+a55);
  a39=(a39/a194);
  a174=(a217*a39);
  a185=(a185-a174);
  a205=(a205/a55);
  a174=(a47*a205);
  a185=(a185-a174);
  a185=(a46*a185);
  a174=(a50*a205);
  a174=(a46*a174);
  a185=(a185+a174);
  a185=(a211*a185);
  a188=(a188-a185);
  a185=cos(a6);
  a48=(a48+a48);
  a218=(a48*a218);
  a54=(a54+a54);
  a39=(a54*a39);
  a218=(a218-a39);
  a39=(a52*a205);
  a218=(a218+a39);
  a218=(a46*a218);
  a205=(a51*a205);
  a205=(a46*a205);
  a218=(a218-a205);
  a218=(a185*a218);
  a188=(a188+a218);
  a218=sin(a6);
  a205=(a11+a11);
  a166=(a16*a166);
  a39=(a197*a166);
  a39=(a39*a19);
  a39=(a31*a39);
  a39=(a196*a39);
  a39=(a40?a39:0);
  a22=(a22+a22);
  a174=(a33*a166);
  a174=(a17*a174);
  a174=(a24*a174);
  a174=(a199*a174);
  a174=(a22*a174);
  a174=(a196*a174);
  a39=(a39-a174);
  a39=(a195*a39);
  a9=(a9+a9);
  a39=(a39/a9);
  a174=(a205*a39);
  a219=(a11+a11);
  a220=(a13/a15);
  a0=(a0*a44);
  a221=(a41*a166);
  a221=(a179*a221);
  a166=(a26*a166);
  a36=(a36/a38);
  a222=(a166*a36);
  a222=(a181*a222);
  a222=(a41*a222);
  a222=(a34?a222:0);
  a221=(a221+a222);
  a222=(a166/a38);
  a222=(a181*a222);
  a222=(a41*a222);
  a222=(a35*a222);
  a222=(-a222);
  a222=(a34?a222:0);
  a221=(a221+a222);
  a30=(a30/a32);
  a222=(a166*a30);
  a222=(a41*a222);
  a223=casadi_sign(a27);
  a222=(a222*a223);
  a222=(-a222);
  a222=(a29?a222:0);
  a221=(a221+a222);
  a166=(a166/a32);
  a166=(a41*a166);
  a27=casadi_sign(a27);
  a166=(a166*a27);
  a166=(-a166);
  a166=(a29?a166:0);
  a221=(a221+a166);
  a166=(a1*a221);
  a0=(a0+a166);
  a166=(a220*a0);
  a222=(a15+a15);
  a166=(a166/a222);
  a224=(a219*a166);
  a174=(a174-a224);
  a0=(a0/a15);
  a224=(a7*a0);
  a174=(a174-a224);
  a174=(a5*a174);
  a224=(a10*a0);
  a224=(a5*a224);
  a174=(a174+a224);
  a174=(a218*a174);
  a188=(a188-a174);
  a6=cos(a6);
  a8=(a8+a8);
  a39=(a8*a39);
  a14=(a14+a14);
  a166=(a14*a166);
  a39=(a39-a166);
  a166=(a12*a0);
  a39=(a39+a166);
  a39=(a5*a39);
  a0=(a11*a0);
  a0=(a5*a0);
  a39=(a39-a0);
  a39=(a6*a39);
  a188=(a188+a39);
  if (res[1]!=0) res[1][8]=a188;
  a188=(a175*a43);
  a177=(a144*a177);
  a39=(a41*a177);
  a39=(a179*a39);
  a25=(a25*a177);
  a157=(a25*a157);
  a157=(a181*a157);
  a157=(a41*a157);
  a157=(a156?a157:0);
  a39=(a39+a157);
  a158=(a25/a158);
  a158=(a181*a158);
  a158=(a41*a158);
  a158=(a35*a158);
  a158=(-a158);
  a156=(a156?a158:0);
  a39=(a39+a156);
  a139=(a25*a139);
  a139=(a41*a139);
  a139=(a139*a184);
  a139=(-a139);
  a139=(a28?a139:0);
  a39=(a39+a139);
  a25=(a25/a154);
  a25=(a41*a25);
  a25=(a25*a147);
  a25=(-a25);
  a28=(a28?a25:0);
  a39=(a39+a28);
  a28=(a2*a39);
  a188=(a188+a28);
  a28=(a188/a142);
  a25=(a138*a28);
  a147=(a133*a25);
  a154=(a130*a28);
  a139=(a123*a154);
  a147=(a147-a139);
  a139=(a127*a147);
  a184=(a149*a154);
  a139=(a139+a184);
  a184=(a124*a139);
  a156=(a132*a147);
  a149=(a149*a25);
  a156=(a156-a149);
  a149=(a121*a156);
  a184=(a184-a149);
  a43=(a171*a43);
  a149=(a1*a39);
  a43=(a43+a149);
  a142=(a43/a142);
  a138=(a138*a142);
  a149=(a128*a138);
  a184=(a184+a149);
  a149=(a151*a25);
  a158=(a148*a154);
  a157=(a152*a25);
  a158=(a158-a157);
  a132=(a132*a158);
  a149=(a149+a132);
  a132=(a197*a177);
  a132=(a132*a137);
  a132=(a31*a132);
  a132=(a196*a132);
  a18=(a18?a132:0);
  a177=(a155*a177);
  a145=(a145*a177);
  a145=(a146*a145);
  a145=(a199*a145);
  a20=(a20*a145);
  a20=(a196*a20);
  a18=(a18-a20);
  a18=(a195*a18);
  a18=(a18/a129);
  a136=(a136*a18);
  a200=(a200*a188);
  a201=(a201*a43);
  a200=(a200+a201);
  a200=(a200/a190);
  a143=(a143*a200);
  a190=(a136-a143);
  a153=(a153*a28);
  a140=(a140*a142);
  a153=(a153+a140);
  a190=(a190+a153);
  a140=(a120*a190);
  a149=(a149-a140);
  a130=(a130*a142);
  a140=(a154+a130);
  a201=(a120*a140);
  a149=(a149+a201);
  a201=(a124*a149);
  a184=(a184+a201);
  a126=(a126*a18);
  a141=(a141*a200);
  a200=(a126-a141);
  a150=(a150*a28);
  a131=(a131*a142);
  a150=(a150+a131);
  a200=(a200+a150);
  a200=(a128*a200);
  a184=(a184+a200);
  a148=(a148*a147);
  a123=(a123*a158);
  a148=(a148+a123);
  a25=(a25+a138);
  a148=(a148-a25);
  a126=(a126-a141);
  a126=(a126+a150);
  a148=(a148-a126);
  a148=(a81*a148);
  a184=(a184+a148);
  a127=(a127*a158);
  a151=(a151*a154);
  a127=(a127-a151);
  a25=(a120*a25);
  a127=(a127+a25);
  a120=(a120*a126);
  a127=(a127+a120);
  a120=(a121*a127);
  a184=(a184+a120);
  a37=(a37*a184);
  a184=-4.8780487804877992e-01;
  a120=(a184*a42);
  a126=(a42*a120);
  a25=(a184*a21);
  a151=(a21*a25);
  a126=(a126+a151);
  a126=(a159*a126);
  a198=(a198*a126);
  a126=(a42*a25);
  a151=(a21*a120);
  a126=(a126-a151);
  a159=(a159*a126);
  a192=(a192*a159);
  a198=(a198+a192);
  a192=(a124*a156);
  a159=(a121*a139);
  a192=(a192+a159);
  a143=(a143-a136);
  a143=(a143-a153);
  a143=(a128*a143);
  a192=(a192+a143);
  a121=(a121*a149);
  a192=(a192+a121);
  a152=(a152*a147);
  a133=(a133*a158);
  a152=(a152+a133);
  a152=(a152+a190);
  a152=(a152-a140);
  a152=(a81*a152);
  a192=(a192+a152);
  a128=(a128*a130);
  a192=(a192+a128);
  a124=(a124*a127);
  a192=(a192-a124);
  a191=(a191*a192);
  a198=(a198+a191);
  a37=(a37-a198);
  a175=(a175*a119);
  a176=(a99*a176);
  a198=(a41*a176);
  a198=(a179*a198);
  a103=(a103*a176);
  a115=(a103*a115);
  a115=(a181*a115);
  a115=(a41*a115);
  a115=(a114?a115:0);
  a198=(a198+a115);
  a116=(a103/a116);
  a116=(a181*a116);
  a116=(a41*a116);
  a116=(a35*a116);
  a116=(-a116);
  a114=(a114?a116:0);
  a198=(a198+a114);
  a111=(a103*a111);
  a111=(a41*a111);
  a111=(a111*a206);
  a111=(-a111);
  a111=(a94?a111:0);
  a198=(a198+a111);
  a103=(a103/a112);
  a103=(a41*a103);
  a103=(a103*a104);
  a103=(-a103);
  a94=(a94?a103:0);
  a198=(a198+a94);
  a94=(a2*a198);
  a175=(a175+a94);
  a94=(a175/a97);
  a103=(a93*a94);
  a104=(a88*a103);
  a112=(a85*a94);
  a111=(a77*a112);
  a104=(a104-a111);
  a111=(a87*a104);
  a206=(a106*a103);
  a111=(a111-a206);
  a206=(a78*a111);
  a114=(a82*a104);
  a106=(a106*a112);
  a114=(a114+a106);
  a106=(a75*a114);
  a206=(a206+a106);
  a209=(a209*a175);
  a119=(a171*a119);
  a175=(a1*a198);
  a119=(a119+a175);
  a210=(a210*a119);
  a209=(a209+a210);
  a209=(a209/a212);
  a98=(a98*a209);
  a212=(a197*a176);
  a212=(a212*a92);
  a212=(a31*a212);
  a212=(a196*a212);
  a118=(a118?a212:0);
  a176=(a113*a176);
  a100=(a100*a176);
  a100=(a102*a100);
  a100=(a199*a100);
  a101=(a101*a100);
  a101=(a196*a101);
  a118=(a118-a101);
  a118=(a195*a118);
  a118=(a118/a84);
  a91=(a91*a118);
  a84=(a98-a91);
  a110=(a110*a94);
  a119=(a119/a97);
  a95=(a95*a119);
  a110=(a110+a95);
  a84=(a84-a110);
  a84=(a83*a84);
  a206=(a206+a84);
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
  a206=(a206+a98);
  a109=(a109*a104);
  a88=(a88*a95);
  a109=(a109+a88);
  a109=(a109+a91);
  a109=(a109-a110);
  a109=(a81*a109);
  a206=(a206+a109);
  a85=(a83*a85);
  a206=(a206+a85);
  a82=(a82*a95);
  a108=(a108*a112);
  a82=(a82-a108);
  a93=(a93*a119);
  a103=(a103+a93);
  a108=(a73*a103);
  a82=(a82+a108);
  a80=(a80*a118);
  a96=(a96*a209);
  a209=(a80-a96);
  a107=(a107*a94);
  a86=(a86*a119);
  a107=(a107+a86);
  a209=(a209+a107);
  a73=(a73*a209);
  a82=(a82+a73);
  a73=(a78*a82);
  a206=(a206-a73);
  a182=(a182*a206);
  a37=(a37-a182);
  a182=(a78*a114);
  a206=(a75*a111);
  a182=(a182-a206);
  a93=(a83*a93);
  a182=(a182+a93);
  a78=(a78*a84);
  a182=(a182+a78);
  a80=(a80-a96);
  a80=(a80+a107);
  a83=(a83*a80);
  a182=(a182+a83);
  a105=(a105*a104);
  a77=(a77*a95);
  a105=(a105+a77);
  a105=(a105-a103);
  a105=(a105-a209);
  a105=(a81*a105);
  a182=(a182+a105);
  a75=(a75*a82);
  a182=(a182+a75);
  a204=(a204*a182);
  a37=(a37+a204);
  a170=(a56*a170);
  a204=(a197*a170);
  a204=(a204*a58);
  a204=(a31*a204);
  a204=(a196*a204);
  a71=(a71?a204:0);
  a204=(a66*a170);
  a57=(a57*a204);
  a57=(a60*a57);
  a57=(a199*a57);
  a59=(a59*a57);
  a59=(a196*a59);
  a71=(a71-a59);
  a71=(a195*a71);
  a71=(a71/a49);
  a203=(a203*a71);
  a72=(a171*a72);
  a49=(a41*a170);
  a49=(a179*a49);
  a61=(a61*a170);
  a68=(a61*a68);
  a68=(a181*a68);
  a68=(a41*a68);
  a68=(a67?a68:0);
  a49=(a49+a68);
  a69=(a61/a69);
  a69=(a181*a69);
  a69=(a41*a69);
  a69=(a35*a69);
  a69=(-a69);
  a67=(a67?a69:0);
  a49=(a49+a67);
  a64=(a61*a64);
  a64=(a41*a64);
  a64=(a64*a213);
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
  a214=(a214*a72);
  a214=(a214/a194);
  a217=(a217*a214);
  a203=(a203-a217);
  a72=(a72/a55);
  a47=(a47*a72);
  a203=(a203-a47);
  a203=(a46*a203);
  a50=(a50*a72);
  a50=(a46*a50);
  a203=(a203+a50);
  a211=(a211*a203);
  a37=(a37-a211);
  a48=(a48*a71);
  a54=(a54*a214);
  a48=(a48-a54);
  a52=(a52*a72);
  a48=(a48+a52);
  a48=(a46*a48);
  a51=(a51*a72);
  a46=(a46*a51);
  a48=(a48-a46);
  a185=(a185*a48);
  a37=(a37+a185);
  a45=(a16*a45);
  a197=(a197*a45);
  a197=(a197*a19);
  a31=(a31*a197);
  a31=(a196*a31);
  a40=(a40?a31:0);
  a31=(a33*a45);
  a17=(a17*a31);
  a17=(a24*a17);
  a199=(a199*a17);
  a22=(a22*a199);
  a196=(a196*a22);
  a40=(a40-a196);
  a195=(a195*a40);
  a195=(a195/a9);
  a205=(a205*a195);
  a171=(a171*a44);
  a44=(a41*a45);
  a179=(a179*a44);
  a26=(a26*a45);
  a36=(a26*a36);
  a36=(a181*a36);
  a36=(a41*a36);
  a36=(a34?a36:0);
  a179=(a179+a36);
  a38=(a26/a38);
  a181=(a181*a38);
  a181=(a41*a181);
  a35=(a35*a181);
  a35=(-a35);
  a34=(a34?a35:0);
  a179=(a179+a34);
  a30=(a26*a30);
  a30=(a41*a30);
  a30=(a30*a223);
  a30=(-a30);
  a30=(a29?a30:0);
  a179=(a179+a30);
  a26=(a26/a32);
  a41=(a41*a26);
  a41=(a41*a27);
  a41=(-a41);
  a29=(a29?a41:0);
  a179=(a179+a29);
  a1=(a1*a179);
  a171=(a171+a1);
  a220=(a220*a171);
  a220=(a220/a222);
  a219=(a219*a220);
  a205=(a205-a219);
  a171=(a171/a15);
  a7=(a7*a171);
  a205=(a205-a7);
  a205=(a5*a205);
  a10=(a10*a171);
  a10=(a5*a10);
  a205=(a205+a10);
  a218=(a218*a205);
  a37=(a37-a218);
  a8=(a8*a195);
  a14=(a14*a220);
  a8=(a8-a14);
  a12=(a12*a171);
  a8=(a8+a12);
  a8=(a5*a8);
  a11=(a11*a171);
  a5=(a5*a11);
  a8=(a8-a5);
  a6=(a6*a8);
  a37=(a37+a6);
  if (res[1]!=0) res[1][9]=a37;
  a37=cos(a74);
  a6=(a165*a186);
  a8=(a2*a186);
  a6=(a6-a8);
  a8=(a163*a6);
  a5=(a160*a186);
  a8=(a8-a5);
  a5=(a183*a161);
  a8=(a8+a5);
  a5=(a162*a178);
  a8=(a8-a5);
  a8=(a37*a8);
  a5=sin(a74);
  a183=(a183*a168);
  a11=(a162*a186);
  a183=(a183-a11);
  a11=(a2*a178);
  a171=(a165*a178);
  a11=(a11-a171);
  a171=(a163*a11);
  a183=(a183+a171);
  a171=(a160*a178);
  a183=(a183+a171);
  a183=(a5*a183);
  a8=(a8-a183);
  a183=sin(a74);
  a171=(a125*a189);
  a12=(a122*a187);
  a171=(a171+a12);
  a12=(a122*a193);
  a171=(a171+a12);
  a12=(a125*a207);
  a171=(a171-a12);
  a171=(a183*a171);
  a8=(a8-a171);
  a171=cos(a74);
  a187=(a125*a187);
  a189=(a122*a189);
  a187=(a187-a189);
  a193=(a125*a193);
  a187=(a187+a193);
  a207=(a122*a207);
  a187=(a187+a207);
  a187=(a171*a187);
  a8=(a8+a187);
  a187=sin(a74);
  a207=(a79*a202);
  a193=(a76*a117);
  a207=(a207+a193);
  a193=(a76*a215);
  a207=(a207+a193);
  a193=(a79*a216);
  a207=(a207-a193);
  a207=(a187*a207);
  a8=(a8-a207);
  a74=cos(a74);
  a117=(a79*a117);
  a202=(a76*a202);
  a117=(a117-a202);
  a215=(a79*a215);
  a117=(a117+a215);
  a216=(a76*a216);
  a117=(a117+a216);
  a117=(a74*a117);
  a8=(a8+a117);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a165*a120);
  a117=(a2*a120);
  a8=(a8-a117);
  a117=(a163*a8);
  a216=(a160*a120);
  a117=(a117-a216);
  a161=(a184*a161);
  a117=(a117+a161);
  a161=(a162*a25);
  a117=(a117-a161);
  a37=(a37*a117);
  a184=(a184*a168);
  a162=(a162*a120);
  a184=(a184-a162);
  a2=(a2*a25);
  a165=(a165*a25);
  a2=(a2-a165);
  a163=(a163*a2);
  a184=(a184+a163);
  a160=(a160*a25);
  a184=(a184+a160);
  a5=(a5*a184);
  a37=(a37-a5);
  a5=(a125*a156);
  a184=(a122*a139);
  a5=(a5+a184);
  a184=(a122*a149);
  a5=(a5+a184);
  a184=(a125*a127);
  a5=(a5-a184);
  a183=(a183*a5);
  a37=(a37-a183);
  a139=(a125*a139);
  a156=(a122*a156);
  a139=(a139-a156);
  a125=(a125*a149);
  a139=(a139+a125);
  a122=(a122*a127);
  a139=(a139+a122);
  a171=(a171*a139);
  a37=(a37+a171);
  a171=(a79*a111);
  a139=(a76*a114);
  a171=(a171+a139);
  a139=(a76*a84);
  a171=(a171+a139);
  a139=(a79*a82);
  a171=(a171-a139);
  a187=(a187*a171);
  a37=(a37-a187);
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
  a74=(a167*a186);
  a114=(a164*a178);
  a74=(a74-a114);
  a6=(a21*a6);
  a11=(a42*a11);
  a6=(a6+a11);
  a6=(a81*a6);
  a6=(a74+a6);
  a11=(a135*a180);
  a6=(a6+a11);
  a11=(a90*a208);
  a6=(a6+a11);
  a70=(a53*a70);
  a6=(a6+a70);
  a221=(a13*a221);
  a6=(a6+a221);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a167*a120);
  a221=(a164*a25);
  a6=(a6-a221);
  a21=(a21*a8);
  a42=(a42*a2);
  a21=(a21+a42);
  a81=(a81*a21);
  a81=(a6+a81);
  a21=(a135*a39);
  a81=(a81+a21);
  a21=(a90*a198);
  a81=(a81+a21);
  a49=(a53*a49);
  a81=(a81+a49);
  a179=(a13*a179);
  a81=(a81+a179);
  if (res[1]!=0) res[1][14]=a81;
  if (res[1]!=0) res[1][15]=a37;
  a186=(a167*a186);
  a74=(a74-a186);
  a178=(a164*a178);
  a74=(a74+a178);
  a180=(a134*a180);
  a74=(a74+a180);
  a208=(a89*a208);
  a74=(a74+a208);
  if (res[1]!=0) res[1][16]=a74;
  a167=(a167*a120);
  a6=(a6-a167);
  a164=(a164*a25);
  a6=(a6+a164);
  a39=(a134*a39);
  a6=(a6+a39);
  a198=(a89*a198);
  a6=(a6+a198);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a23;
  if (res[2]!=0) res[2][1]=a23;
  if (res[2]!=0) res[2][2]=a23;
  if (res[2]!=0) res[2][3]=a23;
  if (res[2]!=0) res[2][4]=a23;
  if (res[2]!=0) res[2][5]=a23;
  if (res[2]!=0) res[2][6]=a23;
  if (res[2]!=0) res[2][7]=a23;
  a23=1.4439765966454325e+00;
  a33=(a33*a24);
  a16=(a16*a33);
  a13=(a13*a16);
  a16=(a4*a13);
  a16=(a23*a16);
  a33=(a3*a16);
  a13=(a172*a13);
  a33=(a33-a13);
  a33=(a173*a33);
  a33=(-a33);
  if (res[3]!=0) res[3][0]=a33;
  if (res[3]!=0) res[3][1]=a16;
  a66=(a66*a60);
  a56=(a56*a66);
  a53=(a53*a56);
  a56=(a4*a53);
  a56=(a23*a56);
  a66=(a3*a56);
  a53=(a172*a53);
  a66=(a66-a53);
  a66=(a173*a66);
  a66=(-a66);
  if (res[3]!=0) res[3][2]=a66;
  if (res[3]!=0) res[3][3]=a56;
  a113=(a113*a102);
  a99=(a99*a113);
  a90=(a90*a99);
  a113=(a4*a90);
  a89=(a89*a99);
  a99=(a169*a89);
  a113=(a113+a99);
  a113=(a23*a113);
  a99=(a3*a113);
  a90=(a172*a90);
  a89=(a173*a89);
  a90=(a90+a89);
  a99=(a99-a90);
  a99=(a173*a99);
  a99=(-a99);
  if (res[3]!=0) res[3][4]=a99;
  if (res[3]!=0) res[3][5]=a113;
  a155=(a155*a146);
  a144=(a144*a155);
  a135=(a135*a144);
  a4=(a4*a135);
  a134=(a134*a144);
  a169=(a169*a134);
  a4=(a4+a169);
  a23=(a23*a4);
  a3=(a3*a23);
  a172=(a172*a135);
  a134=(a173*a134);
  a172=(a172+a134);
  a3=(a3-a172);
  a173=(a173*a3);
  a173=(-a173);
  if (res[3]!=0) res[3][6]=a173;
  if (res[3]!=0) res[3][7]=a23;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    case 4: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201176_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
