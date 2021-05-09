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
  #define CASADI_PREFIX(ID) model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a5=arg[0]? arg[0][4] : 0;
  a6=sin(a5);
  a7=arg[0]? arg[0][0] : 0;
  a8=(a6*a7);
  a9=-5.0000000000000000e-01;
  a10=(a8+a9);
  a11=cos(a5);
  a12=(a11*a7);
  a13=(a10*a12);
  a14=(a11*a7);
  a15=(a6*a7);
  a16=(a14*a15);
  a13=(a13-a16);
  a16=(a8+a9);
  a17=casadi_sq(a16);
  a18=casadi_sq(a14);
  a17=(a17+a18);
  a17=sqrt(a17);
  a13=(a13/a17);
  a18=700.;
  a19=arg[2]? arg[2][0] : 0;
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a20=casadi_sq(a14);
  a9=(a9+a20);
  a9=sqrt(a9);
  a20=4.0000000000000001e-02;
  a21=(a9-a20);
  a22=8.7758256189037276e-01;
  a21=(a21/a22);
  a23=6.9999999999999996e-01;
  a24=(a21/a23);
  a25=1.;
  a24=(a24-a25);
  a26=casadi_sq(a24);
  a27=4.5000000000000001e-01;
  a26=(a26/a27);
  a26=(-a26);
  a26=exp(a26);
  a28=(a19*a26);
  a29=(a13*a1);
  a30=0.;
  a31=(a29<=a30);
  a32=fabs(a29);
  a33=10.;
  a32=(a32/a33);
  a32=(a25-a32);
  a34=fabs(a29);
  a34=(a34/a33);
  a34=(a25+a34);
  a32=(a32/a34);
  a35=(a31?a32:0);
  a36=(!a31);
  a37=1.3300000000000001e+00;
  a38=(a37*a29);
  a38=(a38/a33);
  a39=-8.2500000000000004e-02;
  a38=(a38/a39);
  a38=(a25-a38);
  a40=(a29/a33);
  a40=(a40/a39);
  a40=(a25-a40);
  a38=(a38/a40);
  a41=(a36?a38:0);
  a35=(a35+a41);
  a41=(a28*a35);
  a42=(a20<a21);
  a21=(a21/a23);
  a21=(a21-a25);
  a21=(a33*a21);
  a21=exp(a21);
  a43=(a21-a25);
  a44=1.4741315910257660e+02;
  a43=(a43/a44);
  a43=(a42?a43:0);
  a41=(a41+a43);
  a43=1.0000000000000001e-01;
  a45=7.;
  a46=(a29/a45);
  a46=(a43*a46);
  a41=(a41+a46);
  a41=(a18*a41);
  a46=(a13*a41);
  a47=sin(a5);
  a48=arg[0]? arg[0][1] : 0;
  a49=(a47*a48);
  a50=5.0000000000000000e-01;
  a51=(a49+a50);
  a52=cos(a5);
  a53=(a52*a48);
  a54=(a51*a53);
  a55=(a52*a48);
  a56=(a47*a48);
  a57=(a55*a56);
  a54=(a54-a57);
  a57=(a49+a50);
  a58=casadi_sq(a57);
  a59=casadi_sq(a55);
  a58=(a58+a59);
  a58=sqrt(a58);
  a54=(a54/a58);
  a59=arg[2]? arg[2][1] : 0;
  a49=(a49+a50);
  a50=casadi_sq(a49);
  a60=casadi_sq(a55);
  a50=(a50+a60);
  a50=sqrt(a50);
  a60=(a50-a20);
  a60=(a60/a22);
  a61=(a60/a23);
  a61=(a61-a25);
  a62=casadi_sq(a61);
  a62=(a62/a27);
  a62=(-a62);
  a62=exp(a62);
  a63=(a59*a62);
  a64=(a54*a1);
  a65=(a64<=a30);
  a66=fabs(a64);
  a66=(a66/a33);
  a66=(a25-a66);
  a67=fabs(a64);
  a67=(a67/a33);
  a67=(a25+a67);
  a66=(a66/a67);
  a68=(a65?a66:0);
  a69=(!a65);
  a70=(a37*a64);
  a70=(a70/a33);
  a70=(a70/a39);
  a70=(a25-a70);
  a71=(a64/a33);
  a71=(a71/a39);
  a71=(a25-a71);
  a70=(a70/a71);
  a72=(a69?a70:0);
  a68=(a68+a72);
  a72=(a63*a68);
  a73=(a20<a60);
  a60=(a60/a23);
  a60=(a60-a25);
  a60=(a33*a60);
  a60=exp(a60);
  a74=(a60-a25);
  a74=(a74/a44);
  a74=(a73?a74:0);
  a72=(a72+a74);
  a74=(a64/a45);
  a74=(a43*a74);
  a72=(a72+a74);
  a72=(a18*a72);
  a74=(a54*a72);
  a46=(a46+a74);
  a74=arg[0]? arg[0][5] : 0;
  a75=sin(a74);
  a76=sin(a5);
  a77=(a75*a76);
  a78=cos(a74);
  a79=cos(a5);
  a80=(a78*a79);
  a77=(a77-a80);
  a80=arg[0]? arg[0][2] : 0;
  a81=(a77*a80);
  a82=1.2500000000000000e+00;
  a83=(a82*a76);
  a81=(a81-a83);
  a84=7.5000000000000000e-01;
  a85=(a84*a76);
  a86=(a81+a85);
  a87=(a84*a79);
  a88=(a82*a79);
  a89=(a78*a76);
  a90=(a75*a79);
  a89=(a89+a90);
  a90=(a89*a80);
  a90=(a88-a90);
  a87=(a87-a90);
  a91=(a86*a87);
  a92=(a89*a80);
  a92=(a88-a92);
  a93=(a84*a79);
  a94=(a92-a93);
  a95=(a77*a80);
  a95=(a95-a83);
  a96=(a84*a76);
  a96=(a95+a96);
  a97=(a94*a96);
  a91=(a91+a97);
  a97=(a81+a85);
  a98=casadi_sq(a97);
  a99=(a92-a93);
  a100=casadi_sq(a99);
  a98=(a98+a100);
  a98=sqrt(a98);
  a91=(a91/a98);
  a100=arg[2]? arg[2][2] : 0;
  a81=(a81+a85);
  a85=casadi_sq(a81);
  a92=(a92-a93);
  a93=casadi_sq(a92);
  a85=(a85+a93);
  a85=sqrt(a85);
  a93=(a85-a20);
  a93=(a93/a22);
  a101=(a93/a23);
  a101=(a101-a25);
  a102=casadi_sq(a101);
  a102=(a102/a27);
  a102=(-a102);
  a102=exp(a102);
  a103=(a100*a102);
  a104=(a91*a1);
  a105=(a75*a79);
  a106=(a78*a76);
  a105=(a105+a106);
  a106=(a77*a83);
  a107=(a89*a88);
  a106=(a106+a107);
  a107=(a105*a106);
  a108=(a105*a83);
  a109=(a78*a79);
  a110=(a75*a76);
  a109=(a109-a110);
  a110=(a109*a88);
  a108=(a108+a110);
  a110=(a77*a108);
  a107=(a107-a110);
  a107=(a107-a90);
  a90=(a86*a107);
  a110=(a89*a108);
  a111=(a109*a106);
  a110=(a110-a111);
  a110=(a110+a95);
  a95=(a94*a110);
  a90=(a90+a95);
  a90=(a90/a98);
  a95=(a90*a2);
  a104=(a104+a95);
  a95=(a104<=a30);
  a111=fabs(a104);
  a111=(a111/a33);
  a111=(a25-a111);
  a112=fabs(a104);
  a112=(a112/a33);
  a112=(a25+a112);
  a111=(a111/a112);
  a113=(a95?a111:0);
  a114=(!a95);
  a115=(a37*a104);
  a115=(a115/a33);
  a115=(a115/a39);
  a115=(a25-a115);
  a116=(a104/a33);
  a116=(a116/a39);
  a116=(a25-a116);
  a115=(a115/a116);
  a117=(a114?a115:0);
  a113=(a113+a117);
  a117=(a103*a113);
  a118=(a20<a93);
  a93=(a93/a23);
  a93=(a93-a25);
  a93=(a33*a93);
  a93=exp(a93);
  a119=(a93-a25);
  a119=(a119/a44);
  a119=(a118?a119:0);
  a117=(a117+a119);
  a119=(a104/a45);
  a119=(a43*a119);
  a117=(a117+a119);
  a117=(a18*a117);
  a119=(a91*a117);
  a46=(a46+a119);
  a119=sin(a74);
  a120=sin(a5);
  a121=(a119*a120);
  a122=cos(a74);
  a123=cos(a5);
  a124=(a122*a123);
  a121=(a121-a124);
  a124=arg[0]? arg[0][3] : 0;
  a125=(a121*a124);
  a126=(a82*a120);
  a125=(a125-a126);
  a127=1.7500000000000000e+00;
  a128=(a127*a120);
  a129=(a125+a128);
  a130=(a127*a123);
  a131=(a82*a123);
  a132=(a122*a120);
  a133=(a119*a123);
  a132=(a132+a133);
  a133=(a132*a124);
  a133=(a131-a133);
  a130=(a130-a133);
  a134=(a129*a130);
  a135=(a132*a124);
  a135=(a131-a135);
  a136=(a127*a123);
  a137=(a135-a136);
  a138=(a121*a124);
  a138=(a138-a126);
  a139=(a127*a120);
  a139=(a138+a139);
  a140=(a137*a139);
  a134=(a134+a140);
  a140=(a125+a128);
  a141=casadi_sq(a140);
  a142=(a135-a136);
  a143=casadi_sq(a142);
  a141=(a141+a143);
  a141=sqrt(a141);
  a134=(a134/a141);
  a143=arg[2]? arg[2][3] : 0;
  a125=(a125+a128);
  a128=casadi_sq(a125);
  a135=(a135-a136);
  a136=casadi_sq(a135);
  a128=(a128+a136);
  a128=sqrt(a128);
  a136=(a128-a20);
  a136=(a136/a22);
  a22=(a136/a23);
  a22=(a22-a25);
  a144=casadi_sq(a22);
  a144=(a144/a27);
  a144=(-a144);
  a144=exp(a144);
  a27=(a143*a144);
  a145=(a134*a1);
  a146=(a119*a123);
  a147=(a122*a120);
  a146=(a146+a147);
  a147=(a121*a126);
  a148=(a132*a131);
  a147=(a147+a148);
  a148=(a146*a147);
  a149=(a146*a126);
  a150=(a122*a123);
  a151=(a119*a120);
  a150=(a150-a151);
  a151=(a150*a131);
  a149=(a149+a151);
  a151=(a121*a149);
  a148=(a148-a151);
  a148=(a148-a133);
  a133=(a129*a148);
  a151=(a132*a149);
  a152=(a150*a147);
  a151=(a151-a152);
  a151=(a151+a138);
  a138=(a137*a151);
  a133=(a133+a138);
  a133=(a133/a141);
  a138=(a133*a2);
  a145=(a145+a138);
  a30=(a145<=a30);
  a138=fabs(a145);
  a138=(a138/a33);
  a138=(a25-a138);
  a152=fabs(a145);
  a152=(a152/a33);
  a152=(a25+a152);
  a138=(a138/a152);
  a153=(a30?a138:0);
  a154=(!a30);
  a155=(a37*a145);
  a155=(a155/a33);
  a155=(a155/a39);
  a155=(a25-a155);
  a156=(a145/a33);
  a156=(a156/a39);
  a156=(a25-a156);
  a155=(a155/a156);
  a39=(a154?a155:0);
  a153=(a153+a39);
  a39=(a27*a153);
  a20=(a20<a136);
  a136=(a136/a23);
  a136=(a136-a25);
  a136=(a33*a136);
  a136=exp(a136);
  a23=(a136-a25);
  a23=(a23/a44);
  a23=(a20?a23:0);
  a39=(a39+a23);
  a45=(a145/a45);
  a45=(a43*a45);
  a39=(a39+a45);
  a39=(a18*a39);
  a45=(a134*a39);
  a46=(a46+a45);
  a45=sin(a74);
  a23=cos(a74);
  a44=9.8100000000000005e+00;
  a157=cos(a5);
  a157=(a44*a157);
  a158=(a23*a157);
  a159=sin(a5);
  a159=(a44*a159);
  a160=(a45*a159);
  a158=(a158-a160);
  a160=(a82*a1);
  a161=(a23*a160);
  a162=(a161*a2);
  a158=(a158+a162);
  a162=(a1+a2);
  a163=(a162*a161);
  a158=(a158-a163);
  a163=(a45*a158);
  a164=(a45*a160);
  a165=(a162*a164);
  a166=(a23*a159);
  a167=(a45*a157);
  a166=(a166+a167);
  a167=(a164*a2);
  a166=(a166+a167);
  a165=(a165-a166);
  a166=(a23*a165);
  a163=(a163+a166);
  a163=(a82*a163);
  a46=(a46+a163);
  a4=(a4*a46);
  a163=9.6278838983177639e-01;
  a166=(a90*a117);
  a167=(a133*a39);
  a166=(a166+a167);
  a163=(a163*a166);
  a4=(a4+a163);
  a163=6.9253199970355839e-01;
  a4=(a4/a163);
  a3=(a3*a4);
  a163=9.6278838983177628e-01;
  a163=(a163*a46);
  a46=2.7025639012821789e-01;
  a46=(a46*a166);
  a163=(a163+a46);
  a3=(a3-a163);
  a163=3.7001900289039211e+00;
  a3=(a3/a163);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  a0=(a14+a14);
  a4=1.1394939273245490e+00;
  a3=1.4285714285714286e+00;
  a163=6.7836549063042314e-03;
  a46=3.9024390243902418e-01;
  a166=(a46*a13);
  a166=(a18*a166);
  a167=(a163*a166);
  a167=(a167*a21);
  a167=(a33*a167);
  a167=(a3*a167);
  a167=(a42?a167:0);
  a24=(a24+a24);
  a168=2.2222222222222223e+00;
  a169=(a35*a166);
  a169=(a19*a169);
  a169=(a26*a169);
  a169=(a168*a169);
  a169=(a24*a169);
  a169=(a3*a169);
  a167=(a167-a169);
  a167=(a4*a167);
  a9=(a9+a9);
  a167=(a167/a9);
  a169=(a0*a167);
  a170=(a14+a14);
  a171=(a13/a17);
  a172=(a46*a41);
  a173=1.4285714285714285e-01;
  a174=(a43*a166);
  a174=(a173*a174);
  a175=-1.2121212121212121e+01;
  a166=(a28*a166);
  a38=(a38/a40);
  a176=(a166*a38);
  a176=(a175*a176);
  a176=(a43*a176);
  a176=(a36?a176:0);
  a174=(a174+a176);
  a176=(a166/a40);
  a176=(a175*a176);
  a176=(a43*a176);
  a176=(a37*a176);
  a176=(-a176);
  a176=(a36?a176:0);
  a174=(a174+a176);
  a32=(a32/a34);
  a176=(a166*a32);
  a176=(a43*a176);
  a177=casadi_sign(a29);
  a176=(a176*a177);
  a176=(-a176);
  a176=(a31?a176:0);
  a174=(a174+a176);
  a166=(a166/a34);
  a166=(a43*a166);
  a29=casadi_sign(a29);
  a166=(a166*a29);
  a166=(-a166);
  a166=(a31?a166:0);
  a174=(a174+a166);
  a166=(a1*a174);
  a172=(a172+a166);
  a166=(a171*a172);
  a176=(a17+a17);
  a166=(a166/a176);
  a178=(a170*a166);
  a169=(a169-a178);
  a172=(a172/a17);
  a178=(a15*a172);
  a169=(a169-a178);
  a178=(a11*a169);
  a179=(a14*a172);
  a180=(a6*a179);
  a178=(a178-a180);
  a180=(a10*a172);
  a181=(a11*a180);
  a178=(a178+a181);
  a8=(a8+a8);
  a167=(a8*a167);
  a16=(a16+a16);
  a166=(a16*a166);
  a167=(a167-a166);
  a172=(a12*a172);
  a167=(a167+a172);
  a172=(a6*a167);
  a178=(a178+a172);
  if (res[1]!=0) res[1][0]=a178;
  a178=-3.9024390243902396e-01;
  a172=(a178*a13);
  a172=(a18*a172);
  a166=(a163*a172);
  a166=(a166*a21);
  a166=(a33*a166);
  a166=(a3*a166);
  a42=(a42?a166:0);
  a35=(a35*a172);
  a19=(a19*a35);
  a26=(a26*a19);
  a26=(a168*a26);
  a24=(a24*a26);
  a24=(a3*a24);
  a42=(a42-a24);
  a42=(a4*a42);
  a42=(a42/a9);
  a0=(a0*a42);
  a41=(a178*a41);
  a9=(a43*a172);
  a9=(a173*a9);
  a28=(a28*a172);
  a38=(a28*a38);
  a38=(a175*a38);
  a38=(a43*a38);
  a38=(a36?a38:0);
  a9=(a9+a38);
  a40=(a28/a40);
  a40=(a175*a40);
  a40=(a43*a40);
  a40=(a37*a40);
  a40=(-a40);
  a36=(a36?a40:0);
  a9=(a9+a36);
  a32=(a28*a32);
  a32=(a43*a32);
  a32=(a32*a177);
  a32=(-a32);
  a32=(a31?a32:0);
  a9=(a9+a32);
  a28=(a28/a34);
  a28=(a43*a28);
  a28=(a28*a29);
  a28=(-a28);
  a31=(a31?a28:0);
  a9=(a9+a31);
  a31=(a1*a9);
  a41=(a41+a31);
  a171=(a171*a41);
  a171=(a171/a176);
  a170=(a170*a171);
  a0=(a0-a170);
  a41=(a41/a17);
  a15=(a15*a41);
  a0=(a0-a15);
  a15=(a11*a0);
  a14=(a14*a41);
  a17=(a6*a14);
  a15=(a15-a17);
  a10=(a10*a41);
  a11=(a11*a10);
  a15=(a15+a11);
  a8=(a8*a42);
  a16=(a16*a171);
  a8=(a8-a16);
  a12=(a12*a41);
  a8=(a8+a12);
  a6=(a6*a8);
  a15=(a15+a6);
  if (res[1]!=0) res[1][1]=a15;
  a15=(a55+a55);
  a6=(a46*a54);
  a6=(a18*a6);
  a12=(a163*a6);
  a12=(a12*a60);
  a12=(a33*a12);
  a12=(a3*a12);
  a12=(a73?a12:0);
  a61=(a61+a61);
  a41=(a68*a6);
  a41=(a59*a41);
  a41=(a62*a41);
  a41=(a168*a41);
  a41=(a61*a41);
  a41=(a3*a41);
  a12=(a12-a41);
  a12=(a4*a12);
  a50=(a50+a50);
  a12=(a12/a50);
  a41=(a15*a12);
  a16=(a55+a55);
  a171=(a54/a58);
  a42=(a46*a72);
  a11=(a43*a6);
  a11=(a173*a11);
  a6=(a63*a6);
  a70=(a70/a71);
  a17=(a6*a70);
  a17=(a175*a17);
  a17=(a43*a17);
  a17=(a69?a17:0);
  a11=(a11+a17);
  a17=(a6/a71);
  a17=(a175*a17);
  a17=(a43*a17);
  a17=(a37*a17);
  a17=(-a17);
  a17=(a69?a17:0);
  a11=(a11+a17);
  a66=(a66/a67);
  a17=(a6*a66);
  a17=(a43*a17);
  a170=casadi_sign(a64);
  a17=(a17*a170);
  a17=(-a17);
  a17=(a65?a17:0);
  a11=(a11+a17);
  a6=(a6/a67);
  a6=(a43*a6);
  a64=casadi_sign(a64);
  a6=(a6*a64);
  a6=(-a6);
  a6=(a65?a6:0);
  a11=(a11+a6);
  a6=(a1*a11);
  a42=(a42+a6);
  a6=(a171*a42);
  a17=(a58+a58);
  a6=(a6/a17);
  a176=(a16*a6);
  a41=(a41-a176);
  a42=(a42/a58);
  a176=(a56*a42);
  a41=(a41-a176);
  a176=(a52*a41);
  a31=(a55*a42);
  a28=(a47*a31);
  a176=(a176-a28);
  a28=(a51*a42);
  a29=(a52*a28);
  a176=(a176+a29);
  a49=(a49+a49);
  a12=(a49*a12);
  a57=(a57+a57);
  a6=(a57*a6);
  a12=(a12-a6);
  a42=(a53*a42);
  a12=(a12+a42);
  a42=(a47*a12);
  a176=(a176+a42);
  if (res[1]!=0) res[1][2]=a176;
  a176=(a178*a54);
  a176=(a18*a176);
  a42=(a163*a176);
  a42=(a42*a60);
  a42=(a33*a42);
  a42=(a3*a42);
  a73=(a73?a42:0);
  a68=(a68*a176);
  a59=(a59*a68);
  a62=(a62*a59);
  a62=(a168*a62);
  a61=(a61*a62);
  a61=(a3*a61);
  a73=(a73-a61);
  a73=(a4*a73);
  a73=(a73/a50);
  a15=(a15*a73);
  a72=(a178*a72);
  a50=(a43*a176);
  a50=(a173*a50);
  a63=(a63*a176);
  a70=(a63*a70);
  a70=(a175*a70);
  a70=(a43*a70);
  a70=(a69?a70:0);
  a50=(a50+a70);
  a71=(a63/a71);
  a71=(a175*a71);
  a71=(a43*a71);
  a71=(a37*a71);
  a71=(-a71);
  a69=(a69?a71:0);
  a50=(a50+a69);
  a66=(a63*a66);
  a66=(a43*a66);
  a66=(a66*a170);
  a66=(-a66);
  a66=(a65?a66:0);
  a50=(a50+a66);
  a63=(a63/a67);
  a63=(a43*a63);
  a63=(a63*a64);
  a63=(-a63);
  a65=(a65?a63:0);
  a50=(a50+a65);
  a65=(a1*a50);
  a72=(a72+a65);
  a171=(a171*a72);
  a171=(a171/a17);
  a16=(a16*a171);
  a15=(a15-a16);
  a72=(a72/a58);
  a56=(a56*a72);
  a15=(a15-a56);
  a56=(a52*a15);
  a55=(a55*a72);
  a58=(a47*a55);
  a56=(a56-a58);
  a51=(a51*a72);
  a52=(a52*a51);
  a56=(a56+a52);
  a49=(a49*a73);
  a57=(a57*a171);
  a49=(a49-a57);
  a53=(a53*a72);
  a49=(a49+a53);
  a47=(a47*a49);
  a56=(a56+a47);
  if (res[1]!=0) res[1][3]=a56;
  a56=-3.9024390243902440e-01;
  a47=(a56*a117);
  a53=(a56*a90);
  a72=(a46*a91);
  a53=(a53+a72);
  a53=(a18*a53);
  a72=(a43*a53);
  a72=(a173*a72);
  a57=(a103*a53);
  a115=(a115/a116);
  a171=(a57*a115);
  a171=(a175*a171);
  a171=(a43*a171);
  a171=(a114?a171:0);
  a72=(a72+a171);
  a171=(a57/a116);
  a171=(a175*a171);
  a171=(a43*a171);
  a171=(a37*a171);
  a171=(-a171);
  a171=(a114?a171:0);
  a72=(a72+a171);
  a111=(a111/a112);
  a171=(a57*a111);
  a171=(a43*a171);
  a73=casadi_sign(a104);
  a171=(a171*a73);
  a171=(-a171);
  a171=(a95?a171:0);
  a72=(a72+a171);
  a57=(a57/a112);
  a57=(a43*a57);
  a104=casadi_sign(a104);
  a57=(a57*a104);
  a57=(-a57);
  a57=(a95?a57:0);
  a72=(a72+a57);
  a57=(a2*a72);
  a47=(a47+a57);
  a57=(a47/a98);
  a171=(a94*a57);
  a52=(a46*a117);
  a58=(a1*a72);
  a52=(a52+a58);
  a58=(a52/a98);
  a16=(a94*a58);
  a17=(a171+a16);
  a65=(a77*a17);
  a92=(a92+a92);
  a63=(a163*a53);
  a63=(a63*a93);
  a63=(a33*a63);
  a63=(a3*a63);
  a63=(a118?a63:0);
  a101=(a101+a101);
  a53=(a113*a53);
  a53=(a100*a53);
  a53=(a102*a53);
  a53=(a168*a53);
  a53=(a101*a53);
  a53=(a3*a53);
  a63=(a63-a53);
  a63=(a4*a63);
  a85=(a85+a85);
  a63=(a63/a85);
  a53=(a92*a63);
  a99=(a99+a99);
  a64=(a90/a98);
  a47=(a64*a47);
  a67=(a91/a98);
  a52=(a67*a52);
  a47=(a47+a52);
  a52=(a98+a98);
  a47=(a47/a52);
  a66=(a99*a47);
  a170=(a53-a66);
  a69=(a110*a57);
  a71=(a96*a58);
  a69=(a69+a71);
  a170=(a170+a69);
  a71=(a89*a170);
  a65=(a65-a71);
  a71=(a86*a57);
  a70=(a86*a58);
  a176=(a71+a70);
  a61=(a89*a176);
  a65=(a65+a61);
  a81=(a81+a81);
  a63=(a81*a63);
  a97=(a97+a97);
  a47=(a97*a47);
  a61=(a63-a47);
  a57=(a107*a57);
  a58=(a87*a58);
  a57=(a57+a58);
  a61=(a61+a57);
  a58=(a77*a61);
  a65=(a65+a58);
  if (res[1]!=0) res[1][4]=a65;
  a65=1.3902439024390245e+00;
  a58=(a65*a117);
  a62=(a65*a90);
  a59=(a178*a91);
  a62=(a62+a59);
  a62=(a18*a62);
  a59=(a43*a62);
  a59=(a173*a59);
  a103=(a103*a62);
  a115=(a103*a115);
  a115=(a175*a115);
  a115=(a43*a115);
  a115=(a114?a115:0);
  a59=(a59+a115);
  a116=(a103/a116);
  a116=(a175*a116);
  a116=(a43*a116);
  a116=(a37*a116);
  a116=(-a116);
  a114=(a114?a116:0);
  a59=(a59+a114);
  a111=(a103*a111);
  a111=(a43*a111);
  a111=(a111*a73);
  a111=(-a111);
  a111=(a95?a111:0);
  a59=(a59+a111);
  a103=(a103/a112);
  a103=(a43*a103);
  a103=(a103*a104);
  a103=(-a103);
  a95=(a95?a103:0);
  a59=(a59+a95);
  a95=(a2*a59);
  a58=(a58+a95);
  a95=(a58/a98);
  a103=(a94*a95);
  a117=(a178*a117);
  a104=(a1*a59);
  a117=(a117+a104);
  a98=(a117/a98);
  a94=(a94*a98);
  a104=(a103+a94);
  a112=(a77*a104);
  a111=(a163*a62);
  a111=(a111*a93);
  a111=(a33*a111);
  a111=(a3*a111);
  a118=(a118?a111:0);
  a113=(a113*a62);
  a100=(a100*a113);
  a102=(a102*a100);
  a102=(a168*a102);
  a101=(a101*a102);
  a101=(a3*a101);
  a118=(a118-a101);
  a118=(a4*a118);
  a118=(a118/a85);
  a92=(a92*a118);
  a64=(a64*a58);
  a67=(a67*a117);
  a64=(a64+a67);
  a64=(a64/a52);
  a99=(a99*a64);
  a52=(a92-a99);
  a110=(a110*a95);
  a96=(a96*a98);
  a110=(a110+a96);
  a52=(a52+a110);
  a96=(a89*a52);
  a112=(a112-a96);
  a96=(a86*a95);
  a86=(a86*a98);
  a67=(a96+a86);
  a117=(a89*a67);
  a112=(a112+a117);
  a81=(a81*a118);
  a97=(a97*a64);
  a64=(a81-a97);
  a107=(a107*a95);
  a87=(a87*a98);
  a107=(a107+a87);
  a64=(a64+a107);
  a87=(a77*a64);
  a112=(a112+a87);
  if (res[1]!=0) res[1][5]=a112;
  a112=(a56*a39);
  a56=(a56*a133);
  a87=(a46*a134);
  a56=(a56+a87);
  a56=(a18*a56);
  a87=(a43*a56);
  a87=(a173*a87);
  a98=(a27*a56);
  a155=(a155/a156);
  a95=(a98*a155);
  a95=(a175*a95);
  a95=(a43*a95);
  a95=(a154?a95:0);
  a87=(a87+a95);
  a95=(a98/a156);
  a95=(a175*a95);
  a95=(a43*a95);
  a95=(a37*a95);
  a95=(-a95);
  a95=(a154?a95:0);
  a87=(a87+a95);
  a138=(a138/a152);
  a95=(a98*a138);
  a95=(a43*a95);
  a118=casadi_sign(a145);
  a95=(a95*a118);
  a95=(-a95);
  a95=(a30?a95:0);
  a87=(a87+a95);
  a98=(a98/a152);
  a98=(a43*a98);
  a145=casadi_sign(a145);
  a98=(a98*a145);
  a98=(-a98);
  a98=(a30?a98:0);
  a87=(a87+a98);
  a98=(a2*a87);
  a112=(a112+a98);
  a98=(a112/a141);
  a95=(a137*a98);
  a46=(a46*a39);
  a117=(a1*a87);
  a46=(a46+a117);
  a117=(a46/a141);
  a58=(a137*a117);
  a85=(a95+a58);
  a101=(a121*a85);
  a135=(a135+a135);
  a102=(a163*a56);
  a102=(a102*a136);
  a102=(a33*a102);
  a102=(a3*a102);
  a102=(a20?a102:0);
  a22=(a22+a22);
  a56=(a153*a56);
  a56=(a143*a56);
  a56=(a144*a56);
  a56=(a168*a56);
  a56=(a22*a56);
  a56=(a3*a56);
  a102=(a102-a56);
  a102=(a4*a102);
  a128=(a128+a128);
  a102=(a102/a128);
  a56=(a135*a102);
  a142=(a142+a142);
  a100=(a133/a141);
  a112=(a100*a112);
  a113=(a134/a141);
  a46=(a113*a46);
  a112=(a112+a46);
  a46=(a141+a141);
  a112=(a112/a46);
  a62=(a142*a112);
  a111=(a56-a62);
  a93=(a151*a98);
  a73=(a139*a117);
  a93=(a93+a73);
  a111=(a111+a93);
  a73=(a132*a111);
  a101=(a101-a73);
  a73=(a129*a98);
  a114=(a129*a117);
  a116=(a73+a114);
  a115=(a132*a116);
  a101=(a101+a115);
  a125=(a125+a125);
  a102=(a125*a102);
  a140=(a140+a140);
  a112=(a140*a112);
  a115=(a102-a112);
  a98=(a148*a98);
  a117=(a130*a117);
  a98=(a98+a117);
  a115=(a115+a98);
  a117=(a121*a115);
  a101=(a101+a117);
  if (res[1]!=0) res[1][6]=a101;
  a101=(a65*a39);
  a65=(a65*a133);
  a117=(a178*a134);
  a65=(a65+a117);
  a18=(a18*a65);
  a65=(a43*a18);
  a173=(a173*a65);
  a27=(a27*a18);
  a155=(a27*a155);
  a155=(a175*a155);
  a155=(a43*a155);
  a155=(a154?a155:0);
  a173=(a173+a155);
  a156=(a27/a156);
  a175=(a175*a156);
  a175=(a43*a175);
  a37=(a37*a175);
  a37=(-a37);
  a154=(a154?a37:0);
  a173=(a173+a154);
  a138=(a27*a138);
  a138=(a43*a138);
  a138=(a138*a118);
  a138=(-a138);
  a138=(a30?a138:0);
  a173=(a173+a138);
  a27=(a27/a152);
  a43=(a43*a27);
  a43=(a43*a145);
  a43=(-a43);
  a30=(a30?a43:0);
  a173=(a173+a30);
  a30=(a2*a173);
  a101=(a101+a30);
  a30=(a101/a141);
  a43=(a137*a30);
  a178=(a178*a39);
  a1=(a1*a173);
  a178=(a178+a1);
  a141=(a178/a141);
  a137=(a137*a141);
  a1=(a43+a137);
  a39=(a121*a1);
  a163=(a163*a18);
  a163=(a163*a136);
  a33=(a33*a163);
  a33=(a3*a33);
  a20=(a20?a33:0);
  a153=(a153*a18);
  a143=(a143*a153);
  a144=(a144*a143);
  a168=(a168*a144);
  a22=(a22*a168);
  a3=(a3*a22);
  a20=(a20-a3);
  a4=(a4*a20);
  a4=(a4/a128);
  a135=(a135*a4);
  a100=(a100*a101);
  a113=(a113*a178);
  a100=(a100+a113);
  a100=(a100/a46);
  a142=(a142*a100);
  a46=(a135-a142);
  a151=(a151*a30);
  a139=(a139*a141);
  a151=(a151+a139);
  a46=(a46+a151);
  a139=(a132*a46);
  a39=(a39-a139);
  a139=(a129*a30);
  a129=(a129*a141);
  a113=(a139+a129);
  a178=(a132*a113);
  a39=(a39+a178);
  a125=(a125*a4);
  a140=(a140*a100);
  a100=(a125-a140);
  a148=(a148*a30);
  a130=(a130*a141);
  a148=(a148+a130);
  a100=(a100+a148);
  a130=(a121*a100);
  a39=(a39+a130);
  if (res[1]!=0) res[1][7]=a39;
  a39=cos(a5);
  a130=(a132*a95);
  a141=(a121*a73);
  a130=(a130-a141);
  a141=(a126*a130);
  a30=(a147*a73);
  a141=(a141+a30);
  a30=(a122*a141);
  a4=(a131*a130);
  a178=(a147*a95);
  a4=(a4-a178);
  a178=(a119*a4);
  a30=(a30-a178);
  a58=(a127*a58);
  a30=(a30+a58);
  a58=(a149*a95);
  a178=(a146*a73);
  a95=(a150*a95);
  a178=(a178-a95);
  a95=(a131*a178);
  a58=(a58+a95);
  a95=(a124*a111);
  a58=(a58-a95);
  a95=(a124*a116);
  a58=(a58+a95);
  a95=(a122*a58);
  a30=(a30+a95);
  a102=(a102-a112);
  a102=(a102+a98);
  a102=(a127*a102);
  a30=(a30+a102);
  a102=(a146*a130);
  a98=(a121*a178);
  a102=(a102+a98);
  a102=(a102-a85);
  a102=(a102-a115);
  a102=(a82*a102);
  a30=(a30+a102);
  a102=(a126*a178);
  a73=(a149*a73);
  a102=(a102-a73);
  a85=(a124*a85);
  a102=(a102+a85);
  a115=(a124*a115);
  a102=(a102+a115);
  a115=(a119*a102);
  a30=(a30+a115);
  a30=(a39*a30);
  a115=cos(a5);
  a85=4.8780487804878025e-01;
  a73=(a85*a23);
  a98=(a23*a73);
  a112=(a85*a45);
  a95=(a45*a112);
  a98=(a98+a95);
  a98=(a44*a98);
  a98=(a115*a98);
  a95=sin(a5);
  a101=(a23*a112);
  a128=(a45*a73);
  a101=(a101-a128);
  a101=(a44*a101);
  a101=(a95*a101);
  a98=(a98+a101);
  a101=sin(a5);
  a128=(a122*a4);
  a20=(a119*a141);
  a128=(a128+a20);
  a62=(a62-a56);
  a62=(a62-a93);
  a62=(a127*a62);
  a128=(a128+a62);
  a62=(a119*a58);
  a128=(a128+a62);
  a130=(a150*a130);
  a178=(a132*a178);
  a130=(a130+a178);
  a130=(a130+a111);
  a130=(a130-a116);
  a130=(a82*a130);
  a128=(a128+a130);
  a114=(a127*a114);
  a128=(a128+a114);
  a114=(a122*a102);
  a128=(a128-a114);
  a128=(a101*a128);
  a98=(a98+a128);
  a30=(a30-a98);
  a98=sin(a5);
  a128=(a89*a171);
  a114=(a77*a71);
  a128=(a128-a114);
  a114=(a88*a128);
  a130=(a106*a171);
  a114=(a114-a130);
  a130=(a78*a114);
  a116=(a83*a128);
  a111=(a106*a71);
  a116=(a116+a111);
  a111=(a75*a116);
  a130=(a130+a111);
  a66=(a66-a53);
  a66=(a66-a69);
  a66=(a84*a66);
  a130=(a130+a66);
  a66=(a108*a171);
  a69=(a105*a71);
  a171=(a109*a171);
  a69=(a69-a171);
  a171=(a88*a69);
  a66=(a66+a171);
  a171=(a80*a170);
  a66=(a66-a171);
  a171=(a80*a176);
  a66=(a66+a171);
  a171=(a75*a66);
  a130=(a130+a171);
  a171=(a109*a128);
  a53=(a89*a69);
  a171=(a171+a53);
  a171=(a171+a170);
  a171=(a171-a176);
  a171=(a82*a171);
  a130=(a130+a171);
  a70=(a84*a70);
  a130=(a130+a70);
  a70=(a83*a69);
  a71=(a108*a71);
  a70=(a70-a71);
  a71=(a80*a17);
  a70=(a70+a71);
  a71=(a80*a61);
  a70=(a70+a71);
  a71=(a78*a70);
  a130=(a130-a71);
  a130=(a98*a130);
  a30=(a30-a130);
  a130=cos(a5);
  a71=(a78*a116);
  a171=(a75*a114);
  a71=(a71-a171);
  a16=(a84*a16);
  a71=(a71+a16);
  a16=(a78*a66);
  a71=(a71+a16);
  a63=(a63-a47);
  a63=(a63+a57);
  a63=(a84*a63);
  a71=(a71+a63);
  a128=(a105*a128);
  a69=(a77*a69);
  a128=(a128+a69);
  a128=(a128-a17);
  a128=(a128-a61);
  a128=(a82*a128);
  a71=(a71+a128);
  a128=(a75*a70);
  a71=(a71+a128);
  a71=(a130*a71);
  a30=(a30+a71);
  a71=sin(a5);
  a41=(a48*a41);
  a28=(a48*a28);
  a41=(a41+a28);
  a41=(a71*a41);
  a30=(a30-a41);
  a41=cos(a5);
  a12=(a48*a12);
  a31=(a48*a31);
  a12=(a12-a31);
  a12=(a41*a12);
  a30=(a30+a12);
  a12=sin(a5);
  a169=(a7*a169);
  a180=(a7*a180);
  a169=(a169+a180);
  a169=(a12*a169);
  a30=(a30-a169);
  a5=cos(a5);
  a167=(a7*a167);
  a179=(a7*a179);
  a167=(a167-a179);
  a167=(a5*a167);
  a30=(a30+a167);
  if (res[1]!=0) res[1][8]=a30;
  a30=(a132*a43);
  a167=(a121*a139);
  a30=(a30-a167);
  a167=(a126*a30);
  a179=(a147*a139);
  a167=(a167+a179);
  a179=(a122*a167);
  a169=(a131*a30);
  a147=(a147*a43);
  a169=(a169-a147);
  a147=(a119*a169);
  a179=(a179-a147);
  a137=(a127*a137);
  a179=(a179+a137);
  a137=(a149*a43);
  a147=(a146*a139);
  a43=(a150*a43);
  a147=(a147-a43);
  a131=(a131*a147);
  a137=(a137+a131);
  a131=(a124*a46);
  a137=(a137-a131);
  a131=(a124*a113);
  a137=(a137+a131);
  a131=(a122*a137);
  a179=(a179+a131);
  a125=(a125-a140);
  a125=(a125+a148);
  a125=(a127*a125);
  a179=(a179+a125);
  a146=(a146*a30);
  a121=(a121*a147);
  a146=(a146+a121);
  a146=(a146-a1);
  a146=(a146-a100);
  a146=(a82*a146);
  a179=(a179+a146);
  a126=(a126*a147);
  a149=(a149*a139);
  a126=(a126-a149);
  a1=(a124*a1);
  a126=(a126+a1);
  a124=(a124*a100);
  a126=(a126+a124);
  a124=(a119*a126);
  a179=(a179+a124);
  a39=(a39*a179);
  a179=-4.8780487804877992e-01;
  a124=(a179*a23);
  a100=(a23*a124);
  a1=(a179*a45);
  a149=(a45*a1);
  a100=(a100+a149);
  a100=(a44*a100);
  a115=(a115*a100);
  a100=(a23*a1);
  a149=(a45*a124);
  a100=(a100-a149);
  a44=(a44*a100);
  a95=(a95*a44);
  a115=(a115+a95);
  a95=(a122*a169);
  a44=(a119*a167);
  a95=(a95+a44);
  a142=(a142-a135);
  a142=(a142-a151);
  a142=(a127*a142);
  a95=(a95+a142);
  a119=(a119*a137);
  a95=(a95+a119);
  a150=(a150*a30);
  a132=(a132*a147);
  a150=(a150+a132);
  a150=(a150+a46);
  a150=(a150-a113);
  a150=(a82*a150);
  a95=(a95+a150);
  a127=(a127*a129);
  a95=(a95+a127);
  a122=(a122*a126);
  a95=(a95-a122);
  a101=(a101*a95);
  a115=(a115+a101);
  a39=(a39-a115);
  a115=(a89*a103);
  a101=(a77*a96);
  a115=(a115-a101);
  a101=(a88*a115);
  a95=(a106*a103);
  a101=(a101-a95);
  a95=(a78*a101);
  a122=(a83*a115);
  a106=(a106*a96);
  a122=(a122+a106);
  a106=(a75*a122);
  a95=(a95+a106);
  a99=(a99-a92);
  a99=(a99-a110);
  a99=(a84*a99);
  a95=(a95+a99);
  a99=(a108*a103);
  a110=(a105*a96);
  a103=(a109*a103);
  a110=(a110-a103);
  a88=(a88*a110);
  a99=(a99+a88);
  a88=(a80*a52);
  a99=(a99-a88);
  a88=(a80*a67);
  a99=(a99+a88);
  a88=(a75*a99);
  a95=(a95+a88);
  a109=(a109*a115);
  a89=(a89*a110);
  a109=(a109+a89);
  a109=(a109+a52);
  a109=(a109-a67);
  a109=(a82*a109);
  a95=(a95+a109);
  a86=(a84*a86);
  a95=(a95+a86);
  a83=(a83*a110);
  a108=(a108*a96);
  a83=(a83-a108);
  a108=(a80*a104);
  a83=(a83+a108);
  a80=(a80*a64);
  a83=(a83+a80);
  a80=(a78*a83);
  a95=(a95-a80);
  a98=(a98*a95);
  a39=(a39-a98);
  a98=(a78*a122);
  a95=(a75*a101);
  a98=(a98-a95);
  a94=(a84*a94);
  a98=(a98+a94);
  a78=(a78*a99);
  a98=(a98+a78);
  a81=(a81-a97);
  a81=(a81+a107);
  a84=(a84*a81);
  a98=(a98+a84);
  a105=(a105*a115);
  a77=(a77*a110);
  a105=(a105+a77);
  a105=(a105-a104);
  a105=(a105-a64);
  a105=(a82*a105);
  a98=(a98+a105);
  a75=(a75*a83);
  a98=(a98+a75);
  a130=(a130*a98);
  a39=(a39+a130);
  a15=(a48*a15);
  a51=(a48*a51);
  a15=(a15+a51);
  a71=(a71*a15);
  a39=(a39-a71);
  a49=(a48*a49);
  a48=(a48*a55);
  a49=(a49-a48);
  a41=(a41*a49);
  a39=(a39+a41);
  a0=(a7*a0);
  a10=(a7*a10);
  a0=(a0+a10);
  a12=(a12*a0);
  a39=(a39-a12);
  a8=(a7*a8);
  a7=(a7*a14);
  a8=(a8-a7);
  a5=(a5*a8);
  a39=(a39+a5);
  if (res[1]!=0) res[1][9]=a39;
  a39=cos(a74);
  a5=(a162*a73);
  a8=(a2*a73);
  a5=(a5-a8);
  a8=(a160*a5);
  a7=(a157*a73);
  a8=(a8-a7);
  a7=(a85*a158);
  a8=(a8+a7);
  a7=(a159*a112);
  a8=(a8-a7);
  a8=(a39*a8);
  a7=sin(a74);
  a85=(a85*a165);
  a14=(a159*a73);
  a85=(a85-a14);
  a14=(a2*a112);
  a12=(a162*a112);
  a14=(a14-a12);
  a12=(a160*a14);
  a85=(a85+a12);
  a12=(a157*a112);
  a85=(a85+a12);
  a85=(a7*a85);
  a8=(a8-a85);
  a85=sin(a74);
  a12=(a123*a4);
  a0=(a120*a141);
  a12=(a12+a0);
  a0=(a120*a58);
  a12=(a12+a0);
  a0=(a123*a102);
  a12=(a12-a0);
  a12=(a85*a12);
  a8=(a8-a12);
  a12=cos(a74);
  a141=(a123*a141);
  a4=(a120*a4);
  a141=(a141-a4);
  a58=(a123*a58);
  a141=(a141+a58);
  a102=(a120*a102);
  a141=(a141+a102);
  a141=(a12*a141);
  a8=(a8+a141);
  a141=sin(a74);
  a102=(a79*a114);
  a58=(a76*a116);
  a102=(a102+a58);
  a58=(a76*a66);
  a102=(a102+a58);
  a58=(a79*a70);
  a102=(a102-a58);
  a102=(a141*a102);
  a8=(a8-a102);
  a74=cos(a74);
  a116=(a79*a116);
  a114=(a76*a114);
  a116=(a116-a114);
  a66=(a79*a66);
  a116=(a116+a66);
  a70=(a76*a70);
  a116=(a116+a70);
  a116=(a74*a116);
  a8=(a8+a116);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a162*a124);
  a116=(a2*a124);
  a8=(a8-a116);
  a116=(a160*a8);
  a70=(a157*a124);
  a116=(a116-a70);
  a158=(a179*a158);
  a116=(a116+a158);
  a158=(a159*a1);
  a116=(a116-a158);
  a39=(a39*a116);
  a179=(a179*a165);
  a159=(a159*a124);
  a179=(a179-a159);
  a2=(a2*a1);
  a162=(a162*a1);
  a2=(a2-a162);
  a160=(a160*a2);
  a179=(a179+a160);
  a157=(a157*a1);
  a179=(a179+a157);
  a7=(a7*a179);
  a39=(a39-a7);
  a7=(a123*a169);
  a179=(a120*a167);
  a7=(a7+a179);
  a179=(a120*a137);
  a7=(a7+a179);
  a179=(a123*a126);
  a7=(a7-a179);
  a85=(a85*a7);
  a39=(a39-a85);
  a167=(a123*a167);
  a169=(a120*a169);
  a167=(a167-a169);
  a123=(a123*a137);
  a167=(a167+a123);
  a120=(a120*a126);
  a167=(a167+a120);
  a12=(a12*a167);
  a39=(a39+a12);
  a12=(a79*a101);
  a167=(a76*a122);
  a12=(a12+a167);
  a167=(a76*a99);
  a12=(a12+a167);
  a167=(a79*a83);
  a12=(a12-a167);
  a141=(a141*a12);
  a39=(a39-a141);
  a122=(a79*a122);
  a101=(a76*a101);
  a122=(a122-a101);
  a79=(a79*a99);
  a122=(a122+a79);
  a76=(a76*a83);
  a122=(a122+a76);
  a74=(a74*a122);
  a39=(a39+a74);
  if (res[1]!=0) res[1][11]=a39;
  a39=-1.;
  if (res[1]!=0) res[1][12]=a39;
  a74=(a164*a73);
  a122=(a161*a112);
  a74=(a74-a122);
  a5=(a45*a5);
  a14=(a23*a14);
  a5=(a5+a14);
  a5=(a82*a5);
  a5=(a74+a5);
  a14=(a134*a87);
  a5=(a5+a14);
  a14=(a91*a72);
  a5=(a5+a14);
  a11=(a54*a11);
  a5=(a5+a11);
  a174=(a13*a174);
  a5=(a5+a174);
  if (res[1]!=0) res[1][13]=a5;
  a5=(a164*a124);
  a174=(a161*a1);
  a5=(a5-a174);
  a45=(a45*a8);
  a23=(a23*a2);
  a45=(a45+a23);
  a82=(a82*a45);
  a82=(a5+a82);
  a134=(a134*a173);
  a82=(a82+a134);
  a91=(a91*a59);
  a82=(a82+a91);
  a54=(a54*a50);
  a82=(a82+a54);
  a13=(a13*a9);
  a82=(a82+a13);
  if (res[1]!=0) res[1][14]=a82;
  if (res[1]!=0) res[1][15]=a39;
  a73=(a164*a73);
  a74=(a74-a73);
  a112=(a161*a112);
  a74=(a74+a112);
  a87=(a133*a87);
  a74=(a74+a87);
  a72=(a90*a72);
  a74=(a74+a72);
  if (res[1]!=0) res[1][16]=a74;
  a164=(a164*a124);
  a5=(a5-a164);
  a161=(a161*a1);
  a5=(a5+a161);
  a133=(a133*a173);
  a5=(a5+a133);
  a90=(a90*a59);
  a5=(a5+a90);
  if (res[1]!=0) res[1][17]=a5;
  if (res[2]!=0) res[2][0]=a25;
  if (res[2]!=0) res[2][1]=a25;
  if (res[2]!=0) res[2][2]=a25;
  if (res[2]!=0) res[2][3]=a25;
  if (res[2]!=0) res[2][4]=a25;
  if (res[2]!=0) res[2][5]=a25;
  if (res[2]!=0) res[2][6]=a25;
  if (res[2]!=0) res[2][7]=a25;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15201289_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
