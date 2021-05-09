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
  #define CASADI_PREFIX(ID) model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a18=5.2153797481361005e+02;
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
  a59=6.1071168361795026e+02;
  a60=arg[2]? arg[2][1] : 0;
  a49=(a49+a50);
  a50=casadi_sq(a49);
  a61=casadi_sq(a55);
  a50=(a50+a61);
  a50=sqrt(a50);
  a61=(a50-a20);
  a61=(a61/a22);
  a62=(a61/a23);
  a62=(a62-a25);
  a63=casadi_sq(a62);
  a63=(a63/a27);
  a63=(-a63);
  a63=exp(a63);
  a64=(a60*a63);
  a65=(a54*a1);
  a66=(a65<=a30);
  a67=fabs(a65);
  a67=(a67/a33);
  a67=(a25-a67);
  a68=fabs(a65);
  a68=(a68/a33);
  a68=(a25+a68);
  a67=(a67/a68);
  a69=(a66?a67:0);
  a70=(!a66);
  a71=(a37*a65);
  a71=(a71/a33);
  a71=(a71/a39);
  a71=(a25-a71);
  a72=(a65/a33);
  a72=(a72/a39);
  a72=(a25-a72);
  a71=(a71/a72);
  a73=(a70?a71:0);
  a69=(a69+a73);
  a73=(a64*a69);
  a74=(a20<a61);
  a61=(a61/a23);
  a61=(a61-a25);
  a61=(a33*a61);
  a61=exp(a61);
  a75=(a61-a25);
  a75=(a75/a44);
  a75=(a74?a75:0);
  a73=(a73+a75);
  a75=(a65/a45);
  a75=(a43*a75);
  a73=(a73+a75);
  a73=(a59*a73);
  a75=(a54*a73);
  a46=(a46+a75);
  a75=arg[0]? arg[0][5] : 0;
  a76=sin(a75);
  a77=sin(a5);
  a78=(a76*a77);
  a79=cos(a75);
  a80=cos(a5);
  a81=(a79*a80);
  a78=(a78-a81);
  a81=arg[0]? arg[0][2] : 0;
  a82=(a78*a81);
  a83=1.2500000000000000e+00;
  a84=(a83*a77);
  a82=(a82-a84);
  a85=7.5000000000000000e-01;
  a86=(a85*a77);
  a87=(a82+a86);
  a88=(a85*a80);
  a89=(a83*a80);
  a90=(a79*a77);
  a91=(a76*a80);
  a90=(a90+a91);
  a91=(a90*a81);
  a91=(a89-a91);
  a88=(a88-a91);
  a92=(a87*a88);
  a93=(a90*a81);
  a93=(a89-a93);
  a94=(a85*a80);
  a95=(a93-a94);
  a96=(a78*a81);
  a96=(a96-a84);
  a97=(a85*a77);
  a97=(a96+a97);
  a98=(a95*a97);
  a92=(a92+a98);
  a98=(a82+a86);
  a99=casadi_sq(a98);
  a100=(a93-a94);
  a101=casadi_sq(a100);
  a99=(a99+a101);
  a99=sqrt(a99);
  a92=(a92/a99);
  a101=6.4940891440752682e+02;
  a102=arg[2]? arg[2][2] : 0;
  a82=(a82+a86);
  a86=casadi_sq(a82);
  a93=(a93-a94);
  a94=casadi_sq(a93);
  a86=(a86+a94);
  a86=sqrt(a86);
  a94=(a86-a20);
  a94=(a94/a22);
  a103=(a94/a23);
  a103=(a103-a25);
  a104=casadi_sq(a103);
  a104=(a104/a27);
  a104=(-a104);
  a104=exp(a104);
  a105=(a102*a104);
  a106=(a92*a1);
  a107=(a76*a80);
  a108=(a79*a77);
  a107=(a107+a108);
  a108=(a78*a84);
  a109=(a90*a89);
  a108=(a108+a109);
  a109=(a107*a108);
  a110=(a107*a84);
  a111=(a79*a80);
  a112=(a76*a77);
  a111=(a111-a112);
  a112=(a111*a89);
  a110=(a110+a112);
  a112=(a78*a110);
  a109=(a109-a112);
  a109=(a109-a91);
  a91=(a87*a109);
  a112=(a90*a110);
  a113=(a111*a108);
  a112=(a112-a113);
  a112=(a112+a96);
  a96=(a95*a112);
  a91=(a91+a96);
  a91=(a91/a99);
  a96=(a91*a2);
  a106=(a106+a96);
  a96=(a106<=a30);
  a113=fabs(a106);
  a113=(a113/a33);
  a113=(a25-a113);
  a114=fabs(a106);
  a114=(a114/a33);
  a114=(a25+a114);
  a113=(a113/a114);
  a115=(a96?a113:0);
  a116=(!a96);
  a117=(a37*a106);
  a117=(a117/a33);
  a117=(a117/a39);
  a117=(a25-a117);
  a118=(a106/a33);
  a118=(a118/a39);
  a118=(a25-a118);
  a117=(a117/a118);
  a119=(a116?a117:0);
  a115=(a115+a119);
  a119=(a105*a115);
  a120=(a20<a94);
  a94=(a94/a23);
  a94=(a94-a25);
  a94=(a33*a94);
  a94=exp(a94);
  a121=(a94-a25);
  a121=(a121/a44);
  a121=(a120?a121:0);
  a119=(a119+a121);
  a121=(a106/a45);
  a121=(a43*a121);
  a119=(a119+a121);
  a119=(a101*a119);
  a121=(a92*a119);
  a46=(a46+a121);
  a121=sin(a75);
  a122=sin(a5);
  a123=(a121*a122);
  a124=cos(a75);
  a125=cos(a5);
  a126=(a124*a125);
  a123=(a123-a126);
  a126=arg[0]? arg[0][3] : 0;
  a127=(a123*a126);
  a128=(a83*a122);
  a127=(a127-a128);
  a129=1.7500000000000000e+00;
  a130=(a129*a122);
  a131=(a127+a130);
  a132=(a129*a125);
  a133=(a83*a125);
  a134=(a124*a122);
  a135=(a121*a125);
  a134=(a134+a135);
  a135=(a134*a126);
  a135=(a133-a135);
  a132=(a132-a135);
  a136=(a131*a132);
  a137=(a134*a126);
  a137=(a133-a137);
  a138=(a129*a125);
  a139=(a137-a138);
  a140=(a123*a126);
  a140=(a140-a128);
  a141=(a129*a122);
  a141=(a140+a141);
  a142=(a139*a141);
  a136=(a136+a142);
  a142=(a127+a130);
  a143=casadi_sq(a142);
  a144=(a137-a138);
  a145=casadi_sq(a144);
  a143=(a143+a145);
  a143=sqrt(a143);
  a136=(a136/a143);
  a145=7.4611516518609142e+02;
  a146=arg[2]? arg[2][3] : 0;
  a127=(a127+a130);
  a130=casadi_sq(a127);
  a137=(a137-a138);
  a138=casadi_sq(a137);
  a130=(a130+a138);
  a130=sqrt(a130);
  a138=(a130-a20);
  a138=(a138/a22);
  a22=(a138/a23);
  a22=(a22-a25);
  a147=casadi_sq(a22);
  a147=(a147/a27);
  a147=(-a147);
  a147=exp(a147);
  a27=(a146*a147);
  a148=(a136*a1);
  a149=(a121*a125);
  a150=(a124*a122);
  a149=(a149+a150);
  a150=(a123*a128);
  a151=(a134*a133);
  a150=(a150+a151);
  a151=(a149*a150);
  a152=(a149*a128);
  a153=(a124*a125);
  a154=(a121*a122);
  a153=(a153-a154);
  a154=(a153*a133);
  a152=(a152+a154);
  a154=(a123*a152);
  a151=(a151-a154);
  a151=(a151-a135);
  a135=(a131*a151);
  a154=(a134*a152);
  a155=(a153*a150);
  a154=(a154-a155);
  a154=(a154+a140);
  a140=(a139*a154);
  a135=(a135+a140);
  a135=(a135/a143);
  a140=(a135*a2);
  a148=(a148+a140);
  a30=(a148<=a30);
  a140=fabs(a148);
  a140=(a140/a33);
  a140=(a25-a140);
  a155=fabs(a148);
  a155=(a155/a33);
  a155=(a25+a155);
  a140=(a140/a155);
  a156=(a30?a140:0);
  a157=(!a30);
  a158=(a37*a148);
  a158=(a158/a33);
  a158=(a158/a39);
  a158=(a25-a158);
  a159=(a148/a33);
  a159=(a159/a39);
  a159=(a25-a159);
  a158=(a158/a159);
  a39=(a157?a158:0);
  a156=(a156+a39);
  a39=(a27*a156);
  a20=(a20<a138);
  a138=(a138/a23);
  a138=(a138-a25);
  a138=(a33*a138);
  a138=exp(a138);
  a23=(a138-a25);
  a23=(a23/a44);
  a23=(a20?a23:0);
  a39=(a39+a23);
  a45=(a148/a45);
  a45=(a43*a45);
  a39=(a39+a45);
  a39=(a145*a39);
  a45=(a136*a39);
  a46=(a46+a45);
  a45=sin(a75);
  a23=cos(a75);
  a44=9.8100000000000005e+00;
  a160=cos(a5);
  a160=(a44*a160);
  a161=(a23*a160);
  a162=sin(a5);
  a162=(a44*a162);
  a163=(a45*a162);
  a161=(a161-a163);
  a163=(a83*a1);
  a164=(a23*a163);
  a165=(a164*a2);
  a161=(a161+a165);
  a165=(a1+a2);
  a166=(a165*a164);
  a161=(a161-a166);
  a166=(a45*a161);
  a167=(a45*a163);
  a168=(a165*a167);
  a169=(a23*a162);
  a170=(a45*a160);
  a169=(a169+a170);
  a170=(a167*a2);
  a169=(a169+a170);
  a168=(a168-a169);
  a169=(a23*a168);
  a166=(a166+a169);
  a166=(a83*a166);
  a46=(a46+a166);
  a4=(a4*a46);
  a166=9.6278838983177639e-01;
  a169=(a91*a119);
  a170=(a135*a39);
  a169=(a169+a170);
  a166=(a166*a169);
  a4=(a4+a166);
  a166=6.9253199970355839e-01;
  a4=(a4/a166);
  a3=(a3*a4);
  a166=9.6278838983177628e-01;
  a166=(a166*a46);
  a46=2.7025639012821789e-01;
  a46=(a46*a169);
  a166=(a166+a46);
  a3=(a3-a166);
  a166=3.7001900289039211e+00;
  a3=(a3/a166);
  a0=(a0-a3);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a4);
  if (res[0]!=0) res[0][7]=a0;
  a0=(a14+a14);
  a4=1.1394939273245490e+00;
  a3=1.4285714285714286e+00;
  a166=6.7836549063042314e-03;
  a46=3.9024390243902418e-01;
  a169=(a46*a13);
  a169=(a18*a169);
  a170=(a166*a169);
  a170=(a170*a21);
  a170=(a33*a170);
  a170=(a3*a170);
  a170=(a42?a170:0);
  a24=(a24+a24);
  a171=2.2222222222222223e+00;
  a172=(a35*a169);
  a172=(a19*a172);
  a172=(a26*a172);
  a172=(a171*a172);
  a172=(a24*a172);
  a172=(a3*a172);
  a170=(a170-a172);
  a170=(a4*a170);
  a9=(a9+a9);
  a170=(a170/a9);
  a172=(a0*a170);
  a173=(a14+a14);
  a174=(a13/a17);
  a175=(a46*a41);
  a176=1.4285714285714285e-01;
  a177=(a43*a169);
  a177=(a176*a177);
  a178=-1.2121212121212121e+01;
  a169=(a28*a169);
  a38=(a38/a40);
  a179=(a169*a38);
  a179=(a178*a179);
  a179=(a43*a179);
  a179=(a36?a179:0);
  a177=(a177+a179);
  a179=(a169/a40);
  a179=(a178*a179);
  a179=(a43*a179);
  a179=(a37*a179);
  a179=(-a179);
  a179=(a36?a179:0);
  a177=(a177+a179);
  a32=(a32/a34);
  a179=(a169*a32);
  a179=(a43*a179);
  a180=casadi_sign(a29);
  a179=(a179*a180);
  a179=(-a179);
  a179=(a31?a179:0);
  a177=(a177+a179);
  a169=(a169/a34);
  a169=(a43*a169);
  a29=casadi_sign(a29);
  a169=(a169*a29);
  a169=(-a169);
  a169=(a31?a169:0);
  a177=(a177+a169);
  a169=(a1*a177);
  a175=(a175+a169);
  a169=(a174*a175);
  a179=(a17+a17);
  a169=(a169/a179);
  a181=(a173*a169);
  a172=(a172-a181);
  a175=(a175/a17);
  a181=(a15*a175);
  a172=(a172-a181);
  a181=(a11*a172);
  a182=(a14*a175);
  a183=(a6*a182);
  a181=(a181-a183);
  a183=(a10*a175);
  a184=(a11*a183);
  a181=(a181+a184);
  a8=(a8+a8);
  a170=(a8*a170);
  a16=(a16+a16);
  a169=(a16*a169);
  a170=(a170-a169);
  a175=(a12*a175);
  a170=(a170+a175);
  a175=(a6*a170);
  a181=(a181+a175);
  if (res[1]!=0) res[1][0]=a181;
  a181=-3.9024390243902396e-01;
  a175=(a181*a13);
  a18=(a18*a175);
  a175=(a166*a18);
  a175=(a175*a21);
  a175=(a33*a175);
  a175=(a3*a175);
  a42=(a42?a175:0);
  a35=(a35*a18);
  a19=(a19*a35);
  a26=(a26*a19);
  a26=(a171*a26);
  a24=(a24*a26);
  a24=(a3*a24);
  a42=(a42-a24);
  a42=(a4*a42);
  a42=(a42/a9);
  a0=(a0*a42);
  a41=(a181*a41);
  a9=(a43*a18);
  a9=(a176*a9);
  a28=(a28*a18);
  a38=(a28*a38);
  a38=(a178*a38);
  a38=(a43*a38);
  a38=(a36?a38:0);
  a9=(a9+a38);
  a40=(a28/a40);
  a40=(a178*a40);
  a40=(a43*a40);
  a40=(a37*a40);
  a40=(-a40);
  a36=(a36?a40:0);
  a9=(a9+a36);
  a32=(a28*a32);
  a32=(a43*a32);
  a32=(a32*a180);
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
  a174=(a174*a41);
  a174=(a174/a179);
  a173=(a173*a174);
  a0=(a0-a173);
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
  a16=(a16*a174);
  a8=(a8-a16);
  a12=(a12*a41);
  a8=(a8+a12);
  a6=(a6*a8);
  a15=(a15+a6);
  if (res[1]!=0) res[1][1]=a15;
  a15=(a55+a55);
  a6=(a46*a54);
  a6=(a59*a6);
  a12=(a166*a6);
  a12=(a12*a61);
  a12=(a33*a12);
  a12=(a3*a12);
  a12=(a74?a12:0);
  a62=(a62+a62);
  a41=(a69*a6);
  a41=(a60*a41);
  a41=(a63*a41);
  a41=(a171*a41);
  a41=(a62*a41);
  a41=(a3*a41);
  a12=(a12-a41);
  a12=(a4*a12);
  a50=(a50+a50);
  a12=(a12/a50);
  a41=(a15*a12);
  a16=(a55+a55);
  a174=(a54/a58);
  a42=(a46*a73);
  a11=(a43*a6);
  a11=(a176*a11);
  a6=(a64*a6);
  a71=(a71/a72);
  a17=(a6*a71);
  a17=(a178*a17);
  a17=(a43*a17);
  a17=(a70?a17:0);
  a11=(a11+a17);
  a17=(a6/a72);
  a17=(a178*a17);
  a17=(a43*a17);
  a17=(a37*a17);
  a17=(-a17);
  a17=(a70?a17:0);
  a11=(a11+a17);
  a67=(a67/a68);
  a17=(a6*a67);
  a17=(a43*a17);
  a173=casadi_sign(a65);
  a17=(a17*a173);
  a17=(-a17);
  a17=(a66?a17:0);
  a11=(a11+a17);
  a6=(a6/a68);
  a6=(a43*a6);
  a65=casadi_sign(a65);
  a6=(a6*a65);
  a6=(-a6);
  a6=(a66?a6:0);
  a11=(a11+a6);
  a6=(a1*a11);
  a42=(a42+a6);
  a6=(a174*a42);
  a17=(a58+a58);
  a6=(a6/a17);
  a179=(a16*a6);
  a41=(a41-a179);
  a42=(a42/a58);
  a179=(a56*a42);
  a41=(a41-a179);
  a179=(a52*a41);
  a31=(a55*a42);
  a28=(a47*a31);
  a179=(a179-a28);
  a28=(a51*a42);
  a29=(a52*a28);
  a179=(a179+a29);
  a49=(a49+a49);
  a12=(a49*a12);
  a57=(a57+a57);
  a6=(a57*a6);
  a12=(a12-a6);
  a42=(a53*a42);
  a12=(a12+a42);
  a42=(a47*a12);
  a179=(a179+a42);
  if (res[1]!=0) res[1][2]=a179;
  a179=(a181*a54);
  a59=(a59*a179);
  a179=(a166*a59);
  a179=(a179*a61);
  a179=(a33*a179);
  a179=(a3*a179);
  a74=(a74?a179:0);
  a69=(a69*a59);
  a60=(a60*a69);
  a63=(a63*a60);
  a63=(a171*a63);
  a62=(a62*a63);
  a62=(a3*a62);
  a74=(a74-a62);
  a74=(a4*a74);
  a74=(a74/a50);
  a15=(a15*a74);
  a73=(a181*a73);
  a50=(a43*a59);
  a50=(a176*a50);
  a64=(a64*a59);
  a71=(a64*a71);
  a71=(a178*a71);
  a71=(a43*a71);
  a71=(a70?a71:0);
  a50=(a50+a71);
  a72=(a64/a72);
  a72=(a178*a72);
  a72=(a43*a72);
  a72=(a37*a72);
  a72=(-a72);
  a70=(a70?a72:0);
  a50=(a50+a70);
  a67=(a64*a67);
  a67=(a43*a67);
  a67=(a67*a173);
  a67=(-a67);
  a67=(a66?a67:0);
  a50=(a50+a67);
  a64=(a64/a68);
  a64=(a43*a64);
  a64=(a64*a65);
  a64=(-a64);
  a66=(a66?a64:0);
  a50=(a50+a66);
  a66=(a1*a50);
  a73=(a73+a66);
  a174=(a174*a73);
  a174=(a174/a17);
  a16=(a16*a174);
  a15=(a15-a16);
  a73=(a73/a58);
  a56=(a56*a73);
  a15=(a15-a56);
  a56=(a52*a15);
  a55=(a55*a73);
  a58=(a47*a55);
  a56=(a56-a58);
  a51=(a51*a73);
  a52=(a52*a51);
  a56=(a56+a52);
  a49=(a49*a74);
  a57=(a57*a174);
  a49=(a49-a57);
  a53=(a53*a73);
  a49=(a49+a53);
  a47=(a47*a49);
  a56=(a56+a47);
  if (res[1]!=0) res[1][3]=a56;
  a56=-3.9024390243902440e-01;
  a47=(a56*a119);
  a53=(a56*a91);
  a73=(a46*a92);
  a53=(a53+a73);
  a53=(a101*a53);
  a73=(a43*a53);
  a73=(a176*a73);
  a57=(a105*a53);
  a117=(a117/a118);
  a174=(a57*a117);
  a174=(a178*a174);
  a174=(a43*a174);
  a174=(a116?a174:0);
  a73=(a73+a174);
  a174=(a57/a118);
  a174=(a178*a174);
  a174=(a43*a174);
  a174=(a37*a174);
  a174=(-a174);
  a174=(a116?a174:0);
  a73=(a73+a174);
  a113=(a113/a114);
  a174=(a57*a113);
  a174=(a43*a174);
  a74=casadi_sign(a106);
  a174=(a174*a74);
  a174=(-a174);
  a174=(a96?a174:0);
  a73=(a73+a174);
  a57=(a57/a114);
  a57=(a43*a57);
  a106=casadi_sign(a106);
  a57=(a57*a106);
  a57=(-a57);
  a57=(a96?a57:0);
  a73=(a73+a57);
  a57=(a2*a73);
  a47=(a47+a57);
  a57=(a47/a99);
  a174=(a95*a57);
  a52=(a46*a119);
  a58=(a1*a73);
  a52=(a52+a58);
  a58=(a52/a99);
  a16=(a95*a58);
  a17=(a174+a16);
  a66=(a78*a17);
  a93=(a93+a93);
  a64=(a166*a53);
  a64=(a64*a94);
  a64=(a33*a64);
  a64=(a3*a64);
  a64=(a120?a64:0);
  a103=(a103+a103);
  a53=(a115*a53);
  a53=(a102*a53);
  a53=(a104*a53);
  a53=(a171*a53);
  a53=(a103*a53);
  a53=(a3*a53);
  a64=(a64-a53);
  a64=(a4*a64);
  a86=(a86+a86);
  a64=(a64/a86);
  a53=(a93*a64);
  a100=(a100+a100);
  a65=(a91/a99);
  a47=(a65*a47);
  a68=(a92/a99);
  a52=(a68*a52);
  a47=(a47+a52);
  a52=(a99+a99);
  a47=(a47/a52);
  a67=(a100*a47);
  a173=(a53-a67);
  a70=(a112*a57);
  a72=(a97*a58);
  a70=(a70+a72);
  a173=(a173+a70);
  a72=(a90*a173);
  a66=(a66-a72);
  a72=(a87*a57);
  a71=(a87*a58);
  a59=(a72+a71);
  a62=(a90*a59);
  a66=(a66+a62);
  a82=(a82+a82);
  a64=(a82*a64);
  a98=(a98+a98);
  a47=(a98*a47);
  a62=(a64-a47);
  a57=(a109*a57);
  a58=(a88*a58);
  a57=(a57+a58);
  a62=(a62+a57);
  a58=(a78*a62);
  a66=(a66+a58);
  if (res[1]!=0) res[1][4]=a66;
  a66=1.3902439024390245e+00;
  a58=(a66*a119);
  a63=(a66*a91);
  a60=(a181*a92);
  a63=(a63+a60);
  a101=(a101*a63);
  a63=(a43*a101);
  a63=(a176*a63);
  a105=(a105*a101);
  a117=(a105*a117);
  a117=(a178*a117);
  a117=(a43*a117);
  a117=(a116?a117:0);
  a63=(a63+a117);
  a118=(a105/a118);
  a118=(a178*a118);
  a118=(a43*a118);
  a118=(a37*a118);
  a118=(-a118);
  a116=(a116?a118:0);
  a63=(a63+a116);
  a113=(a105*a113);
  a113=(a43*a113);
  a113=(a113*a74);
  a113=(-a113);
  a113=(a96?a113:0);
  a63=(a63+a113);
  a105=(a105/a114);
  a105=(a43*a105);
  a105=(a105*a106);
  a105=(-a105);
  a96=(a96?a105:0);
  a63=(a63+a96);
  a96=(a2*a63);
  a58=(a58+a96);
  a96=(a58/a99);
  a105=(a95*a96);
  a119=(a181*a119);
  a106=(a1*a63);
  a119=(a119+a106);
  a99=(a119/a99);
  a95=(a95*a99);
  a106=(a105+a95);
  a114=(a78*a106);
  a113=(a166*a101);
  a113=(a113*a94);
  a113=(a33*a113);
  a113=(a3*a113);
  a120=(a120?a113:0);
  a115=(a115*a101);
  a102=(a102*a115);
  a104=(a104*a102);
  a104=(a171*a104);
  a103=(a103*a104);
  a103=(a3*a103);
  a120=(a120-a103);
  a120=(a4*a120);
  a120=(a120/a86);
  a93=(a93*a120);
  a65=(a65*a58);
  a68=(a68*a119);
  a65=(a65+a68);
  a65=(a65/a52);
  a100=(a100*a65);
  a52=(a93-a100);
  a112=(a112*a96);
  a97=(a97*a99);
  a112=(a112+a97);
  a52=(a52+a112);
  a97=(a90*a52);
  a114=(a114-a97);
  a97=(a87*a96);
  a87=(a87*a99);
  a68=(a97+a87);
  a119=(a90*a68);
  a114=(a114+a119);
  a82=(a82*a120);
  a98=(a98*a65);
  a65=(a82-a98);
  a109=(a109*a96);
  a88=(a88*a99);
  a109=(a109+a88);
  a65=(a65+a109);
  a88=(a78*a65);
  a114=(a114+a88);
  if (res[1]!=0) res[1][5]=a114;
  a114=(a56*a39);
  a56=(a56*a135);
  a88=(a46*a136);
  a56=(a56+a88);
  a56=(a145*a56);
  a88=(a43*a56);
  a88=(a176*a88);
  a99=(a27*a56);
  a158=(a158/a159);
  a96=(a99*a158);
  a96=(a178*a96);
  a96=(a43*a96);
  a96=(a157?a96:0);
  a88=(a88+a96);
  a96=(a99/a159);
  a96=(a178*a96);
  a96=(a43*a96);
  a96=(a37*a96);
  a96=(-a96);
  a96=(a157?a96:0);
  a88=(a88+a96);
  a140=(a140/a155);
  a96=(a99*a140);
  a96=(a43*a96);
  a120=casadi_sign(a148);
  a96=(a96*a120);
  a96=(-a96);
  a96=(a30?a96:0);
  a88=(a88+a96);
  a99=(a99/a155);
  a99=(a43*a99);
  a148=casadi_sign(a148);
  a99=(a99*a148);
  a99=(-a99);
  a99=(a30?a99:0);
  a88=(a88+a99);
  a99=(a2*a88);
  a114=(a114+a99);
  a99=(a114/a143);
  a96=(a139*a99);
  a46=(a46*a39);
  a119=(a1*a88);
  a46=(a46+a119);
  a119=(a46/a143);
  a58=(a139*a119);
  a86=(a96+a58);
  a103=(a123*a86);
  a137=(a137+a137);
  a104=(a166*a56);
  a104=(a104*a138);
  a104=(a33*a104);
  a104=(a3*a104);
  a104=(a20?a104:0);
  a22=(a22+a22);
  a56=(a156*a56);
  a56=(a146*a56);
  a56=(a147*a56);
  a56=(a171*a56);
  a56=(a22*a56);
  a56=(a3*a56);
  a104=(a104-a56);
  a104=(a4*a104);
  a130=(a130+a130);
  a104=(a104/a130);
  a56=(a137*a104);
  a144=(a144+a144);
  a102=(a135/a143);
  a114=(a102*a114);
  a115=(a136/a143);
  a46=(a115*a46);
  a114=(a114+a46);
  a46=(a143+a143);
  a114=(a114/a46);
  a101=(a144*a114);
  a113=(a56-a101);
  a94=(a154*a99);
  a74=(a141*a119);
  a94=(a94+a74);
  a113=(a113+a94);
  a74=(a134*a113);
  a103=(a103-a74);
  a74=(a131*a99);
  a116=(a131*a119);
  a118=(a74+a116);
  a117=(a134*a118);
  a103=(a103+a117);
  a127=(a127+a127);
  a104=(a127*a104);
  a142=(a142+a142);
  a114=(a142*a114);
  a117=(a104-a114);
  a99=(a151*a99);
  a119=(a132*a119);
  a99=(a99+a119);
  a117=(a117+a99);
  a119=(a123*a117);
  a103=(a103+a119);
  if (res[1]!=0) res[1][6]=a103;
  a103=(a66*a39);
  a66=(a66*a135);
  a119=(a181*a136);
  a66=(a66+a119);
  a145=(a145*a66);
  a66=(a43*a145);
  a176=(a176*a66);
  a27=(a27*a145);
  a158=(a27*a158);
  a158=(a178*a158);
  a158=(a43*a158);
  a158=(a157?a158:0);
  a176=(a176+a158);
  a159=(a27/a159);
  a178=(a178*a159);
  a178=(a43*a178);
  a37=(a37*a178);
  a37=(-a37);
  a157=(a157?a37:0);
  a176=(a176+a157);
  a140=(a27*a140);
  a140=(a43*a140);
  a140=(a140*a120);
  a140=(-a140);
  a140=(a30?a140:0);
  a176=(a176+a140);
  a27=(a27/a155);
  a43=(a43*a27);
  a43=(a43*a148);
  a43=(-a43);
  a30=(a30?a43:0);
  a176=(a176+a30);
  a30=(a2*a176);
  a103=(a103+a30);
  a30=(a103/a143);
  a43=(a139*a30);
  a181=(a181*a39);
  a1=(a1*a176);
  a181=(a181+a1);
  a143=(a181/a143);
  a139=(a139*a143);
  a1=(a43+a139);
  a39=(a123*a1);
  a166=(a166*a145);
  a166=(a166*a138);
  a33=(a33*a166);
  a33=(a3*a33);
  a20=(a20?a33:0);
  a156=(a156*a145);
  a146=(a146*a156);
  a147=(a147*a146);
  a171=(a171*a147);
  a22=(a22*a171);
  a3=(a3*a22);
  a20=(a20-a3);
  a4=(a4*a20);
  a4=(a4/a130);
  a137=(a137*a4);
  a102=(a102*a103);
  a115=(a115*a181);
  a102=(a102+a115);
  a102=(a102/a46);
  a144=(a144*a102);
  a46=(a137-a144);
  a154=(a154*a30);
  a141=(a141*a143);
  a154=(a154+a141);
  a46=(a46+a154);
  a141=(a134*a46);
  a39=(a39-a141);
  a141=(a131*a30);
  a131=(a131*a143);
  a115=(a141+a131);
  a181=(a134*a115);
  a39=(a39+a181);
  a127=(a127*a4);
  a142=(a142*a102);
  a102=(a127-a142);
  a151=(a151*a30);
  a132=(a132*a143);
  a151=(a151+a132);
  a102=(a102+a151);
  a132=(a123*a102);
  a39=(a39+a132);
  if (res[1]!=0) res[1][7]=a39;
  a39=cos(a5);
  a132=(a134*a96);
  a143=(a123*a74);
  a132=(a132-a143);
  a143=(a128*a132);
  a30=(a150*a74);
  a143=(a143+a30);
  a30=(a124*a143);
  a4=(a133*a132);
  a181=(a150*a96);
  a4=(a4-a181);
  a181=(a121*a4);
  a30=(a30-a181);
  a58=(a129*a58);
  a30=(a30+a58);
  a58=(a152*a96);
  a181=(a149*a74);
  a96=(a153*a96);
  a181=(a181-a96);
  a96=(a133*a181);
  a58=(a58+a96);
  a96=(a126*a113);
  a58=(a58-a96);
  a96=(a126*a118);
  a58=(a58+a96);
  a96=(a124*a58);
  a30=(a30+a96);
  a104=(a104-a114);
  a104=(a104+a99);
  a104=(a129*a104);
  a30=(a30+a104);
  a104=(a149*a132);
  a99=(a123*a181);
  a104=(a104+a99);
  a104=(a104-a86);
  a104=(a104-a117);
  a104=(a83*a104);
  a30=(a30+a104);
  a104=(a128*a181);
  a74=(a152*a74);
  a104=(a104-a74);
  a86=(a126*a86);
  a104=(a104+a86);
  a117=(a126*a117);
  a104=(a104+a117);
  a117=(a121*a104);
  a30=(a30+a117);
  a30=(a39*a30);
  a117=cos(a5);
  a86=4.8780487804878025e-01;
  a74=(a86*a23);
  a99=(a23*a74);
  a114=(a86*a45);
  a96=(a45*a114);
  a99=(a99+a96);
  a99=(a44*a99);
  a99=(a117*a99);
  a96=sin(a5);
  a103=(a23*a114);
  a130=(a45*a74);
  a103=(a103-a130);
  a103=(a44*a103);
  a103=(a96*a103);
  a99=(a99+a103);
  a103=sin(a5);
  a130=(a124*a4);
  a20=(a121*a143);
  a130=(a130+a20);
  a101=(a101-a56);
  a101=(a101-a94);
  a101=(a129*a101);
  a130=(a130+a101);
  a101=(a121*a58);
  a130=(a130+a101);
  a132=(a153*a132);
  a181=(a134*a181);
  a132=(a132+a181);
  a132=(a132+a113);
  a132=(a132-a118);
  a132=(a83*a132);
  a130=(a130+a132);
  a116=(a129*a116);
  a130=(a130+a116);
  a116=(a124*a104);
  a130=(a130-a116);
  a130=(a103*a130);
  a99=(a99+a130);
  a30=(a30-a99);
  a99=sin(a5);
  a130=(a90*a174);
  a116=(a78*a72);
  a130=(a130-a116);
  a116=(a89*a130);
  a132=(a108*a174);
  a116=(a116-a132);
  a132=(a79*a116);
  a118=(a84*a130);
  a113=(a108*a72);
  a118=(a118+a113);
  a113=(a76*a118);
  a132=(a132+a113);
  a67=(a67-a53);
  a67=(a67-a70);
  a67=(a85*a67);
  a132=(a132+a67);
  a67=(a110*a174);
  a70=(a107*a72);
  a174=(a111*a174);
  a70=(a70-a174);
  a174=(a89*a70);
  a67=(a67+a174);
  a174=(a81*a173);
  a67=(a67-a174);
  a174=(a81*a59);
  a67=(a67+a174);
  a174=(a76*a67);
  a132=(a132+a174);
  a174=(a111*a130);
  a53=(a90*a70);
  a174=(a174+a53);
  a174=(a174+a173);
  a174=(a174-a59);
  a174=(a83*a174);
  a132=(a132+a174);
  a71=(a85*a71);
  a132=(a132+a71);
  a71=(a84*a70);
  a72=(a110*a72);
  a71=(a71-a72);
  a72=(a81*a17);
  a71=(a71+a72);
  a72=(a81*a62);
  a71=(a71+a72);
  a72=(a79*a71);
  a132=(a132-a72);
  a132=(a99*a132);
  a30=(a30-a132);
  a132=cos(a5);
  a72=(a79*a118);
  a174=(a76*a116);
  a72=(a72-a174);
  a16=(a85*a16);
  a72=(a72+a16);
  a16=(a79*a67);
  a72=(a72+a16);
  a64=(a64-a47);
  a64=(a64+a57);
  a64=(a85*a64);
  a72=(a72+a64);
  a130=(a107*a130);
  a70=(a78*a70);
  a130=(a130+a70);
  a130=(a130-a17);
  a130=(a130-a62);
  a130=(a83*a130);
  a72=(a72+a130);
  a130=(a76*a71);
  a72=(a72+a130);
  a72=(a132*a72);
  a30=(a30+a72);
  a72=sin(a5);
  a41=(a48*a41);
  a28=(a48*a28);
  a41=(a41+a28);
  a41=(a72*a41);
  a30=(a30-a41);
  a41=cos(a5);
  a12=(a48*a12);
  a31=(a48*a31);
  a12=(a12-a31);
  a12=(a41*a12);
  a30=(a30+a12);
  a12=sin(a5);
  a172=(a7*a172);
  a183=(a7*a183);
  a172=(a172+a183);
  a172=(a12*a172);
  a30=(a30-a172);
  a5=cos(a5);
  a170=(a7*a170);
  a182=(a7*a182);
  a170=(a170-a182);
  a170=(a5*a170);
  a30=(a30+a170);
  if (res[1]!=0) res[1][8]=a30;
  a30=(a134*a43);
  a170=(a123*a141);
  a30=(a30-a170);
  a170=(a128*a30);
  a182=(a150*a141);
  a170=(a170+a182);
  a182=(a124*a170);
  a172=(a133*a30);
  a150=(a150*a43);
  a172=(a172-a150);
  a150=(a121*a172);
  a182=(a182-a150);
  a139=(a129*a139);
  a182=(a182+a139);
  a139=(a152*a43);
  a150=(a149*a141);
  a43=(a153*a43);
  a150=(a150-a43);
  a133=(a133*a150);
  a139=(a139+a133);
  a133=(a126*a46);
  a139=(a139-a133);
  a133=(a126*a115);
  a139=(a139+a133);
  a133=(a124*a139);
  a182=(a182+a133);
  a127=(a127-a142);
  a127=(a127+a151);
  a127=(a129*a127);
  a182=(a182+a127);
  a149=(a149*a30);
  a123=(a123*a150);
  a149=(a149+a123);
  a149=(a149-a1);
  a149=(a149-a102);
  a149=(a83*a149);
  a182=(a182+a149);
  a128=(a128*a150);
  a152=(a152*a141);
  a128=(a128-a152);
  a1=(a126*a1);
  a128=(a128+a1);
  a126=(a126*a102);
  a128=(a128+a126);
  a126=(a121*a128);
  a182=(a182+a126);
  a39=(a39*a182);
  a182=-4.8780487804877992e-01;
  a126=(a182*a23);
  a102=(a23*a126);
  a1=(a182*a45);
  a152=(a45*a1);
  a102=(a102+a152);
  a102=(a44*a102);
  a117=(a117*a102);
  a102=(a23*a1);
  a152=(a45*a126);
  a102=(a102-a152);
  a44=(a44*a102);
  a96=(a96*a44);
  a117=(a117+a96);
  a96=(a124*a172);
  a44=(a121*a170);
  a96=(a96+a44);
  a144=(a144-a137);
  a144=(a144-a154);
  a144=(a129*a144);
  a96=(a96+a144);
  a121=(a121*a139);
  a96=(a96+a121);
  a153=(a153*a30);
  a134=(a134*a150);
  a153=(a153+a134);
  a153=(a153+a46);
  a153=(a153-a115);
  a153=(a83*a153);
  a96=(a96+a153);
  a129=(a129*a131);
  a96=(a96+a129);
  a124=(a124*a128);
  a96=(a96-a124);
  a103=(a103*a96);
  a117=(a117+a103);
  a39=(a39-a117);
  a117=(a90*a105);
  a103=(a78*a97);
  a117=(a117-a103);
  a103=(a89*a117);
  a96=(a108*a105);
  a103=(a103-a96);
  a96=(a79*a103);
  a124=(a84*a117);
  a108=(a108*a97);
  a124=(a124+a108);
  a108=(a76*a124);
  a96=(a96+a108);
  a100=(a100-a93);
  a100=(a100-a112);
  a100=(a85*a100);
  a96=(a96+a100);
  a100=(a110*a105);
  a112=(a107*a97);
  a105=(a111*a105);
  a112=(a112-a105);
  a89=(a89*a112);
  a100=(a100+a89);
  a89=(a81*a52);
  a100=(a100-a89);
  a89=(a81*a68);
  a100=(a100+a89);
  a89=(a76*a100);
  a96=(a96+a89);
  a111=(a111*a117);
  a90=(a90*a112);
  a111=(a111+a90);
  a111=(a111+a52);
  a111=(a111-a68);
  a111=(a83*a111);
  a96=(a96+a111);
  a87=(a85*a87);
  a96=(a96+a87);
  a84=(a84*a112);
  a110=(a110*a97);
  a84=(a84-a110);
  a110=(a81*a106);
  a84=(a84+a110);
  a81=(a81*a65);
  a84=(a84+a81);
  a81=(a79*a84);
  a96=(a96-a81);
  a99=(a99*a96);
  a39=(a39-a99);
  a99=(a79*a124);
  a96=(a76*a103);
  a99=(a99-a96);
  a95=(a85*a95);
  a99=(a99+a95);
  a79=(a79*a100);
  a99=(a99+a79);
  a82=(a82-a98);
  a82=(a82+a109);
  a85=(a85*a82);
  a99=(a99+a85);
  a107=(a107*a117);
  a78=(a78*a112);
  a107=(a107+a78);
  a107=(a107-a106);
  a107=(a107-a65);
  a107=(a83*a107);
  a99=(a99+a107);
  a76=(a76*a84);
  a99=(a99+a76);
  a132=(a132*a99);
  a39=(a39+a132);
  a15=(a48*a15);
  a51=(a48*a51);
  a15=(a15+a51);
  a72=(a72*a15);
  a39=(a39-a72);
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
  a39=cos(a75);
  a5=(a165*a74);
  a8=(a2*a74);
  a5=(a5-a8);
  a8=(a163*a5);
  a7=(a160*a74);
  a8=(a8-a7);
  a7=(a86*a161);
  a8=(a8+a7);
  a7=(a162*a114);
  a8=(a8-a7);
  a8=(a39*a8);
  a7=sin(a75);
  a86=(a86*a168);
  a14=(a162*a74);
  a86=(a86-a14);
  a14=(a2*a114);
  a12=(a165*a114);
  a14=(a14-a12);
  a12=(a163*a14);
  a86=(a86+a12);
  a12=(a160*a114);
  a86=(a86+a12);
  a86=(a7*a86);
  a8=(a8-a86);
  a86=sin(a75);
  a12=(a125*a4);
  a0=(a122*a143);
  a12=(a12+a0);
  a0=(a122*a58);
  a12=(a12+a0);
  a0=(a125*a104);
  a12=(a12-a0);
  a12=(a86*a12);
  a8=(a8-a12);
  a12=cos(a75);
  a143=(a125*a143);
  a4=(a122*a4);
  a143=(a143-a4);
  a58=(a125*a58);
  a143=(a143+a58);
  a104=(a122*a104);
  a143=(a143+a104);
  a143=(a12*a143);
  a8=(a8+a143);
  a143=sin(a75);
  a104=(a80*a116);
  a58=(a77*a118);
  a104=(a104+a58);
  a58=(a77*a67);
  a104=(a104+a58);
  a58=(a80*a71);
  a104=(a104-a58);
  a104=(a143*a104);
  a8=(a8-a104);
  a75=cos(a75);
  a118=(a80*a118);
  a116=(a77*a116);
  a118=(a118-a116);
  a67=(a80*a67);
  a118=(a118+a67);
  a71=(a77*a71);
  a118=(a118+a71);
  a118=(a75*a118);
  a8=(a8+a118);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a165*a126);
  a118=(a2*a126);
  a8=(a8-a118);
  a118=(a163*a8);
  a71=(a160*a126);
  a118=(a118-a71);
  a161=(a182*a161);
  a118=(a118+a161);
  a161=(a162*a1);
  a118=(a118-a161);
  a39=(a39*a118);
  a182=(a182*a168);
  a162=(a162*a126);
  a182=(a182-a162);
  a2=(a2*a1);
  a165=(a165*a1);
  a2=(a2-a165);
  a163=(a163*a2);
  a182=(a182+a163);
  a160=(a160*a1);
  a182=(a182+a160);
  a7=(a7*a182);
  a39=(a39-a7);
  a7=(a125*a172);
  a182=(a122*a170);
  a7=(a7+a182);
  a182=(a122*a139);
  a7=(a7+a182);
  a182=(a125*a128);
  a7=(a7-a182);
  a86=(a86*a7);
  a39=(a39-a86);
  a170=(a125*a170);
  a172=(a122*a172);
  a170=(a170-a172);
  a125=(a125*a139);
  a170=(a170+a125);
  a122=(a122*a128);
  a170=(a170+a122);
  a12=(a12*a170);
  a39=(a39+a12);
  a12=(a80*a103);
  a170=(a77*a124);
  a12=(a12+a170);
  a170=(a77*a100);
  a12=(a12+a170);
  a170=(a80*a84);
  a12=(a12-a170);
  a143=(a143*a12);
  a39=(a39-a143);
  a124=(a80*a124);
  a103=(a77*a103);
  a124=(a124-a103);
  a80=(a80*a100);
  a124=(a124+a80);
  a77=(a77*a84);
  a124=(a124+a77);
  a75=(a75*a124);
  a39=(a39+a75);
  if (res[1]!=0) res[1][11]=a39;
  a39=-1.;
  if (res[1]!=0) res[1][12]=a39;
  a75=(a167*a74);
  a124=(a164*a114);
  a75=(a75-a124);
  a5=(a45*a5);
  a14=(a23*a14);
  a5=(a5+a14);
  a5=(a83*a5);
  a5=(a75+a5);
  a14=(a136*a88);
  a5=(a5+a14);
  a14=(a92*a73);
  a5=(a5+a14);
  a11=(a54*a11);
  a5=(a5+a11);
  a177=(a13*a177);
  a5=(a5+a177);
  if (res[1]!=0) res[1][13]=a5;
  a5=(a167*a126);
  a177=(a164*a1);
  a5=(a5-a177);
  a45=(a45*a8);
  a23=(a23*a2);
  a45=(a45+a23);
  a83=(a83*a45);
  a83=(a5+a83);
  a136=(a136*a176);
  a83=(a83+a136);
  a92=(a92*a63);
  a83=(a83+a92);
  a54=(a54*a50);
  a83=(a83+a54);
  a13=(a13*a9);
  a83=(a83+a13);
  if (res[1]!=0) res[1][14]=a83;
  if (res[1]!=0) res[1][15]=a39;
  a74=(a167*a74);
  a75=(a75-a74);
  a114=(a164*a114);
  a75=(a75+a114);
  a88=(a135*a88);
  a75=(a75+a88);
  a73=(a91*a73);
  a75=(a75+a73);
  if (res[1]!=0) res[1][16]=a75;
  a167=(a167*a126);
  a5=(a5-a167);
  a164=(a164*a1);
  a5=(a5+a164);
  a135=(a135*a176);
  a5=(a5+a135);
  a91=(a91*a63);
  a5=(a5+a91);
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

CASADI_SYMBOL_EXPORT int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15210121_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
