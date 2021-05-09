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
  #define CASADI_PREFIX(ID) model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_ ## ID
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

/* model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x4,8nz],o4[8x0]) */
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
  a8=(a8+a9);
  a9=casadi_sq(a8);
  a19=casadi_sq(a14);
  a9=(a9+a19);
  a9=sqrt(a9);
  a19=4.0000000000000001e-02;
  a20=(a9-a19);
  a21=8.7758256189037276e-01;
  a20=(a20/a21);
  a22=6.9999999999999996e-01;
  a23=(a20/a22);
  a24=1.;
  a23=(a23-a24);
  a25=casadi_sq(a23);
  a26=4.5000000000000001e-01;
  a25=(a25/a26);
  a25=(-a25);
  a25=exp(a25);
  a27=(a18*a25);
  a28=(a13*a1);
  a29=0.;
  a30=(a28<=a29);
  a31=fabs(a28);
  a32=10.;
  a31=(a31/a32);
  a31=(a24-a31);
  a33=fabs(a28);
  a33=(a33/a32);
  a33=(a24+a33);
  a31=(a31/a33);
  a34=(a30?a31:0);
  a35=(!a30);
  a36=1.3300000000000001e+00;
  a37=(a36*a28);
  a37=(a37/a32);
  a38=-8.2500000000000004e-02;
  a37=(a37/a38);
  a37=(a24-a37);
  a39=(a28/a32);
  a39=(a39/a38);
  a39=(a24-a39);
  a37=(a37/a39);
  a40=(a35?a37:0);
  a34=(a34+a40);
  a40=(a27*a34);
  a41=(a19<a20);
  a20=(a20/a22);
  a20=(a20-a24);
  a20=(a32*a20);
  a20=exp(a20);
  a42=(a20-a24);
  a43=1.4741315910257660e+02;
  a42=(a42/a43);
  a42=(a41?a42:0);
  a40=(a40+a42);
  a42=1.0000000000000001e-01;
  a44=7.;
  a45=(a28/a44);
  a45=(a42*a45);
  a40=(a40+a45);
  a45=(a17*a40);
  a46=(a13*a45);
  a47=sin(a6);
  a48=(a5*a47);
  a49=(a48+a5);
  a50=cos(a6);
  a51=(a5*a50);
  a52=(a49*a51);
  a53=(a5*a50);
  a47=(a5*a47);
  a54=(a53*a47);
  a52=(a52-a54);
  a54=(a48+a5);
  a55=casadi_sq(a54);
  a56=casadi_sq(a53);
  a55=(a55+a56);
  a55=sqrt(a55);
  a52=(a52/a55);
  a56=arg[0]? arg[0][1] : 0;
  a57=arg[2]? arg[2][1] : 0;
  a48=(a48+a5);
  a58=casadi_sq(a48);
  a59=casadi_sq(a53);
  a58=(a58+a59);
  a58=sqrt(a58);
  a59=(a58-a19);
  a59=(a59/a21);
  a60=(a59/a22);
  a60=(a60-a24);
  a61=casadi_sq(a60);
  a61=(a61/a26);
  a61=(-a61);
  a61=exp(a61);
  a62=(a57*a61);
  a63=(a52*a1);
  a64=(a63<=a29);
  a65=fabs(a63);
  a65=(a65/a32);
  a65=(a24-a65);
  a66=fabs(a63);
  a66=(a66/a32);
  a66=(a24+a66);
  a65=(a65/a66);
  a67=(a64?a65:0);
  a68=(!a64);
  a69=(a36*a63);
  a69=(a69/a32);
  a69=(a69/a38);
  a69=(a24-a69);
  a70=(a63/a32);
  a70=(a70/a38);
  a70=(a24-a70);
  a69=(a69/a70);
  a71=(a68?a69:0);
  a67=(a67+a71);
  a71=(a62*a67);
  a72=(a19<a59);
  a59=(a59/a22);
  a59=(a59-a24);
  a59=(a32*a59);
  a59=exp(a59);
  a73=(a59-a24);
  a73=(a73/a43);
  a73=(a72?a73:0);
  a71=(a71+a73);
  a73=(a63/a44);
  a73=(a42*a73);
  a71=(a71+a73);
  a73=(a56*a71);
  a74=(a52*a73);
  a46=(a46+a74);
  a74=arg[0]? arg[0][5] : 0;
  a75=sin(a74);
  a76=sin(a6);
  a77=(a75*a76);
  a78=cos(a74);
  a79=cos(a6);
  a80=(a78*a79);
  a77=(a77-a80);
  a80=(a5*a77);
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
  a89=(a5*a88);
  a89=(a87-a89);
  a86=(a86-a89);
  a90=(a85*a86);
  a91=(a5*a88);
  a91=(a87-a91);
  a92=(a83*a79);
  a93=(a91-a92);
  a94=(a5*a77);
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
  a92=(a84-a19);
  a92=(a92/a21);
  a101=(a92/a22);
  a101=(a101-a24);
  a102=casadi_sq(a101);
  a102=(a102/a26);
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
  a94=(a104<=a29);
  a111=fabs(a104);
  a111=(a111/a32);
  a111=(a24-a111);
  a112=fabs(a104);
  a112=(a112/a32);
  a112=(a24+a112);
  a111=(a111/a112);
  a113=(a94?a111:0);
  a114=(!a94);
  a115=(a36*a104);
  a115=(a115/a32);
  a115=(a115/a38);
  a115=(a24-a115);
  a116=(a104/a32);
  a116=(a116/a38);
  a116=(a24-a116);
  a115=(a115/a116);
  a117=(a114?a115:0);
  a113=(a113+a117);
  a117=(a103*a113);
  a118=(a19<a92);
  a92=(a92/a22);
  a92=(a92-a24);
  a92=(a32*a92);
  a92=exp(a92);
  a119=(a92-a24);
  a119=(a119/a43);
  a119=(a118?a119:0);
  a117=(a117+a119);
  a119=(a104/a44);
  a119=(a42*a119);
  a117=(a117+a119);
  a119=(a99*a117);
  a120=(a90*a119);
  a46=(a46+a120);
  a120=sin(a74);
  a121=sin(a6);
  a122=(a120*a121);
  a123=cos(a74);
  a124=cos(a6);
  a125=(a123*a124);
  a122=(a122-a125);
  a125=(a5*a122);
  a126=(a81*a121);
  a125=(a125-a126);
  a127=1.7500000000000000e+00;
  a128=(a127*a121);
  a129=(a125+a128);
  a130=(a127*a124);
  a131=(a81*a124);
  a132=(a123*a121);
  a133=(a120*a124);
  a132=(a132+a133);
  a133=(a5*a132);
  a133=(a131-a133);
  a130=(a130-a133);
  a134=(a129*a130);
  a135=(a5*a132);
  a135=(a131-a135);
  a136=(a127*a124);
  a137=(a135-a136);
  a138=(a5*a122);
  a138=(a138-a126);
  a139=(a127*a121);
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
  a143=arg[0]? arg[0][3] : 0;
  a144=arg[2]? arg[2][3] : 0;
  a125=(a125+a128);
  a128=casadi_sq(a125);
  a135=(a135-a136);
  a136=casadi_sq(a135);
  a128=(a128+a136);
  a128=sqrt(a128);
  a136=(a128-a19);
  a136=(a136/a21);
  a21=(a136/a22);
  a21=(a21-a24);
  a145=casadi_sq(a21);
  a145=(a145/a26);
  a145=(-a145);
  a145=exp(a145);
  a26=(a144*a145);
  a146=(a134*a1);
  a147=(a120*a124);
  a148=(a123*a121);
  a147=(a147+a148);
  a148=(a122*a126);
  a149=(a132*a131);
  a148=(a148+a149);
  a149=(a147*a148);
  a150=(a147*a126);
  a151=(a123*a124);
  a152=(a120*a121);
  a151=(a151-a152);
  a152=(a151*a131);
  a150=(a150+a152);
  a152=(a122*a150);
  a149=(a149-a152);
  a149=(a149-a133);
  a133=(a129*a149);
  a152=(a132*a150);
  a153=(a151*a148);
  a152=(a152-a153);
  a152=(a152+a138);
  a138=(a137*a152);
  a133=(a133+a138);
  a133=(a133/a141);
  a138=(a133*a2);
  a146=(a146+a138);
  a29=(a146<=a29);
  a138=fabs(a146);
  a138=(a138/a32);
  a138=(a24-a138);
  a153=fabs(a146);
  a153=(a153/a32);
  a153=(a24+a153);
  a138=(a138/a153);
  a154=(a29?a138:0);
  a155=(!a29);
  a156=(a36*a146);
  a156=(a156/a32);
  a156=(a156/a38);
  a156=(a24-a156);
  a157=(a146/a32);
  a157=(a157/a38);
  a157=(a24-a157);
  a156=(a156/a157);
  a38=(a155?a156:0);
  a154=(a154+a38);
  a38=(a26*a154);
  a19=(a19<a136);
  a136=(a136/a22);
  a136=(a136-a24);
  a136=(a32*a136);
  a136=exp(a136);
  a22=(a136-a24);
  a22=(a22/a43);
  a22=(a19?a22:0);
  a38=(a38+a22);
  a44=(a146/a44);
  a44=(a42*a44);
  a38=(a38+a44);
  a44=(a143*a38);
  a22=(a134*a44);
  a46=(a46+a22);
  a22=sin(a74);
  a43=cos(a74);
  a158=9.8100000000000005e+00;
  a159=cos(a6);
  a159=(a158*a159);
  a160=(a43*a159);
  a161=sin(a6);
  a161=(a158*a161);
  a162=(a22*a161);
  a160=(a160-a162);
  a162=(a81*a1);
  a163=(a43*a162);
  a164=(a163*a2);
  a160=(a160+a164);
  a164=(a1+a2);
  a165=(a164*a163);
  a160=(a160-a165);
  a165=(a22*a160);
  a166=(a22*a162);
  a167=(a164*a166);
  a168=(a43*a161);
  a169=(a22*a159);
  a168=(a168+a169);
  a169=(a166*a2);
  a168=(a168+a169);
  a167=(a167-a168);
  a168=(a43*a167);
  a165=(a165+a168);
  a165=(a81*a165);
  a46=(a46+a165);
  a165=(a4*a46);
  a168=9.6278838983177639e-01;
  a169=(a89*a119);
  a170=(a133*a44);
  a169=(a169+a170);
  a170=(a168*a169);
  a165=(a165+a170);
  a170=6.9253199970355839e-01;
  a165=(a165/a170);
  a170=(a3*a165);
  a171=9.6278838983177628e-01;
  a46=(a171*a46);
  a172=2.7025639012821789e-01;
  a169=(a172*a169);
  a46=(a46+a169);
  a170=(a170-a46);
  a46=3.7001900289039211e+00;
  a170=(a170/a46);
  a0=(a0-a170);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a0=(a0+a165);
  if (res[0]!=0) res[0][7]=a0;
  a0=3.9024390243902418e-01;
  a165=(a0*a13);
  a170=(a40*a165);
  if (res[1]!=0) res[1][0]=a170;
  a170=-3.9024390243902396e-01;
  a46=(a170*a13);
  a40=(a40*a46);
  if (res[1]!=0) res[1][1]=a40;
  a40=(a0*a52);
  a169=(a71*a40);
  if (res[1]!=0) res[1][2]=a169;
  a169=(a170*a52);
  a71=(a71*a169);
  if (res[1]!=0) res[1][3]=a71;
  a71=-3.9024390243902440e-01;
  a173=(a71*a89);
  a174=(a0*a90);
  a173=(a173+a174);
  a174=(a117*a173);
  if (res[1]!=0) res[1][4]=a174;
  a174=1.3902439024390245e+00;
  a175=(a174*a89);
  a176=(a170*a90);
  a175=(a175+a176);
  a117=(a117*a175);
  if (res[1]!=0) res[1][5]=a117;
  a117=(a71*a133);
  a176=(a0*a134);
  a117=(a117+a176);
  a176=(a38*a117);
  if (res[1]!=0) res[1][6]=a176;
  a176=(a174*a133);
  a177=(a170*a134);
  a176=(a176+a177);
  a38=(a38*a176);
  if (res[1]!=0) res[1][7]=a38;
  a38=cos(a6);
  a177=(a71*a44);
  a178=1.4285714285714285e-01;
  a117=(a143*a117);
  a179=(a42*a117);
  a179=(a178*a179);
  a180=-1.2121212121212121e+01;
  a181=(a26*a117);
  a156=(a156/a157);
  a182=(a181*a156);
  a182=(a180*a182);
  a182=(a42*a182);
  a182=(a155?a182:0);
  a179=(a179+a182);
  a182=(a181/a157);
  a182=(a180*a182);
  a182=(a42*a182);
  a182=(a36*a182);
  a182=(-a182);
  a182=(a155?a182:0);
  a179=(a179+a182);
  a138=(a138/a153);
  a182=(a181*a138);
  a182=(a42*a182);
  a183=casadi_sign(a146);
  a182=(a182*a183);
  a182=(-a182);
  a182=(a29?a182:0);
  a179=(a179+a182);
  a181=(a181/a153);
  a181=(a42*a181);
  a146=casadi_sign(a146);
  a181=(a181*a146);
  a181=(-a181);
  a181=(a29?a181:0);
  a179=(a179+a181);
  a181=(a2*a179);
  a177=(a177+a181);
  a181=(a177/a141);
  a182=(a137*a181);
  a184=(a132*a182);
  a185=(a129*a181);
  a186=(a122*a185);
  a184=(a184-a186);
  a186=(a126*a184);
  a187=(a148*a185);
  a186=(a186+a187);
  a187=(a123*a186);
  a188=(a131*a184);
  a189=(a148*a182);
  a188=(a188-a189);
  a189=(a120*a188);
  a187=(a187-a189);
  a189=(a0*a44);
  a190=(a1*a179);
  a189=(a189+a190);
  a190=(a189/a141);
  a191=(a137*a190);
  a192=(a127*a191);
  a187=(a187+a192);
  a192=(a150*a182);
  a193=(a147*a185);
  a194=(a151*a182);
  a193=(a193-a194);
  a194=(a131*a193);
  a192=(a192+a194);
  a135=(a135+a135);
  a194=1.1394939273245490e+00;
  a195=1.4285714285714286e+00;
  a196=6.7836549063042314e-03;
  a197=(a196*a117);
  a197=(a197*a136);
  a197=(a32*a197);
  a197=(a195*a197);
  a197=(a19?a197:0);
  a21=(a21+a21);
  a198=2.2222222222222223e+00;
  a117=(a154*a117);
  a117=(a144*a117);
  a117=(a145*a117);
  a117=(a198*a117);
  a117=(a21*a117);
  a117=(a195*a117);
  a197=(a197-a117);
  a197=(a194*a197);
  a128=(a128+a128);
  a197=(a197/a128);
  a117=(a135*a197);
  a142=(a142+a142);
  a199=(a133/a141);
  a177=(a199*a177);
  a200=(a134/a141);
  a189=(a200*a189);
  a177=(a177+a189);
  a189=(a141+a141);
  a177=(a177/a189);
  a201=(a142*a177);
  a202=(a117-a201);
  a203=(a152*a181);
  a204=(a139*a190);
  a203=(a203+a204);
  a202=(a202+a203);
  a204=(a5*a202);
  a192=(a192-a204);
  a204=(a129*a190);
  a205=(a185+a204);
  a206=(a5*a205);
  a192=(a192+a206);
  a206=(a123*a192);
  a187=(a187+a206);
  a125=(a125+a125);
  a197=(a125*a197);
  a140=(a140+a140);
  a177=(a140*a177);
  a206=(a197-a177);
  a181=(a149*a181);
  a190=(a130*a190);
  a181=(a181+a190);
  a206=(a206+a181);
  a206=(a127*a206);
  a187=(a187+a206);
  a206=(a147*a184);
  a190=(a122*a193);
  a206=(a206+a190);
  a182=(a182+a191);
  a206=(a206-a182);
  a197=(a197-a177);
  a197=(a197+a181);
  a206=(a206-a197);
  a206=(a81*a206);
  a187=(a187+a206);
  a206=(a126*a193);
  a185=(a150*a185);
  a206=(a206-a185);
  a182=(a5*a182);
  a206=(a206+a182);
  a197=(a5*a197);
  a206=(a206+a197);
  a197=(a120*a206);
  a187=(a187+a197);
  a187=(a38*a187);
  a197=cos(a6);
  a182=4.8780487804878025e-01;
  a185=(a182*a43);
  a181=(a43*a185);
  a177=(a182*a22);
  a191=(a22*a177);
  a181=(a181+a191);
  a181=(a158*a181);
  a181=(a197*a181);
  a191=sin(a6);
  a190=(a43*a177);
  a207=(a22*a185);
  a190=(a190-a207);
  a190=(a158*a190);
  a190=(a191*a190);
  a181=(a181+a190);
  a190=sin(a6);
  a207=(a123*a188);
  a208=(a120*a186);
  a207=(a207+a208);
  a201=(a201-a117);
  a201=(a201-a203);
  a201=(a127*a201);
  a207=(a207+a201);
  a201=(a120*a192);
  a207=(a207+a201);
  a184=(a151*a184);
  a193=(a132*a193);
  a184=(a184+a193);
  a184=(a184+a202);
  a184=(a184-a205);
  a184=(a81*a184);
  a207=(a207+a184);
  a204=(a127*a204);
  a207=(a207+a204);
  a204=(a123*a206);
  a207=(a207-a204);
  a207=(a190*a207);
  a181=(a181+a207);
  a187=(a187-a181);
  a181=sin(a6);
  a71=(a71*a119);
  a173=(a99*a173);
  a207=(a42*a173);
  a207=(a178*a207);
  a204=(a103*a173);
  a115=(a115/a116);
  a184=(a204*a115);
  a184=(a180*a184);
  a184=(a42*a184);
  a184=(a114?a184:0);
  a207=(a207+a184);
  a184=(a204/a116);
  a184=(a180*a184);
  a184=(a42*a184);
  a184=(a36*a184);
  a184=(-a184);
  a184=(a114?a184:0);
  a207=(a207+a184);
  a111=(a111/a112);
  a184=(a204*a111);
  a184=(a42*a184);
  a205=casadi_sign(a104);
  a184=(a184*a205);
  a184=(-a184);
  a184=(a94?a184:0);
  a207=(a207+a184);
  a204=(a204/a112);
  a204=(a42*a204);
  a104=casadi_sign(a104);
  a204=(a204*a104);
  a204=(-a204);
  a204=(a94?a204:0);
  a207=(a207+a204);
  a204=(a2*a207);
  a71=(a71+a204);
  a204=(a71/a97);
  a184=(a93*a204);
  a202=(a88*a184);
  a193=(a85*a204);
  a201=(a77*a193);
  a202=(a202-a201);
  a201=(a87*a202);
  a203=(a106*a184);
  a201=(a201-a203);
  a203=(a78*a201);
  a117=(a82*a202);
  a208=(a106*a193);
  a117=(a117+a208);
  a208=(a75*a117);
  a203=(a203+a208);
  a98=(a98+a98);
  a208=(a89/a97);
  a71=(a208*a71);
  a209=(a90/a97);
  a210=(a0*a119);
  a211=(a1*a207);
  a210=(a210+a211);
  a211=(a209*a210);
  a71=(a71+a211);
  a211=(a97+a97);
  a71=(a71/a211);
  a212=(a98*a71);
  a91=(a91+a91);
  a213=(a196*a173);
  a213=(a213*a92);
  a213=(a32*a213);
  a213=(a195*a213);
  a213=(a118?a213:0);
  a101=(a101+a101);
  a173=(a113*a173);
  a173=(a100*a173);
  a173=(a102*a173);
  a173=(a198*a173);
  a173=(a101*a173);
  a173=(a195*a173);
  a213=(a213-a173);
  a213=(a194*a213);
  a84=(a84+a84);
  a213=(a213/a84);
  a173=(a91*a213);
  a214=(a212-a173);
  a215=(a110*a204);
  a210=(a210/a97);
  a216=(a95*a210);
  a215=(a215+a216);
  a214=(a214-a215);
  a214=(a83*a214);
  a203=(a203+a214);
  a214=(a108*a184);
  a216=(a105*a193);
  a217=(a109*a184);
  a216=(a216-a217);
  a217=(a87*a216);
  a214=(a214+a217);
  a173=(a173-a212);
  a173=(a173+a215);
  a215=(a5*a173);
  a214=(a214-a215);
  a215=(a85*a210);
  a212=(a193+a215);
  a217=(a5*a212);
  a214=(a214+a217);
  a217=(a75*a214);
  a203=(a203+a217);
  a217=(a109*a202);
  a218=(a88*a216);
  a217=(a217+a218);
  a217=(a217+a173);
  a217=(a217-a212);
  a217=(a81*a217);
  a203=(a203+a217);
  a215=(a83*a215);
  a203=(a203+a215);
  a215=(a82*a216);
  a193=(a108*a193);
  a215=(a215-a193);
  a193=(a93*a210);
  a184=(a184+a193);
  a217=(a5*a184);
  a215=(a215+a217);
  a80=(a80+a80);
  a213=(a80*a213);
  a96=(a96+a96);
  a71=(a96*a71);
  a217=(a213-a71);
  a204=(a107*a204);
  a210=(a86*a210);
  a204=(a204+a210);
  a217=(a217+a204);
  a210=(a5*a217);
  a215=(a215+a210);
  a210=(a78*a215);
  a203=(a203-a210);
  a203=(a181*a203);
  a187=(a187-a203);
  a203=cos(a6);
  a210=(a78*a117);
  a212=(a75*a201);
  a210=(a210-a212);
  a193=(a83*a193);
  a210=(a210+a193);
  a193=(a78*a214);
  a210=(a210+a193);
  a213=(a213-a71);
  a213=(a213+a204);
  a213=(a83*a213);
  a210=(a210+a213);
  a202=(a105*a202);
  a216=(a77*a216);
  a202=(a202+a216);
  a202=(a202-a184);
  a202=(a202-a217);
  a202=(a81*a202);
  a210=(a210+a202);
  a202=(a75*a215);
  a210=(a210+a202);
  a210=(a203*a210);
  a187=(a187+a210);
  a210=sin(a6);
  a40=(a56*a40);
  a202=(a196*a40);
  a202=(a202*a59);
  a202=(a32*a202);
  a202=(a195*a202);
  a202=(a72?a202:0);
  a60=(a60+a60);
  a217=(a67*a40);
  a217=(a57*a217);
  a217=(a61*a217);
  a217=(a198*a217);
  a217=(a60*a217);
  a217=(a195*a217);
  a202=(a202-a217);
  a202=(a194*a202);
  a58=(a58+a58);
  a202=(a202/a58);
  a217=(a50*a202);
  a184=(a52/a55);
  a216=(a0*a73);
  a213=(a42*a40);
  a213=(a178*a213);
  a40=(a62*a40);
  a69=(a69/a70);
  a204=(a40*a69);
  a204=(a180*a204);
  a204=(a42*a204);
  a204=(a68?a204:0);
  a213=(a213+a204);
  a204=(a40/a70);
  a204=(a180*a204);
  a204=(a42*a204);
  a204=(a36*a204);
  a204=(-a204);
  a204=(a68?a204:0);
  a213=(a213+a204);
  a65=(a65/a66);
  a204=(a40*a65);
  a204=(a42*a204);
  a71=casadi_sign(a63);
  a204=(a204*a71);
  a204=(-a204);
  a204=(a64?a204:0);
  a213=(a213+a204);
  a40=(a40/a66);
  a40=(a42*a40);
  a63=casadi_sign(a63);
  a40=(a40*a63);
  a40=(-a40);
  a40=(a64?a40:0);
  a213=(a213+a40);
  a40=(a1*a213);
  a216=(a216+a40);
  a40=(a184*a216);
  a204=(a55+a55);
  a40=(a40/a204);
  a193=(a50*a40);
  a217=(a217-a193);
  a216=(a216/a55);
  a193=(a47*a216);
  a217=(a217-a193);
  a217=(a5*a217);
  a193=(a49*a216);
  a193=(a5*a193);
  a217=(a217+a193);
  a217=(a210*a217);
  a187=(a187-a217);
  a217=cos(a6);
  a48=(a48+a48);
  a202=(a48*a202);
  a54=(a54+a54);
  a40=(a54*a40);
  a202=(a202-a40);
  a40=(a51*a216);
  a202=(a202+a40);
  a202=(a5*a202);
  a216=(a53*a216);
  a216=(a5*a216);
  a202=(a202-a216);
  a202=(a217*a202);
  a187=(a187+a202);
  a202=sin(a6);
  a165=(a17*a165);
  a216=(a196*a165);
  a216=(a216*a20);
  a216=(a32*a216);
  a216=(a195*a216);
  a216=(a41?a216:0);
  a23=(a23+a23);
  a40=(a34*a165);
  a40=(a18*a40);
  a40=(a25*a40);
  a40=(a198*a40);
  a40=(a23*a40);
  a40=(a195*a40);
  a216=(a216-a40);
  a216=(a194*a216);
  a9=(a9+a9);
  a216=(a216/a9);
  a40=(a11*a216);
  a193=(a13/a16);
  a0=(a0*a45);
  a212=(a42*a165);
  a212=(a178*a212);
  a165=(a27*a165);
  a37=(a37/a39);
  a173=(a165*a37);
  a173=(a180*a173);
  a173=(a42*a173);
  a173=(a35?a173:0);
  a212=(a212+a173);
  a173=(a165/a39);
  a173=(a180*a173);
  a173=(a42*a173);
  a173=(a36*a173);
  a173=(-a173);
  a173=(a35?a173:0);
  a212=(a212+a173);
  a31=(a31/a33);
  a173=(a165*a31);
  a173=(a42*a173);
  a218=casadi_sign(a28);
  a173=(a173*a218);
  a173=(-a173);
  a173=(a30?a173:0);
  a212=(a212+a173);
  a165=(a165/a33);
  a165=(a42*a165);
  a28=casadi_sign(a28);
  a165=(a165*a28);
  a165=(-a165);
  a165=(a30?a165:0);
  a212=(a212+a165);
  a165=(a1*a212);
  a0=(a0+a165);
  a165=(a193*a0);
  a173=(a16+a16);
  a165=(a165/a173);
  a219=(a11*a165);
  a40=(a40-a219);
  a0=(a0/a16);
  a219=(a7*a0);
  a40=(a40-a219);
  a40=(a5*a40);
  a219=(a10*a0);
  a219=(a5*a219);
  a40=(a40+a219);
  a40=(a202*a40);
  a187=(a187-a40);
  a6=cos(a6);
  a8=(a8+a8);
  a216=(a8*a216);
  a15=(a15+a15);
  a165=(a15*a165);
  a216=(a216-a165);
  a165=(a12*a0);
  a216=(a216+a165);
  a216=(a5*a216);
  a0=(a14*a0);
  a0=(a5*a0);
  a216=(a216-a0);
  a216=(a6*a216);
  a187=(a187+a216);
  if (res[1]!=0) res[1][8]=a187;
  a187=(a174*a44);
  a176=(a143*a176);
  a216=(a42*a176);
  a216=(a178*a216);
  a26=(a26*a176);
  a156=(a26*a156);
  a156=(a180*a156);
  a156=(a42*a156);
  a156=(a155?a156:0);
  a216=(a216+a156);
  a157=(a26/a157);
  a157=(a180*a157);
  a157=(a42*a157);
  a157=(a36*a157);
  a157=(-a157);
  a155=(a155?a157:0);
  a216=(a216+a155);
  a138=(a26*a138);
  a138=(a42*a138);
  a138=(a138*a183);
  a138=(-a138);
  a138=(a29?a138:0);
  a216=(a216+a138);
  a26=(a26/a153);
  a26=(a42*a26);
  a26=(a26*a146);
  a26=(-a26);
  a29=(a29?a26:0);
  a216=(a216+a29);
  a29=(a2*a216);
  a187=(a187+a29);
  a29=(a187/a141);
  a26=(a137*a29);
  a146=(a132*a26);
  a153=(a129*a29);
  a138=(a122*a153);
  a146=(a146-a138);
  a138=(a126*a146);
  a183=(a148*a153);
  a138=(a138+a183);
  a183=(a123*a138);
  a155=(a131*a146);
  a148=(a148*a26);
  a155=(a155-a148);
  a148=(a120*a155);
  a183=(a183-a148);
  a44=(a170*a44);
  a148=(a1*a216);
  a44=(a44+a148);
  a141=(a44/a141);
  a137=(a137*a141);
  a148=(a127*a137);
  a183=(a183+a148);
  a148=(a150*a26);
  a157=(a147*a153);
  a156=(a151*a26);
  a157=(a157-a156);
  a131=(a131*a157);
  a148=(a148+a131);
  a131=(a196*a176);
  a131=(a131*a136);
  a131=(a32*a131);
  a131=(a195*a131);
  a19=(a19?a131:0);
  a176=(a154*a176);
  a144=(a144*a176);
  a144=(a145*a144);
  a144=(a198*a144);
  a21=(a21*a144);
  a21=(a195*a21);
  a19=(a19-a21);
  a19=(a194*a19);
  a19=(a19/a128);
  a135=(a135*a19);
  a199=(a199*a187);
  a200=(a200*a44);
  a199=(a199+a200);
  a199=(a199/a189);
  a142=(a142*a199);
  a189=(a135-a142);
  a152=(a152*a29);
  a139=(a139*a141);
  a152=(a152+a139);
  a189=(a189+a152);
  a139=(a5*a189);
  a148=(a148-a139);
  a129=(a129*a141);
  a139=(a153+a129);
  a200=(a5*a139);
  a148=(a148+a200);
  a200=(a123*a148);
  a183=(a183+a200);
  a125=(a125*a19);
  a140=(a140*a199);
  a199=(a125-a140);
  a149=(a149*a29);
  a130=(a130*a141);
  a149=(a149+a130);
  a199=(a199+a149);
  a199=(a127*a199);
  a183=(a183+a199);
  a147=(a147*a146);
  a122=(a122*a157);
  a147=(a147+a122);
  a26=(a26+a137);
  a147=(a147-a26);
  a125=(a125-a140);
  a125=(a125+a149);
  a147=(a147-a125);
  a147=(a81*a147);
  a183=(a183+a147);
  a126=(a126*a157);
  a150=(a150*a153);
  a126=(a126-a150);
  a26=(a5*a26);
  a126=(a126+a26);
  a125=(a5*a125);
  a126=(a126+a125);
  a125=(a120*a126);
  a183=(a183+a125);
  a38=(a38*a183);
  a183=-4.8780487804877992e-01;
  a125=(a183*a43);
  a26=(a43*a125);
  a150=(a183*a22);
  a153=(a22*a150);
  a26=(a26+a153);
  a26=(a158*a26);
  a197=(a197*a26);
  a26=(a43*a150);
  a153=(a22*a125);
  a26=(a26-a153);
  a158=(a158*a26);
  a191=(a191*a158);
  a197=(a197+a191);
  a191=(a123*a155);
  a158=(a120*a138);
  a191=(a191+a158);
  a142=(a142-a135);
  a142=(a142-a152);
  a142=(a127*a142);
  a191=(a191+a142);
  a120=(a120*a148);
  a191=(a191+a120);
  a151=(a151*a146);
  a132=(a132*a157);
  a151=(a151+a132);
  a151=(a151+a189);
  a151=(a151-a139);
  a151=(a81*a151);
  a191=(a191+a151);
  a127=(a127*a129);
  a191=(a191+a127);
  a123=(a123*a126);
  a191=(a191-a123);
  a190=(a190*a191);
  a197=(a197+a190);
  a38=(a38-a197);
  a174=(a174*a119);
  a175=(a99*a175);
  a197=(a42*a175);
  a197=(a178*a197);
  a103=(a103*a175);
  a115=(a103*a115);
  a115=(a180*a115);
  a115=(a42*a115);
  a115=(a114?a115:0);
  a197=(a197+a115);
  a116=(a103/a116);
  a116=(a180*a116);
  a116=(a42*a116);
  a116=(a36*a116);
  a116=(-a116);
  a114=(a114?a116:0);
  a197=(a197+a114);
  a111=(a103*a111);
  a111=(a42*a111);
  a111=(a111*a205);
  a111=(-a111);
  a111=(a94?a111:0);
  a197=(a197+a111);
  a103=(a103/a112);
  a103=(a42*a103);
  a103=(a103*a104);
  a103=(-a103);
  a94=(a94?a103:0);
  a197=(a197+a94);
  a94=(a2*a197);
  a174=(a174+a94);
  a94=(a174/a97);
  a103=(a93*a94);
  a104=(a88*a103);
  a112=(a85*a94);
  a111=(a77*a112);
  a104=(a104-a111);
  a111=(a87*a104);
  a205=(a106*a103);
  a111=(a111-a205);
  a205=(a78*a111);
  a114=(a82*a104);
  a106=(a106*a112);
  a114=(a114+a106);
  a106=(a75*a114);
  a205=(a205+a106);
  a208=(a208*a174);
  a119=(a170*a119);
  a174=(a1*a197);
  a119=(a119+a174);
  a209=(a209*a119);
  a208=(a208+a209);
  a208=(a208/a211);
  a98=(a98*a208);
  a211=(a196*a175);
  a211=(a211*a92);
  a211=(a32*a211);
  a211=(a195*a211);
  a118=(a118?a211:0);
  a175=(a113*a175);
  a100=(a100*a175);
  a100=(a102*a100);
  a100=(a198*a100);
  a101=(a101*a100);
  a101=(a195*a101);
  a118=(a118-a101);
  a118=(a194*a118);
  a118=(a118/a84);
  a91=(a91*a118);
  a84=(a98-a91);
  a110=(a110*a94);
  a119=(a119/a97);
  a95=(a95*a119);
  a110=(a110+a95);
  a84=(a84-a110);
  a84=(a83*a84);
  a205=(a205+a84);
  a84=(a108*a103);
  a95=(a105*a112);
  a97=(a109*a103);
  a95=(a95-a97);
  a87=(a87*a95);
  a84=(a84+a87);
  a91=(a91-a98);
  a91=(a91+a110);
  a110=(a5*a91);
  a84=(a84-a110);
  a85=(a85*a119);
  a110=(a112+a85);
  a98=(a5*a110);
  a84=(a84+a98);
  a98=(a75*a84);
  a205=(a205+a98);
  a109=(a109*a104);
  a88=(a88*a95);
  a109=(a109+a88);
  a109=(a109+a91);
  a109=(a109-a110);
  a109=(a81*a109);
  a205=(a205+a109);
  a85=(a83*a85);
  a205=(a205+a85);
  a82=(a82*a95);
  a108=(a108*a112);
  a82=(a82-a108);
  a93=(a93*a119);
  a103=(a103+a93);
  a108=(a5*a103);
  a82=(a82+a108);
  a80=(a80*a118);
  a96=(a96*a208);
  a208=(a80-a96);
  a107=(a107*a94);
  a86=(a86*a119);
  a107=(a107+a86);
  a208=(a208+a107);
  a86=(a5*a208);
  a82=(a82+a86);
  a86=(a78*a82);
  a205=(a205-a86);
  a181=(a181*a205);
  a38=(a38-a181);
  a181=(a78*a114);
  a205=(a75*a111);
  a181=(a181-a205);
  a93=(a83*a93);
  a181=(a181+a93);
  a78=(a78*a84);
  a181=(a181+a78);
  a80=(a80-a96);
  a80=(a80+a107);
  a83=(a83*a80);
  a181=(a181+a83);
  a105=(a105*a104);
  a77=(a77*a95);
  a105=(a105+a77);
  a105=(a105-a103);
  a105=(a105-a208);
  a105=(a81*a105);
  a181=(a181+a105);
  a75=(a75*a82);
  a181=(a181+a75);
  a203=(a203*a181);
  a38=(a38+a203);
  a169=(a56*a169);
  a203=(a196*a169);
  a203=(a203*a59);
  a203=(a32*a203);
  a203=(a195*a203);
  a72=(a72?a203:0);
  a203=(a67*a169);
  a57=(a57*a203);
  a57=(a61*a57);
  a57=(a198*a57);
  a60=(a60*a57);
  a60=(a195*a60);
  a72=(a72-a60);
  a72=(a194*a72);
  a72=(a72/a58);
  a58=(a50*a72);
  a73=(a170*a73);
  a60=(a42*a169);
  a60=(a178*a60);
  a62=(a62*a169);
  a69=(a62*a69);
  a69=(a180*a69);
  a69=(a42*a69);
  a69=(a68?a69:0);
  a60=(a60+a69);
  a70=(a62/a70);
  a70=(a180*a70);
  a70=(a42*a70);
  a70=(a36*a70);
  a70=(-a70);
  a68=(a68?a70:0);
  a60=(a60+a68);
  a65=(a62*a65);
  a65=(a42*a65);
  a65=(a65*a71);
  a65=(-a65);
  a65=(a64?a65:0);
  a60=(a60+a65);
  a62=(a62/a66);
  a62=(a42*a62);
  a62=(a62*a63);
  a62=(-a62);
  a64=(a64?a62:0);
  a60=(a60+a64);
  a64=(a1*a60);
  a73=(a73+a64);
  a184=(a184*a73);
  a184=(a184/a204);
  a50=(a50*a184);
  a58=(a58-a50);
  a73=(a73/a55);
  a47=(a47*a73);
  a58=(a58-a47);
  a58=(a5*a58);
  a49=(a49*a73);
  a49=(a5*a49);
  a58=(a58+a49);
  a210=(a210*a58);
  a38=(a38-a210);
  a48=(a48*a72);
  a54=(a54*a184);
  a48=(a48-a54);
  a51=(a51*a73);
  a48=(a48+a51);
  a48=(a5*a48);
  a53=(a53*a73);
  a53=(a5*a53);
  a48=(a48-a53);
  a217=(a217*a48);
  a38=(a38+a217);
  a46=(a17*a46);
  a196=(a196*a46);
  a196=(a196*a20);
  a32=(a32*a196);
  a32=(a195*a32);
  a41=(a41?a32:0);
  a32=(a34*a46);
  a18=(a18*a32);
  a18=(a25*a18);
  a198=(a198*a18);
  a23=(a23*a198);
  a195=(a195*a23);
  a41=(a41-a195);
  a194=(a194*a41);
  a194=(a194/a9);
  a9=(a11*a194);
  a170=(a170*a45);
  a45=(a42*a46);
  a178=(a178*a45);
  a27=(a27*a46);
  a37=(a27*a37);
  a37=(a180*a37);
  a37=(a42*a37);
  a37=(a35?a37:0);
  a178=(a178+a37);
  a39=(a27/a39);
  a180=(a180*a39);
  a180=(a42*a180);
  a36=(a36*a180);
  a36=(-a36);
  a35=(a35?a36:0);
  a178=(a178+a35);
  a31=(a27*a31);
  a31=(a42*a31);
  a31=(a31*a218);
  a31=(-a31);
  a31=(a30?a31:0);
  a178=(a178+a31);
  a27=(a27/a33);
  a42=(a42*a27);
  a42=(a42*a28);
  a42=(-a42);
  a30=(a30?a42:0);
  a178=(a178+a30);
  a1=(a1*a178);
  a170=(a170+a1);
  a193=(a193*a170);
  a193=(a193/a173);
  a11=(a11*a193);
  a9=(a9-a11);
  a170=(a170/a16);
  a7=(a7*a170);
  a9=(a9-a7);
  a9=(a5*a9);
  a10=(a10*a170);
  a10=(a5*a10);
  a9=(a9+a10);
  a202=(a202*a9);
  a38=(a38-a202);
  a8=(a8*a194);
  a15=(a15*a193);
  a8=(a8-a15);
  a12=(a12*a170);
  a8=(a8+a12);
  a8=(a5*a8);
  a14=(a14*a170);
  a5=(a5*a14);
  a8=(a8-a5);
  a6=(a6*a8);
  a38=(a38+a6);
  if (res[1]!=0) res[1][9]=a38;
  a38=cos(a74);
  a6=(a164*a185);
  a8=(a2*a185);
  a6=(a6-a8);
  a8=(a162*a6);
  a5=(a159*a185);
  a8=(a8-a5);
  a5=(a182*a160);
  a8=(a8+a5);
  a5=(a161*a177);
  a8=(a8-a5);
  a8=(a38*a8);
  a5=sin(a74);
  a182=(a182*a167);
  a14=(a161*a185);
  a182=(a182-a14);
  a14=(a2*a177);
  a170=(a164*a177);
  a14=(a14-a170);
  a170=(a162*a14);
  a182=(a182+a170);
  a170=(a159*a177);
  a182=(a182+a170);
  a182=(a5*a182);
  a8=(a8-a182);
  a182=sin(a74);
  a170=(a124*a188);
  a12=(a121*a186);
  a170=(a170+a12);
  a12=(a121*a192);
  a170=(a170+a12);
  a12=(a124*a206);
  a170=(a170-a12);
  a170=(a182*a170);
  a8=(a8-a170);
  a170=cos(a74);
  a186=(a124*a186);
  a188=(a121*a188);
  a186=(a186-a188);
  a192=(a124*a192);
  a186=(a186+a192);
  a206=(a121*a206);
  a186=(a186+a206);
  a186=(a170*a186);
  a8=(a8+a186);
  a186=sin(a74);
  a206=(a79*a201);
  a192=(a76*a117);
  a206=(a206+a192);
  a192=(a76*a214);
  a206=(a206+a192);
  a192=(a79*a215);
  a206=(a206-a192);
  a206=(a186*a206);
  a8=(a8-a206);
  a74=cos(a74);
  a117=(a79*a117);
  a201=(a76*a201);
  a117=(a117-a201);
  a214=(a79*a214);
  a117=(a117+a214);
  a215=(a76*a215);
  a117=(a117+a215);
  a117=(a74*a117);
  a8=(a8+a117);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a164*a125);
  a117=(a2*a125);
  a8=(a8-a117);
  a117=(a162*a8);
  a215=(a159*a125);
  a117=(a117-a215);
  a160=(a183*a160);
  a117=(a117+a160);
  a160=(a161*a150);
  a117=(a117-a160);
  a38=(a38*a117);
  a183=(a183*a167);
  a161=(a161*a125);
  a183=(a183-a161);
  a2=(a2*a150);
  a164=(a164*a150);
  a2=(a2-a164);
  a162=(a162*a2);
  a183=(a183+a162);
  a159=(a159*a150);
  a183=(a183+a159);
  a5=(a5*a183);
  a38=(a38-a5);
  a5=(a124*a155);
  a183=(a121*a138);
  a5=(a5+a183);
  a183=(a121*a148);
  a5=(a5+a183);
  a183=(a124*a126);
  a5=(a5-a183);
  a182=(a182*a5);
  a38=(a38-a182);
  a138=(a124*a138);
  a155=(a121*a155);
  a138=(a138-a155);
  a124=(a124*a148);
  a138=(a138+a124);
  a121=(a121*a126);
  a138=(a138+a121);
  a170=(a170*a138);
  a38=(a38+a170);
  a170=(a79*a111);
  a138=(a76*a114);
  a170=(a170+a138);
  a138=(a76*a84);
  a170=(a170+a138);
  a138=(a79*a82);
  a170=(a170-a138);
  a186=(a186*a170);
  a38=(a38-a186);
  a114=(a79*a114);
  a111=(a76*a111);
  a114=(a114-a111);
  a79=(a79*a84);
  a114=(a114+a79);
  a76=(a76*a82);
  a114=(a114+a76);
  a74=(a74*a114);
  a38=(a38+a74);
  if (res[1]!=0) res[1][11]=a38;
  a38=-1.;
  if (res[1]!=0) res[1][12]=a38;
  a74=(a166*a185);
  a114=(a163*a177);
  a74=(a74-a114);
  a6=(a22*a6);
  a14=(a43*a14);
  a6=(a6+a14);
  a6=(a81*a6);
  a6=(a74+a6);
  a14=(a134*a179);
  a6=(a6+a14);
  a14=(a90*a207);
  a6=(a6+a14);
  a213=(a52*a213);
  a6=(a6+a213);
  a212=(a13*a212);
  a6=(a6+a212);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a166*a125);
  a212=(a163*a150);
  a6=(a6-a212);
  a22=(a22*a8);
  a43=(a43*a2);
  a22=(a22+a43);
  a81=(a81*a22);
  a81=(a6+a81);
  a22=(a134*a216);
  a81=(a81+a22);
  a22=(a90*a197);
  a81=(a81+a22);
  a60=(a52*a60);
  a81=(a81+a60);
  a178=(a13*a178);
  a81=(a81+a178);
  if (res[1]!=0) res[1][14]=a81;
  if (res[1]!=0) res[1][15]=a38;
  a185=(a166*a185);
  a74=(a74-a185);
  a177=(a163*a177);
  a74=(a74+a177);
  a179=(a133*a179);
  a74=(a74+a179);
  a207=(a89*a207);
  a74=(a74+a207);
  if (res[1]!=0) res[1][16]=a74;
  a166=(a166*a125);
  a6=(a6-a166);
  a163=(a163*a150);
  a6=(a6+a163);
  a216=(a133*a216);
  a6=(a6+a216);
  a197=(a89*a197);
  a6=(a6+a197);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a24;
  if (res[2]!=0) res[2][1]=a24;
  if (res[2]!=0) res[2][2]=a24;
  if (res[2]!=0) res[2][3]=a24;
  if (res[2]!=0) res[2][4]=a24;
  if (res[2]!=0) res[2][5]=a24;
  if (res[2]!=0) res[2][6]=a24;
  if (res[2]!=0) res[2][7]=a24;
  a24=1.4439765966454325e+00;
  a34=(a34*a25);
  a17=(a17*a34);
  a13=(a13*a17);
  a17=(a4*a13);
  a17=(a24*a17);
  a34=(a3*a17);
  a13=(a171*a13);
  a34=(a34-a13);
  a34=(a172*a34);
  a34=(-a34);
  if (res[3]!=0) res[3][0]=a34;
  if (res[3]!=0) res[3][1]=a17;
  a67=(a67*a61);
  a56=(a56*a67);
  a52=(a52*a56);
  a56=(a4*a52);
  a56=(a24*a56);
  a67=(a3*a56);
  a52=(a171*a52);
  a67=(a67-a52);
  a67=(a172*a67);
  a67=(-a67);
  if (res[3]!=0) res[3][2]=a67;
  if (res[3]!=0) res[3][3]=a56;
  a113=(a113*a102);
  a99=(a99*a113);
  a90=(a90*a99);
  a113=(a4*a90);
  a89=(a89*a99);
  a99=(a168*a89);
  a113=(a113+a99);
  a113=(a24*a113);
  a99=(a3*a113);
  a90=(a171*a90);
  a89=(a172*a89);
  a90=(a90+a89);
  a99=(a99-a90);
  a99=(a172*a99);
  a99=(-a99);
  if (res[3]!=0) res[3][4]=a99;
  if (res[3]!=0) res[3][5]=a113;
  a154=(a154*a145);
  a143=(a143*a154);
  a134=(a134*a143);
  a4=(a4*a134);
  a133=(a133*a143);
  a168=(a168*a133);
  a4=(a4+a168);
  a24=(a24*a4);
  a3=(a3*a24);
  a171=(a171*a134);
  a133=(a172*a133);
  a171=(a171+a133);
  a3=(a3-a171);
  a172=(a172*a3);
  a172=(-a172);
  if (res[3]!=0) res[3][6]=a172;
  if (res[3]!=0) res[3][7]=a24;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    case 4: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_09391998_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
