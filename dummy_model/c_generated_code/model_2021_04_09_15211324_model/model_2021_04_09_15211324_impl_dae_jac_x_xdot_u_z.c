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
  #define CASADI_PREFIX(ID) model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_ ## ID
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

/* model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8x8,18nz],o1[8x8,8nz],o2[8x4,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[2]? arg[2][0] : 0;
  a1=5.0562195126178466e-01;
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
  a8=(a1*a8);
  a20=(a19*a8);
  a3=(a1*a3);
  a21=(a9*a3);
  a20=(a20-a21);
  a4=(a4+a5);
  a5=casadi_sq(a4);
  a21=casadi_sq(a9);
  a5=(a5+a21);
  a5=sqrt(a5);
  a20=(a20/a5);
  a21=arg[0]? arg[0][6] : 0;
  a22=(a20*a21);
  a23=0.;
  a24=(a22<=a23);
  a25=fabs(a22);
  a26=10.;
  a25=(a25/a26);
  a25=(a15-a25);
  a27=fabs(a22);
  a27=(a27/a26);
  a27=(a15+a27);
  a25=(a25/a27);
  a28=(a24?a25:0);
  a29=(!a24);
  a30=1.3300000000000001e+00;
  a31=(a30*a22);
  a31=(a31/a26);
  a32=-8.2500000000000004e-02;
  a31=(a31/a32);
  a31=(a15-a31);
  a33=(a22/a26);
  a33=(a33/a32);
  a33=(a15-a33);
  a31=(a31/a33);
  a34=(a29?a31:0);
  a28=(a28+a34);
  a34=(a18*a28);
  a35=(a10<a11);
  a11=(a11/a13);
  a11=(a11-a15);
  a11=(a26*a11);
  a11=exp(a11);
  a36=(a11-a15);
  a37=1.4741315910257660e+02;
  a36=(a36/a37);
  a36=(a35?a36:0);
  a34=(a34+a36);
  a36=1.0000000000000001e-01;
  a38=7.;
  a39=(a22/a38);
  a39=(a36*a39);
  a34=(a34+a39);
  a39=3.9024390243902418e-01;
  a40=(a39*a20);
  a41=(a34*a40);
  if (res[0]!=0) res[0][0]=a41;
  a41=-3.9024390243902396e-01;
  a42=(a41*a20);
  a43=(a34*a42);
  if (res[0]!=0) res[0][1]=a43;
  a43=arg[2]? arg[2][1] : 0;
  a44=5.0149234280221489e-01;
  a45=sin(a2);
  a46=(a44*a45);
  a47=5.0000000000000000e-01;
  a48=(a46+a47);
  a49=casadi_sq(a48);
  a50=cos(a2);
  a51=(a44*a50);
  a52=casadi_sq(a51);
  a49=(a49+a52);
  a49=sqrt(a49);
  a52=(a49-a10);
  a52=(a52/a12);
  a53=(a52/a13);
  a53=(a53-a15);
  a54=casadi_sq(a53);
  a54=(a54/a17);
  a54=(-a54);
  a54=exp(a54);
  a55=(a43*a54);
  a56=(a46+a47);
  a50=(a44*a50);
  a57=(a56*a50);
  a45=(a44*a45);
  a58=(a51*a45);
  a57=(a57-a58);
  a46=(a46+a47);
  a47=casadi_sq(a46);
  a58=casadi_sq(a51);
  a47=(a47+a58);
  a47=sqrt(a47);
  a57=(a57/a47);
  a58=(a57*a21);
  a59=(a58<=a23);
  a60=fabs(a58);
  a60=(a60/a26);
  a60=(a15-a60);
  a61=fabs(a58);
  a61=(a61/a26);
  a61=(a15+a61);
  a60=(a60/a61);
  a62=(a59?a60:0);
  a63=(!a59);
  a64=(a30*a58);
  a64=(a64/a26);
  a64=(a64/a32);
  a64=(a15-a64);
  a65=(a58/a26);
  a65=(a65/a32);
  a65=(a15-a65);
  a64=(a64/a65);
  a66=(a63?a64:0);
  a62=(a62+a66);
  a66=(a55*a62);
  a67=(a10<a52);
  a52=(a52/a13);
  a52=(a52-a15);
  a52=(a26*a52);
  a52=exp(a52);
  a68=(a52-a15);
  a68=(a68/a37);
  a68=(a67?a68:0);
  a66=(a66+a68);
  a68=(a58/a38);
  a68=(a36*a68);
  a66=(a66+a68);
  a68=(a39*a57);
  a69=(a66*a68);
  if (res[0]!=0) res[0][2]=a69;
  a69=(a41*a57);
  a70=(a66*a69);
  if (res[0]!=0) res[0][3]=a70;
  a70=arg[2]? arg[2][2] : 0;
  a71=5.0291081702804252e-01;
  a72=arg[0]? arg[0][5] : 0;
  a73=sin(a72);
  a74=sin(a2);
  a75=(a73*a74);
  a76=cos(a72);
  a77=cos(a2);
  a78=(a76*a77);
  a75=(a75-a78);
  a78=(a71*a75);
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
  a87=(a71*a86);
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
  a93=(a70*a92);
  a94=(a78+a82);
  a95=(a81*a77);
  a96=(a71*a86);
  a96=(a85-a96);
  a95=(a95-a96);
  a97=(a94*a95);
  a98=(a87-a88);
  a99=(a71*a75);
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
  a88=(a97*a21);
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
  a107=(a88<=a23);
  a108=fabs(a88);
  a108=(a108/a26);
  a108=(a15-a108);
  a109=fabs(a88);
  a109=(a109/a26);
  a109=(a15+a109);
  a108=(a108/a109);
  a110=(a107?a108:0);
  a111=(!a107);
  a112=(a30*a88);
  a112=(a112/a26);
  a112=(a112/a32);
  a112=(a15-a112);
  a113=(a88/a26);
  a113=(a113/a32);
  a113=(a15-a113);
  a112=(a112/a113);
  a114=(a111?a112:0);
  a110=(a110+a114);
  a114=(a93*a110);
  a115=(a10<a90);
  a90=(a90/a13);
  a90=(a90-a15);
  a90=(a26*a90);
  a90=exp(a90);
  a116=(a90-a15);
  a116=(a116/a37);
  a116=(a115?a116:0);
  a114=(a114+a116);
  a116=(a88/a38);
  a116=(a36*a116);
  a114=(a114+a116);
  a116=-3.9024390243902440e-01;
  a117=(a116*a96);
  a118=(a39*a97);
  a117=(a117+a118);
  a118=(a114*a117);
  if (res[0]!=0) res[0][4]=a118;
  a118=1.3902439024390245e+00;
  a119=(a118*a96);
  a120=(a41*a97);
  a119=(a119+a120);
  a120=(a114*a119);
  if (res[0]!=0) res[0][5]=a120;
  a120=arg[2]? arg[2][3] : 0;
  a121=5.0110447401701919e-01;
  a122=sin(a72);
  a123=sin(a2);
  a124=(a122*a123);
  a125=cos(a72);
  a126=cos(a2);
  a127=(a125*a126);
  a124=(a124-a127);
  a127=(a121*a124);
  a128=(a79*a123);
  a127=(a127-a128);
  a129=1.7500000000000000e+00;
  a130=(a129*a123);
  a131=(a127+a130);
  a132=casadi_sq(a131);
  a133=(a79*a126);
  a134=(a125*a123);
  a135=(a122*a126);
  a134=(a134+a135);
  a135=(a121*a134);
  a135=(a133-a135);
  a136=(a129*a126);
  a137=(a135-a136);
  a138=casadi_sq(a137);
  a132=(a132+a138);
  a132=sqrt(a132);
  a138=(a132-a10);
  a138=(a138/a12);
  a12=(a138/a13);
  a12=(a12-a15);
  a139=casadi_sq(a12);
  a139=(a139/a17);
  a139=(-a139);
  a139=exp(a139);
  a17=(a120*a139);
  a140=(a127+a130);
  a141=(a129*a126);
  a142=(a121*a134);
  a142=(a133-a142);
  a141=(a141-a142);
  a143=(a140*a141);
  a144=(a135-a136);
  a145=(a121*a124);
  a145=(a145-a128);
  a146=(a129*a123);
  a146=(a145+a146);
  a147=(a144*a146);
  a143=(a143+a147);
  a127=(a127+a130);
  a130=casadi_sq(a127);
  a135=(a135-a136);
  a136=casadi_sq(a135);
  a130=(a130+a136);
  a130=sqrt(a130);
  a143=(a143/a130);
  a136=(a143*a21);
  a147=(a122*a126);
  a148=(a125*a123);
  a147=(a147+a148);
  a148=(a124*a128);
  a149=(a134*a133);
  a148=(a148+a149);
  a149=(a147*a148);
  a150=(a147*a128);
  a151=(a125*a126);
  a152=(a122*a123);
  a151=(a151-a152);
  a152=(a151*a133);
  a150=(a150+a152);
  a152=(a124*a150);
  a149=(a149-a152);
  a149=(a149-a142);
  a142=(a140*a149);
  a152=(a134*a150);
  a153=(a151*a148);
  a152=(a152-a153);
  a152=(a152+a145);
  a145=(a144*a152);
  a142=(a142+a145);
  a142=(a142/a130);
  a145=(a142*a99);
  a136=(a136+a145);
  a23=(a136<=a23);
  a145=fabs(a136);
  a145=(a145/a26);
  a145=(a15-a145);
  a153=fabs(a136);
  a153=(a153/a26);
  a153=(a15+a153);
  a145=(a145/a153);
  a154=(a23?a145:0);
  a155=(!a23);
  a156=(a30*a136);
  a156=(a156/a26);
  a156=(a156/a32);
  a156=(a15-a156);
  a157=(a136/a26);
  a157=(a157/a32);
  a157=(a15-a157);
  a156=(a156/a157);
  a32=(a155?a156:0);
  a154=(a154+a32);
  a32=(a17*a154);
  a10=(a10<a138);
  a138=(a138/a13);
  a138=(a138-a15);
  a138=(a26*a138);
  a138=exp(a138);
  a13=(a138-a15);
  a13=(a13/a37);
  a13=(a10?a13:0);
  a32=(a32+a13);
  a38=(a136/a38);
  a38=(a36*a38);
  a32=(a32+a38);
  a38=(a116*a142);
  a13=(a39*a143);
  a38=(a38+a13);
  a13=(a32*a38);
  if (res[0]!=0) res[0][6]=a13;
  a13=(a118*a142);
  a37=(a41*a143);
  a13=(a13+a37);
  a37=(a32*a13);
  if (res[0]!=0) res[0][7]=a37;
  a37=cos(a2);
  a158=arg[0]? arg[0][3] : 0;
  a32=(a158*a32);
  a159=(a116*a32);
  a160=1.4285714285714285e-01;
  a38=(a158*a38);
  a161=(a36*a38);
  a161=(a160*a161);
  a162=-1.2121212121212121e+01;
  a163=(a17*a38);
  a156=(a156/a157);
  a164=(a163*a156);
  a164=(a162*a164);
  a164=(a36*a164);
  a164=(a155?a164:0);
  a161=(a161+a164);
  a164=(a163/a157);
  a164=(a162*a164);
  a164=(a36*a164);
  a164=(a30*a164);
  a164=(-a164);
  a164=(a155?a164:0);
  a161=(a161+a164);
  a145=(a145/a153);
  a164=(a163*a145);
  a164=(a36*a164);
  a165=casadi_sign(a136);
  a164=(a164*a165);
  a164=(-a164);
  a164=(a23?a164:0);
  a161=(a161+a164);
  a163=(a163/a153);
  a163=(a36*a163);
  a136=casadi_sign(a136);
  a163=(a163*a136);
  a163=(-a163);
  a163=(a23?a163:0);
  a161=(a161+a163);
  a163=(a99*a161);
  a159=(a159+a163);
  a163=(a159/a130);
  a164=(a144*a163);
  a166=(a134*a164);
  a167=(a140*a163);
  a168=(a124*a167);
  a166=(a166-a168);
  a168=(a128*a166);
  a169=(a148*a167);
  a168=(a168+a169);
  a169=(a125*a168);
  a170=(a133*a166);
  a171=(a148*a164);
  a170=(a170-a171);
  a171=(a122*a170);
  a169=(a169-a171);
  a171=(a39*a32);
  a172=(a21*a161);
  a171=(a171+a172);
  a172=(a171/a130);
  a173=(a144*a172);
  a174=(a129*a173);
  a169=(a169+a174);
  a174=(a150*a164);
  a175=(a147*a167);
  a176=(a151*a164);
  a175=(a175-a176);
  a176=(a133*a175);
  a174=(a174+a176);
  a137=(a137+a137);
  a176=1.1394939273245490e+00;
  a177=1.4285714285714286e+00;
  a178=6.7836549063042314e-03;
  a179=(a178*a38);
  a179=(a179*a138);
  a179=(a26*a179);
  a179=(a177*a179);
  a179=(a10?a179:0);
  a12=(a12+a12);
  a180=2.2222222222222223e+00;
  a38=(a154*a38);
  a38=(a120*a38);
  a38=(a139*a38);
  a38=(a180*a38);
  a38=(a12*a38);
  a38=(a177*a38);
  a179=(a179-a38);
  a179=(a176*a179);
  a132=(a132+a132);
  a179=(a179/a132);
  a38=(a137*a179);
  a135=(a135+a135);
  a181=(a142/a130);
  a159=(a181*a159);
  a182=(a143/a130);
  a171=(a182*a171);
  a159=(a159+a171);
  a171=(a130+a130);
  a159=(a159/a171);
  a183=(a135*a159);
  a184=(a38-a183);
  a185=(a152*a163);
  a186=(a146*a172);
  a185=(a185+a186);
  a184=(a184+a185);
  a186=(a121*a184);
  a174=(a174-a186);
  a186=(a140*a172);
  a187=(a167+a186);
  a188=(a121*a187);
  a174=(a174+a188);
  a188=(a125*a174);
  a169=(a169+a188);
  a131=(a131+a131);
  a179=(a131*a179);
  a127=(a127+a127);
  a159=(a127*a159);
  a188=(a179-a159);
  a163=(a149*a163);
  a172=(a141*a172);
  a163=(a163+a172);
  a188=(a188+a163);
  a188=(a129*a188);
  a169=(a169+a188);
  a188=(a147*a166);
  a172=(a124*a175);
  a188=(a188+a172);
  a164=(a164+a173);
  a188=(a188-a164);
  a179=(a179-a159);
  a179=(a179+a163);
  a188=(a188-a179);
  a188=(a79*a188);
  a169=(a169+a188);
  a188=(a128*a175);
  a167=(a150*a167);
  a188=(a188-a167);
  a164=(a121*a164);
  a188=(a188+a164);
  a179=(a121*a179);
  a188=(a188+a179);
  a179=(a122*a188);
  a169=(a169+a179);
  a169=(a37*a169);
  a179=cos(a2);
  a164=9.8100000000000005e+00;
  a167=cos(a72);
  a163=4.8780487804878025e-01;
  a159=(a163*a167);
  a173=(a167*a159);
  a172=sin(a72);
  a189=(a163*a172);
  a190=(a172*a189);
  a173=(a173+a190);
  a173=(a164*a173);
  a173=(a179*a173);
  a190=sin(a2);
  a191=(a167*a189);
  a192=(a172*a159);
  a191=(a191-a192);
  a191=(a164*a191);
  a191=(a190*a191);
  a173=(a173+a191);
  a191=sin(a2);
  a192=(a125*a170);
  a193=(a122*a168);
  a192=(a192+a193);
  a183=(a183-a38);
  a183=(a183-a185);
  a183=(a129*a183);
  a192=(a192+a183);
  a183=(a122*a174);
  a192=(a192+a183);
  a166=(a151*a166);
  a175=(a134*a175);
  a166=(a166+a175);
  a166=(a166+a184);
  a166=(a166-a187);
  a166=(a79*a166);
  a192=(a192+a166);
  a186=(a129*a186);
  a192=(a192+a186);
  a186=(a125*a188);
  a192=(a192-a186);
  a192=(a191*a192);
  a173=(a173+a192);
  a169=(a169-a173);
  a173=sin(a2);
  a192=arg[0]? arg[0][2] : 0;
  a114=(a192*a114);
  a116=(a116*a114);
  a117=(a192*a117);
  a186=(a36*a117);
  a186=(a160*a186);
  a166=(a93*a117);
  a112=(a112/a113);
  a187=(a166*a112);
  a187=(a162*a187);
  a187=(a36*a187);
  a187=(a111?a187:0);
  a186=(a186+a187);
  a187=(a166/a113);
  a187=(a162*a187);
  a187=(a36*a187);
  a187=(a30*a187);
  a187=(-a187);
  a187=(a111?a187:0);
  a186=(a186+a187);
  a108=(a108/a109);
  a187=(a166*a108);
  a187=(a36*a187);
  a184=casadi_sign(a88);
  a187=(a187*a184);
  a187=(-a187);
  a187=(a107?a187:0);
  a186=(a186+a187);
  a166=(a166/a109);
  a166=(a36*a166);
  a88=casadi_sign(a88);
  a166=(a166*a88);
  a166=(-a166);
  a166=(a107?a166:0);
  a186=(a186+a166);
  a166=(a99*a186);
  a116=(a116+a166);
  a166=(a116/a82);
  a187=(a98*a166);
  a175=(a86*a187);
  a183=(a94*a166);
  a185=(a75*a183);
  a175=(a175-a185);
  a185=(a85*a175);
  a38=(a102*a187);
  a185=(a185-a38);
  a38=(a76*a185);
  a193=(a80*a175);
  a194=(a102*a183);
  a193=(a193+a194);
  a194=(a73*a193);
  a38=(a38+a194);
  a87=(a87+a87);
  a194=(a96/a82);
  a116=(a194*a116);
  a195=(a97/a82);
  a196=(a39*a114);
  a197=(a21*a186);
  a196=(a196+a197);
  a197=(a195*a196);
  a116=(a116+a197);
  a197=(a82+a82);
  a116=(a116/a197);
  a198=(a87*a116);
  a89=(a89+a89);
  a199=(a178*a117);
  a199=(a199*a90);
  a199=(a26*a199);
  a199=(a177*a199);
  a199=(a115?a199:0);
  a91=(a91+a91);
  a117=(a110*a117);
  a117=(a70*a117);
  a117=(a92*a117);
  a117=(a180*a117);
  a117=(a91*a117);
  a117=(a177*a117);
  a199=(a199-a117);
  a199=(a176*a199);
  a84=(a84+a84);
  a199=(a199/a84);
  a117=(a89*a199);
  a200=(a198-a117);
  a201=(a106*a166);
  a196=(a196/a82);
  a202=(a100*a196);
  a201=(a201+a202);
  a200=(a200-a201);
  a200=(a81*a200);
  a38=(a38+a200);
  a200=(a104*a187);
  a202=(a101*a183);
  a203=(a105*a187);
  a202=(a202-a203);
  a203=(a85*a202);
  a200=(a200+a203);
  a117=(a117-a198);
  a117=(a117+a201);
  a201=(a71*a117);
  a200=(a200-a201);
  a201=(a94*a196);
  a198=(a183+a201);
  a203=(a71*a198);
  a200=(a200+a203);
  a203=(a73*a200);
  a38=(a38+a203);
  a203=(a105*a175);
  a204=(a86*a202);
  a203=(a203+a204);
  a203=(a203+a117);
  a203=(a203-a198);
  a203=(a79*a203);
  a38=(a38+a203);
  a201=(a81*a201);
  a38=(a38+a201);
  a201=(a80*a202);
  a183=(a104*a183);
  a201=(a201-a183);
  a183=(a98*a196);
  a187=(a187+a183);
  a203=(a71*a187);
  a201=(a201+a203);
  a83=(a83+a83);
  a199=(a83*a199);
  a78=(a78+a78);
  a116=(a78*a116);
  a203=(a199-a116);
  a166=(a103*a166);
  a196=(a95*a196);
  a166=(a166+a196);
  a203=(a203+a166);
  a196=(a71*a203);
  a201=(a201+a196);
  a196=(a76*a201);
  a38=(a38-a196);
  a38=(a173*a38);
  a169=(a169-a38);
  a38=cos(a2);
  a196=(a76*a193);
  a198=(a73*a185);
  a196=(a196-a198);
  a183=(a81*a183);
  a196=(a196+a183);
  a183=(a76*a200);
  a196=(a196+a183);
  a199=(a199-a116);
  a199=(a199+a166);
  a199=(a81*a199);
  a196=(a196+a199);
  a175=(a101*a175);
  a202=(a75*a202);
  a175=(a175+a202);
  a175=(a175-a187);
  a175=(a175-a203);
  a175=(a79*a175);
  a196=(a196+a175);
  a175=(a73*a201);
  a196=(a196+a175);
  a196=(a38*a196);
  a169=(a169+a196);
  a196=sin(a2);
  a175=(a51+a51);
  a203=arg[0]? arg[0][1] : 0;
  a68=(a203*a68);
  a187=(a178*a68);
  a187=(a187*a52);
  a187=(a26*a187);
  a187=(a177*a187);
  a187=(a67?a187:0);
  a53=(a53+a53);
  a202=(a62*a68);
  a202=(a43*a202);
  a202=(a54*a202);
  a202=(a180*a202);
  a202=(a53*a202);
  a202=(a177*a202);
  a187=(a187-a202);
  a187=(a176*a187);
  a49=(a49+a49);
  a187=(a187/a49);
  a202=(a175*a187);
  a199=(a51+a51);
  a166=(a57/a47);
  a66=(a203*a66);
  a116=(a39*a66);
  a183=(a36*a68);
  a183=(a160*a183);
  a68=(a55*a68);
  a64=(a64/a65);
  a198=(a68*a64);
  a198=(a162*a198);
  a198=(a36*a198);
  a198=(a63?a198:0);
  a183=(a183+a198);
  a198=(a68/a65);
  a198=(a162*a198);
  a198=(a36*a198);
  a198=(a30*a198);
  a198=(-a198);
  a198=(a63?a198:0);
  a183=(a183+a198);
  a60=(a60/a61);
  a198=(a68*a60);
  a198=(a36*a198);
  a117=casadi_sign(a58);
  a198=(a198*a117);
  a198=(-a198);
  a198=(a59?a198:0);
  a183=(a183+a198);
  a68=(a68/a61);
  a68=(a36*a68);
  a58=casadi_sign(a58);
  a68=(a68*a58);
  a68=(-a68);
  a68=(a59?a68:0);
  a183=(a183+a68);
  a68=(a21*a183);
  a116=(a116+a68);
  a68=(a166*a116);
  a198=(a47+a47);
  a68=(a68/a198);
  a204=(a199*a68);
  a202=(a202-a204);
  a116=(a116/a47);
  a204=(a45*a116);
  a202=(a202-a204);
  a202=(a44*a202);
  a204=(a56*a116);
  a204=(a44*a204);
  a202=(a202+a204);
  a202=(a196*a202);
  a169=(a169-a202);
  a202=cos(a2);
  a48=(a48+a48);
  a187=(a48*a187);
  a46=(a46+a46);
  a68=(a46*a68);
  a187=(a187-a68);
  a68=(a50*a116);
  a187=(a187+a68);
  a187=(a44*a187);
  a116=(a51*a116);
  a116=(a44*a116);
  a187=(a187-a116);
  a187=(a202*a187);
  a169=(a169+a187);
  a187=sin(a2);
  a116=(a9+a9);
  a68=arg[0]? arg[0][0] : 0;
  a40=(a68*a40);
  a204=(a178*a40);
  a204=(a204*a11);
  a204=(a26*a204);
  a204=(a177*a204);
  a204=(a35?a204:0);
  a14=(a14+a14);
  a205=(a28*a40);
  a205=(a0*a205);
  a205=(a16*a205);
  a205=(a180*a205);
  a205=(a14*a205);
  a205=(a177*a205);
  a204=(a204-a205);
  a204=(a176*a204);
  a7=(a7+a7);
  a204=(a204/a7);
  a205=(a116*a204);
  a206=(a9+a9);
  a207=(a20/a5);
  a34=(a68*a34);
  a39=(a39*a34);
  a208=(a36*a40);
  a208=(a160*a208);
  a40=(a18*a40);
  a31=(a31/a33);
  a209=(a40*a31);
  a209=(a162*a209);
  a209=(a36*a209);
  a209=(a29?a209:0);
  a208=(a208+a209);
  a209=(a40/a33);
  a209=(a162*a209);
  a209=(a36*a209);
  a209=(a30*a209);
  a209=(-a209);
  a209=(a29?a209:0);
  a208=(a208+a209);
  a25=(a25/a27);
  a209=(a40*a25);
  a209=(a36*a209);
  a210=casadi_sign(a22);
  a209=(a209*a210);
  a209=(-a209);
  a209=(a24?a209:0);
  a208=(a208+a209);
  a40=(a40/a27);
  a40=(a36*a40);
  a22=casadi_sign(a22);
  a40=(a40*a22);
  a40=(-a40);
  a40=(a24?a40:0);
  a208=(a208+a40);
  a40=(a21*a208);
  a39=(a39+a40);
  a40=(a207*a39);
  a209=(a5+a5);
  a40=(a40/a209);
  a211=(a206*a40);
  a205=(a205-a211);
  a39=(a39/a5);
  a211=(a3*a39);
  a205=(a205-a211);
  a205=(a1*a205);
  a211=(a19*a39);
  a211=(a1*a211);
  a205=(a205+a211);
  a205=(a187*a205);
  a169=(a169-a205);
  a205=cos(a2);
  a6=(a6+a6);
  a204=(a6*a204);
  a4=(a4+a4);
  a40=(a4*a40);
  a204=(a204-a40);
  a40=(a8*a39);
  a204=(a204+a40);
  a204=(a1*a204);
  a39=(a9*a39);
  a39=(a1*a39);
  a204=(a204-a39);
  a204=(a205*a204);
  a169=(a169+a204);
  if (res[0]!=0) res[0][8]=a169;
  a169=(a118*a32);
  a13=(a158*a13);
  a204=(a36*a13);
  a204=(a160*a204);
  a17=(a17*a13);
  a156=(a17*a156);
  a156=(a162*a156);
  a156=(a36*a156);
  a156=(a155?a156:0);
  a204=(a204+a156);
  a157=(a17/a157);
  a157=(a162*a157);
  a157=(a36*a157);
  a157=(a30*a157);
  a157=(-a157);
  a155=(a155?a157:0);
  a204=(a204+a155);
  a145=(a17*a145);
  a145=(a36*a145);
  a145=(a145*a165);
  a145=(-a145);
  a145=(a23?a145:0);
  a204=(a204+a145);
  a17=(a17/a153);
  a17=(a36*a17);
  a17=(a17*a136);
  a17=(-a17);
  a23=(a23?a17:0);
  a204=(a204+a23);
  a23=(a99*a204);
  a169=(a169+a23);
  a23=(a169/a130);
  a17=(a144*a23);
  a136=(a134*a17);
  a153=(a140*a23);
  a145=(a124*a153);
  a136=(a136-a145);
  a145=(a128*a136);
  a165=(a148*a153);
  a145=(a145+a165);
  a165=(a125*a145);
  a155=(a133*a136);
  a148=(a148*a17);
  a155=(a155-a148);
  a148=(a122*a155);
  a165=(a165-a148);
  a32=(a41*a32);
  a148=(a21*a204);
  a32=(a32+a148);
  a130=(a32/a130);
  a144=(a144*a130);
  a148=(a129*a144);
  a165=(a165+a148);
  a148=(a150*a17);
  a157=(a147*a153);
  a156=(a151*a17);
  a157=(a157-a156);
  a133=(a133*a157);
  a148=(a148+a133);
  a133=(a178*a13);
  a133=(a133*a138);
  a133=(a26*a133);
  a133=(a177*a133);
  a10=(a10?a133:0);
  a13=(a154*a13);
  a120=(a120*a13);
  a120=(a139*a120);
  a120=(a180*a120);
  a12=(a12*a120);
  a12=(a177*a12);
  a10=(a10-a12);
  a10=(a176*a10);
  a10=(a10/a132);
  a137=(a137*a10);
  a181=(a181*a169);
  a182=(a182*a32);
  a181=(a181+a182);
  a181=(a181/a171);
  a135=(a135*a181);
  a171=(a137-a135);
  a152=(a152*a23);
  a146=(a146*a130);
  a152=(a152+a146);
  a171=(a171+a152);
  a146=(a121*a171);
  a148=(a148-a146);
  a140=(a140*a130);
  a146=(a153+a140);
  a182=(a121*a146);
  a148=(a148+a182);
  a182=(a125*a148);
  a165=(a165+a182);
  a131=(a131*a10);
  a127=(a127*a181);
  a181=(a131-a127);
  a149=(a149*a23);
  a141=(a141*a130);
  a149=(a149+a141);
  a181=(a181+a149);
  a181=(a129*a181);
  a165=(a165+a181);
  a147=(a147*a136);
  a124=(a124*a157);
  a147=(a147+a124);
  a17=(a17+a144);
  a147=(a147-a17);
  a131=(a131-a127);
  a131=(a131+a149);
  a147=(a147-a131);
  a147=(a79*a147);
  a165=(a165+a147);
  a128=(a128*a157);
  a150=(a150*a153);
  a128=(a128-a150);
  a17=(a121*a17);
  a128=(a128+a17);
  a121=(a121*a131);
  a128=(a128+a121);
  a121=(a122*a128);
  a165=(a165+a121);
  a37=(a37*a165);
  a165=-4.8780487804877992e-01;
  a121=(a165*a167);
  a131=(a167*a121);
  a17=(a165*a172);
  a150=(a172*a17);
  a131=(a131+a150);
  a131=(a164*a131);
  a179=(a179*a131);
  a131=(a167*a17);
  a150=(a172*a121);
  a131=(a131-a150);
  a131=(a164*a131);
  a190=(a190*a131);
  a179=(a179+a190);
  a190=(a125*a155);
  a131=(a122*a145);
  a190=(a190+a131);
  a135=(a135-a137);
  a135=(a135-a152);
  a135=(a129*a135);
  a190=(a190+a135);
  a122=(a122*a148);
  a190=(a190+a122);
  a151=(a151*a136);
  a134=(a134*a157);
  a151=(a151+a134);
  a151=(a151+a171);
  a151=(a151-a146);
  a151=(a79*a151);
  a190=(a190+a151);
  a129=(a129*a140);
  a190=(a190+a129);
  a125=(a125*a128);
  a190=(a190-a125);
  a191=(a191*a190);
  a179=(a179+a191);
  a37=(a37-a179);
  a118=(a118*a114);
  a119=(a192*a119);
  a179=(a36*a119);
  a179=(a160*a179);
  a93=(a93*a119);
  a112=(a93*a112);
  a112=(a162*a112);
  a112=(a36*a112);
  a112=(a111?a112:0);
  a179=(a179+a112);
  a113=(a93/a113);
  a113=(a162*a113);
  a113=(a36*a113);
  a113=(a30*a113);
  a113=(-a113);
  a111=(a111?a113:0);
  a179=(a179+a111);
  a108=(a93*a108);
  a108=(a36*a108);
  a108=(a108*a184);
  a108=(-a108);
  a108=(a107?a108:0);
  a179=(a179+a108);
  a93=(a93/a109);
  a93=(a36*a93);
  a93=(a93*a88);
  a93=(-a93);
  a107=(a107?a93:0);
  a179=(a179+a107);
  a107=(a99*a179);
  a118=(a118+a107);
  a107=(a118/a82);
  a93=(a98*a107);
  a88=(a86*a93);
  a109=(a94*a107);
  a108=(a75*a109);
  a88=(a88-a108);
  a108=(a85*a88);
  a184=(a102*a93);
  a108=(a108-a184);
  a184=(a76*a108);
  a111=(a80*a88);
  a102=(a102*a109);
  a111=(a111+a102);
  a102=(a73*a111);
  a184=(a184+a102);
  a194=(a194*a118);
  a114=(a41*a114);
  a118=(a21*a179);
  a114=(a114+a118);
  a195=(a195*a114);
  a194=(a194+a195);
  a194=(a194/a197);
  a87=(a87*a194);
  a197=(a178*a119);
  a197=(a197*a90);
  a197=(a26*a197);
  a197=(a177*a197);
  a115=(a115?a197:0);
  a119=(a110*a119);
  a70=(a70*a119);
  a70=(a92*a70);
  a70=(a180*a70);
  a91=(a91*a70);
  a91=(a177*a91);
  a115=(a115-a91);
  a115=(a176*a115);
  a115=(a115/a84);
  a89=(a89*a115);
  a84=(a87-a89);
  a106=(a106*a107);
  a114=(a114/a82);
  a100=(a100*a114);
  a106=(a106+a100);
  a84=(a84-a106);
  a84=(a81*a84);
  a184=(a184+a84);
  a84=(a104*a93);
  a100=(a101*a109);
  a82=(a105*a93);
  a100=(a100-a82);
  a85=(a85*a100);
  a84=(a84+a85);
  a89=(a89-a87);
  a89=(a89+a106);
  a106=(a71*a89);
  a84=(a84-a106);
  a94=(a94*a114);
  a106=(a109+a94);
  a87=(a71*a106);
  a84=(a84+a87);
  a87=(a73*a84);
  a184=(a184+a87);
  a105=(a105*a88);
  a86=(a86*a100);
  a105=(a105+a86);
  a105=(a105+a89);
  a105=(a105-a106);
  a105=(a79*a105);
  a184=(a184+a105);
  a94=(a81*a94);
  a184=(a184+a94);
  a80=(a80*a100);
  a104=(a104*a109);
  a80=(a80-a104);
  a98=(a98*a114);
  a93=(a93+a98);
  a104=(a71*a93);
  a80=(a80+a104);
  a83=(a83*a115);
  a78=(a78*a194);
  a194=(a83-a78);
  a103=(a103*a107);
  a95=(a95*a114);
  a103=(a103+a95);
  a194=(a194+a103);
  a71=(a71*a194);
  a80=(a80+a71);
  a71=(a76*a80);
  a184=(a184-a71);
  a173=(a173*a184);
  a37=(a37-a173);
  a173=(a76*a111);
  a184=(a73*a108);
  a173=(a173-a184);
  a98=(a81*a98);
  a173=(a173+a98);
  a76=(a76*a84);
  a173=(a173+a76);
  a83=(a83-a78);
  a83=(a83+a103);
  a81=(a81*a83);
  a173=(a173+a81);
  a101=(a101*a88);
  a75=(a75*a100);
  a101=(a101+a75);
  a101=(a101-a93);
  a101=(a101-a194);
  a101=(a79*a101);
  a173=(a173+a101);
  a73=(a73*a80);
  a173=(a173+a73);
  a38=(a38*a173);
  a37=(a37+a38);
  a69=(a203*a69);
  a38=(a178*a69);
  a38=(a38*a52);
  a38=(a26*a38);
  a38=(a177*a38);
  a67=(a67?a38:0);
  a38=(a62*a69);
  a43=(a43*a38);
  a43=(a54*a43);
  a43=(a180*a43);
  a53=(a53*a43);
  a53=(a177*a53);
  a67=(a67-a53);
  a67=(a176*a67);
  a67=(a67/a49);
  a175=(a175*a67);
  a66=(a41*a66);
  a49=(a36*a69);
  a49=(a160*a49);
  a55=(a55*a69);
  a64=(a55*a64);
  a64=(a162*a64);
  a64=(a36*a64);
  a64=(a63?a64:0);
  a49=(a49+a64);
  a65=(a55/a65);
  a65=(a162*a65);
  a65=(a36*a65);
  a65=(a30*a65);
  a65=(-a65);
  a63=(a63?a65:0);
  a49=(a49+a63);
  a60=(a55*a60);
  a60=(a36*a60);
  a60=(a60*a117);
  a60=(-a60);
  a60=(a59?a60:0);
  a49=(a49+a60);
  a55=(a55/a61);
  a55=(a36*a55);
  a55=(a55*a58);
  a55=(-a55);
  a59=(a59?a55:0);
  a49=(a49+a59);
  a59=(a21*a49);
  a66=(a66+a59);
  a166=(a166*a66);
  a166=(a166/a198);
  a199=(a199*a166);
  a175=(a175-a199);
  a66=(a66/a47);
  a45=(a45*a66);
  a175=(a175-a45);
  a175=(a44*a175);
  a56=(a56*a66);
  a56=(a44*a56);
  a175=(a175+a56);
  a196=(a196*a175);
  a37=(a37-a196);
  a48=(a48*a67);
  a46=(a46*a166);
  a48=(a48-a46);
  a50=(a50*a66);
  a48=(a48+a50);
  a48=(a44*a48);
  a51=(a51*a66);
  a44=(a44*a51);
  a48=(a48-a44);
  a202=(a202*a48);
  a37=(a37+a202);
  a42=(a68*a42);
  a178=(a178*a42);
  a178=(a178*a11);
  a26=(a26*a178);
  a26=(a177*a26);
  a35=(a35?a26:0);
  a26=(a28*a42);
  a0=(a0*a26);
  a0=(a16*a0);
  a180=(a180*a0);
  a14=(a14*a180);
  a177=(a177*a14);
  a35=(a35-a177);
  a176=(a176*a35);
  a176=(a176/a7);
  a116=(a116*a176);
  a41=(a41*a34);
  a34=(a36*a42);
  a160=(a160*a34);
  a18=(a18*a42);
  a31=(a18*a31);
  a31=(a162*a31);
  a31=(a36*a31);
  a31=(a29?a31:0);
  a160=(a160+a31);
  a33=(a18/a33);
  a162=(a162*a33);
  a162=(a36*a162);
  a30=(a30*a162);
  a30=(-a30);
  a29=(a29?a30:0);
  a160=(a160+a29);
  a25=(a18*a25);
  a25=(a36*a25);
  a25=(a25*a210);
  a25=(-a25);
  a25=(a24?a25:0);
  a160=(a160+a25);
  a18=(a18/a27);
  a36=(a36*a18);
  a36=(a36*a22);
  a36=(-a36);
  a24=(a24?a36:0);
  a160=(a160+a24);
  a24=(a21*a160);
  a41=(a41+a24);
  a207=(a207*a41);
  a207=(a207/a209);
  a206=(a206*a207);
  a116=(a116-a206);
  a41=(a41/a5);
  a3=(a3*a41);
  a116=(a116-a3);
  a116=(a1*a116);
  a19=(a19*a41);
  a19=(a1*a19);
  a116=(a116+a19);
  a187=(a187*a116);
  a37=(a37-a187);
  a6=(a6*a176);
  a4=(a4*a207);
  a6=(a6-a4);
  a8=(a8*a41);
  a6=(a6+a8);
  a6=(a1*a6);
  a9=(a9*a41);
  a1=(a1*a9);
  a6=(a6-a1);
  a205=(a205*a6);
  a37=(a37+a205);
  if (res[0]!=0) res[0][9]=a37;
  a37=cos(a72);
  a205=(a79*a21);
  a21=(a21+a99);
  a6=(a21*a159);
  a1=(a99*a159);
  a6=(a6-a1);
  a1=(a205*a6);
  a9=cos(a2);
  a9=(a164*a9);
  a41=(a9*a159);
  a1=(a1-a41);
  a41=(a167*a9);
  a2=sin(a2);
  a164=(a164*a2);
  a2=(a172*a164);
  a41=(a41-a2);
  a2=(a167*a205);
  a8=(a2*a99);
  a41=(a41+a8);
  a8=(a21*a2);
  a41=(a41-a8);
  a8=(a163*a41);
  a1=(a1+a8);
  a8=(a164*a189);
  a1=(a1-a8);
  a1=(a37*a1);
  a8=sin(a72);
  a4=(a172*a205);
  a207=(a21*a4);
  a176=(a167*a164);
  a187=(a172*a9);
  a176=(a176+a187);
  a187=(a4*a99);
  a176=(a176+a187);
  a207=(a207-a176);
  a163=(a163*a207);
  a176=(a164*a159);
  a163=(a163-a176);
  a176=(a99*a189);
  a187=(a21*a189);
  a176=(a176-a187);
  a187=(a205*a176);
  a163=(a163+a187);
  a187=(a9*a189);
  a163=(a163+a187);
  a163=(a8*a163);
  a1=(a1-a163);
  a163=sin(a72);
  a187=(a126*a170);
  a116=(a123*a168);
  a187=(a187+a116);
  a116=(a123*a174);
  a187=(a187+a116);
  a116=(a126*a188);
  a187=(a187-a116);
  a187=(a163*a187);
  a1=(a1-a187);
  a187=cos(a72);
  a168=(a126*a168);
  a170=(a123*a170);
  a168=(a168-a170);
  a174=(a126*a174);
  a168=(a168+a174);
  a188=(a123*a188);
  a168=(a168+a188);
  a168=(a187*a168);
  a1=(a1+a168);
  a168=sin(a72);
  a188=(a77*a185);
  a174=(a74*a193);
  a188=(a188+a174);
  a174=(a74*a200);
  a188=(a188+a174);
  a174=(a77*a201);
  a188=(a188-a174);
  a188=(a168*a188);
  a1=(a1-a188);
  a72=cos(a72);
  a193=(a77*a193);
  a185=(a74*a185);
  a193=(a193-a185);
  a200=(a77*a200);
  a193=(a193+a200);
  a201=(a74*a201);
  a193=(a193+a201);
  a193=(a72*a193);
  a1=(a1+a193);
  if (res[0]!=0) res[0][10]=a1;
  a1=(a21*a121);
  a193=(a99*a121);
  a1=(a1-a193);
  a193=(a205*a1);
  a201=(a9*a121);
  a193=(a193-a201);
  a41=(a165*a41);
  a193=(a193+a41);
  a41=(a164*a17);
  a193=(a193-a41);
  a37=(a37*a193);
  a165=(a165*a207);
  a164=(a164*a121);
  a165=(a165-a164);
  a99=(a99*a17);
  a21=(a21*a17);
  a99=(a99-a21);
  a205=(a205*a99);
  a165=(a165+a205);
  a9=(a9*a17);
  a165=(a165+a9);
  a8=(a8*a165);
  a37=(a37-a8);
  a8=(a126*a155);
  a165=(a123*a145);
  a8=(a8+a165);
  a165=(a123*a148);
  a8=(a8+a165);
  a165=(a126*a128);
  a8=(a8-a165);
  a163=(a163*a8);
  a37=(a37-a163);
  a145=(a126*a145);
  a155=(a123*a155);
  a145=(a145-a155);
  a126=(a126*a148);
  a145=(a145+a126);
  a123=(a123*a128);
  a145=(a145+a123);
  a187=(a187*a145);
  a37=(a37+a187);
  a187=(a77*a108);
  a145=(a74*a111);
  a187=(a187+a145);
  a145=(a74*a84);
  a187=(a187+a145);
  a145=(a77*a80);
  a187=(a187-a145);
  a168=(a168*a187);
  a37=(a37-a168);
  a111=(a77*a111);
  a108=(a74*a108);
  a111=(a111-a108);
  a77=(a77*a84);
  a111=(a111+a77);
  a74=(a74*a80);
  a111=(a111+a74);
  a72=(a72*a111);
  a37=(a37+a72);
  if (res[0]!=0) res[0][11]=a37;
  a37=-1.;
  if (res[0]!=0) res[0][12]=a37;
  a72=(a4*a159);
  a111=(a2*a189);
  a72=(a72-a111);
  a6=(a172*a6);
  a176=(a167*a176);
  a6=(a6+a176);
  a6=(a79*a6);
  a6=(a72+a6);
  a176=(a143*a161);
  a6=(a6+a176);
  a176=(a97*a186);
  a6=(a6+a176);
  a183=(a57*a183);
  a6=(a6+a183);
  a208=(a20*a208);
  a6=(a6+a208);
  if (res[0]!=0) res[0][13]=a6;
  a6=(a4*a121);
  a208=(a2*a17);
  a6=(a6-a208);
  a172=(a172*a1);
  a167=(a167*a99);
  a172=(a172+a167);
  a79=(a79*a172);
  a79=(a6+a79);
  a172=(a143*a204);
  a79=(a79+a172);
  a172=(a97*a179);
  a79=(a79+a172);
  a49=(a57*a49);
  a79=(a79+a49);
  a160=(a20*a160);
  a79=(a79+a160);
  if (res[0]!=0) res[0][14]=a79;
  if (res[0]!=0) res[0][15]=a37;
  a159=(a4*a159);
  a72=(a72-a159);
  a189=(a2*a189);
  a72=(a72+a189);
  a161=(a142*a161);
  a72=(a72+a161);
  a186=(a96*a186);
  a72=(a72+a186);
  if (res[0]!=0) res[0][16]=a72;
  a4=(a4*a121);
  a6=(a6-a4);
  a2=(a2*a17);
  a6=(a6+a2);
  a204=(a142*a204);
  a6=(a6+a204);
  a179=(a96*a179);
  a6=(a6+a179);
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
  a179=1.4439765966454325e+00;
  a204=-2.7025639012821762e-01;
  a28=(a28*a16);
  a68=(a68*a28);
  a20=(a20*a68);
  a68=(a204*a20);
  a68=(a179*a68);
  a28=(a6*a68);
  a16=9.6278838983177628e-01;
  a20=(a16*a20);
  a28=(a28-a20);
  a28=(a15*a28);
  a28=(-a28);
  if (res[2]!=0) res[2][0]=a28;
  if (res[2]!=0) res[2][1]=a68;
  a62=(a62*a54);
  a203=(a203*a62);
  a57=(a57*a203);
  a203=(a204*a57);
  a203=(a179*a203);
  a62=(a6*a203);
  a57=(a16*a57);
  a62=(a62-a57);
  a62=(a15*a62);
  a62=(-a62);
  if (res[2]!=0) res[2][2]=a62;
  if (res[2]!=0) res[2][3]=a203;
  a110=(a110*a92);
  a192=(a192*a110);
  a97=(a97*a192);
  a110=(a204*a97);
  a92=9.6278838983177639e-01;
  a96=(a96*a192);
  a192=(a92*a96);
  a110=(a110+a192);
  a110=(a179*a110);
  a192=(a6*a110);
  a97=(a16*a97);
  a96=(a15*a96);
  a97=(a97+a96);
  a192=(a192-a97);
  a192=(a15*a192);
  a192=(-a192);
  if (res[2]!=0) res[2][4]=a192;
  if (res[2]!=0) res[2][5]=a110;
  a154=(a154*a139);
  a158=(a158*a154);
  a143=(a143*a158);
  a204=(a204*a143);
  a142=(a142*a158);
  a92=(a92*a142);
  a204=(a204+a92);
  a179=(a179*a204);
  a6=(a6*a179);
  a16=(a16*a143);
  a142=(a15*a142);
  a16=(a16+a142);
  a6=(a6-a16);
  a15=(a15*a6);
  a15=(-a15);
  if (res[2]!=0) res[2][6]=a15;
  if (res[2]!=0) res[2][7]=a179;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_04_09_15211324_impl_dae_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
