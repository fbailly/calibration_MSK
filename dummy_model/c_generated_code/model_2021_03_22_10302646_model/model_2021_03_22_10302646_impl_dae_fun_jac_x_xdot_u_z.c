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
  #define CASADI_PREFIX(ID) model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_ ## ID
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

/* model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x4,8nz],o4[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a215, a216, a217, a218, a219, a22, a220, a221, a222, a223, a224, a225, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a22=4.0000006419552925e-01;
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
  a22=(a20-a24);
  a42=1.4741315910257660e+02;
  a22=(a22/a42);
  a22=(a41?a22:0);
  a40=(a40+a22);
  a22=1.0000000000000001e-01;
  a43=4.0000006419552925e+00;
  a43=(a28/a43);
  a43=(a22*a43);
  a40=(a40+a43);
  a43=(a17*a40);
  a44=(a13*a43);
  a45=sin(a6);
  a46=(a5*a45);
  a47=(a46+a5);
  a48=cos(a6);
  a49=(a5*a48);
  a50=(a47*a49);
  a51=(a5*a48);
  a45=(a5*a45);
  a52=(a51*a45);
  a50=(a50-a52);
  a52=(a46+a5);
  a53=casadi_sq(a52);
  a54=casadi_sq(a51);
  a53=(a53+a54);
  a53=sqrt(a53);
  a50=(a50/a53);
  a54=arg[0]? arg[0][1] : 0;
  a55=arg[2]? arg[2][1] : 0;
  a46=(a46+a5);
  a56=casadi_sq(a46);
  a57=casadi_sq(a51);
  a56=(a56+a57);
  a56=sqrt(a56);
  a57=(a56-a19);
  a57=(a57/a21);
  a58=5.2936696467126987e-01;
  a59=(a57/a58);
  a59=(a59-a24);
  a60=casadi_sq(a59);
  a60=(a60/a26);
  a60=(-a60);
  a60=exp(a60);
  a61=(a55*a60);
  a62=(a50*a1);
  a63=(a62<=a29);
  a64=fabs(a62);
  a64=(a64/a32);
  a64=(a24-a64);
  a65=fabs(a62);
  a65=(a65/a32);
  a65=(a24+a65);
  a64=(a64/a65);
  a66=(a63?a64:0);
  a67=(!a63);
  a68=(a36*a62);
  a68=(a68/a32);
  a68=(a68/a38);
  a68=(a24-a68);
  a69=(a62/a32);
  a69=(a69/a38);
  a69=(a24-a69);
  a68=(a68/a69);
  a70=(a67?a68:0);
  a66=(a66+a70);
  a70=(a61*a66);
  a71=(a19<a57);
  a57=(a57/a58);
  a57=(a57-a24);
  a57=(a32*a57);
  a57=exp(a57);
  a58=(a57-a24);
  a58=(a58/a42);
  a58=(a71?a58:0);
  a70=(a70+a58);
  a58=5.2936696467126989e+00;
  a58=(a62/a58);
  a58=(a22*a58);
  a70=(a70+a58);
  a58=(a54*a70);
  a72=(a50*a58);
  a44=(a44+a72);
  a72=arg[0]? arg[0][5] : 0;
  a73=sin(a72);
  a74=sin(a6);
  a75=(a73*a74);
  a76=cos(a72);
  a77=cos(a6);
  a78=(a76*a77);
  a75=(a75-a78);
  a78=(a5*a75);
  a79=1.2500000000000000e+00;
  a80=(a79*a74);
  a78=(a78-a80);
  a81=7.5000000000000000e-01;
  a82=(a81*a74);
  a83=(a78+a82);
  a84=(a81*a77);
  a85=(a79*a77);
  a86=(a76*a74);
  a87=(a73*a77);
  a86=(a86+a87);
  a87=(a5*a86);
  a87=(a85-a87);
  a84=(a84-a87);
  a88=(a83*a84);
  a89=(a5*a86);
  a89=(a85-a89);
  a90=(a81*a77);
  a91=(a89-a90);
  a92=(a5*a75);
  a92=(a92-a80);
  a93=(a81*a74);
  a93=(a92+a93);
  a94=(a91*a93);
  a88=(a88+a94);
  a94=(a78+a82);
  a95=casadi_sq(a94);
  a96=(a89-a90);
  a97=casadi_sq(a96);
  a95=(a95+a97);
  a95=sqrt(a95);
  a88=(a88/a95);
  a97=arg[0]? arg[0][2] : 0;
  a98=arg[2]? arg[2][2] : 0;
  a78=(a78+a82);
  a82=casadi_sq(a78);
  a89=(a89-a90);
  a90=casadi_sq(a89);
  a82=(a82+a90);
  a82=sqrt(a82);
  a90=(a82-a19);
  a90=(a90/a21);
  a99=4.0000003306580578e-01;
  a100=(a90/a99);
  a100=(a100-a24);
  a101=casadi_sq(a100);
  a101=(a101/a26);
  a101=(-a101);
  a101=exp(a101);
  a102=(a98*a101);
  a103=(a88*a1);
  a104=(a73*a77);
  a105=(a76*a74);
  a104=(a104+a105);
  a105=(a75*a80);
  a106=(a86*a85);
  a105=(a105+a106);
  a106=(a104*a105);
  a107=(a104*a80);
  a108=(a76*a77);
  a109=(a73*a74);
  a108=(a108-a109);
  a109=(a108*a85);
  a107=(a107+a109);
  a109=(a75*a107);
  a106=(a106-a109);
  a106=(a106-a87);
  a87=(a83*a106);
  a109=(a86*a107);
  a110=(a108*a105);
  a109=(a109-a110);
  a109=(a109+a92);
  a92=(a91*a109);
  a87=(a87+a92);
  a87=(a87/a95);
  a92=(a87*a2);
  a103=(a103+a92);
  a92=(a103<=a29);
  a110=fabs(a103);
  a110=(a110/a32);
  a110=(a24-a110);
  a111=fabs(a103);
  a111=(a111/a32);
  a111=(a24+a111);
  a110=(a110/a111);
  a112=(a92?a110:0);
  a113=(!a92);
  a114=(a36*a103);
  a114=(a114/a32);
  a114=(a114/a38);
  a114=(a24-a114);
  a115=(a103/a32);
  a115=(a115/a38);
  a115=(a24-a115);
  a114=(a114/a115);
  a116=(a113?a114:0);
  a112=(a112+a116);
  a116=(a102*a112);
  a117=(a19<a90);
  a90=(a90/a99);
  a90=(a90-a24);
  a90=(a32*a90);
  a90=exp(a90);
  a99=(a90-a24);
  a99=(a99/a42);
  a99=(a117?a99:0);
  a116=(a116+a99);
  a99=4.0000003306580574e+00;
  a99=(a103/a99);
  a99=(a22*a99);
  a116=(a116+a99);
  a99=(a97*a116);
  a118=(a88*a99);
  a44=(a44+a118);
  a118=sin(a72);
  a119=sin(a6);
  a120=(a118*a119);
  a121=cos(a72);
  a122=cos(a6);
  a123=(a121*a122);
  a120=(a120-a123);
  a123=(a5*a120);
  a124=(a79*a119);
  a123=(a123-a124);
  a125=1.7500000000000000e+00;
  a126=(a125*a119);
  a127=(a123+a126);
  a128=(a125*a122);
  a129=(a79*a122);
  a130=(a121*a119);
  a131=(a118*a122);
  a130=(a130+a131);
  a131=(a5*a130);
  a131=(a129-a131);
  a128=(a128-a131);
  a132=(a127*a128);
  a133=(a5*a130);
  a133=(a129-a133);
  a134=(a125*a122);
  a135=(a133-a134);
  a136=(a5*a120);
  a136=(a136-a124);
  a137=(a125*a119);
  a137=(a136+a137);
  a138=(a135*a137);
  a132=(a132+a138);
  a138=(a123+a126);
  a139=casadi_sq(a138);
  a140=(a133-a134);
  a141=casadi_sq(a140);
  a139=(a139+a141);
  a139=sqrt(a139);
  a132=(a132/a139);
  a141=arg[0]? arg[0][3] : 0;
  a142=arg[2]? arg[2][3] : 0;
  a123=(a123+a126);
  a126=casadi_sq(a123);
  a133=(a133-a134);
  a134=casadi_sq(a133);
  a126=(a126+a134);
  a126=sqrt(a126);
  a134=(a126-a19);
  a134=(a134/a21);
  a21=5.5180586452188241e-01;
  a143=(a134/a21);
  a143=(a143-a24);
  a144=casadi_sq(a143);
  a144=(a144/a26);
  a144=(-a144);
  a144=exp(a144);
  a26=(a142*a144);
  a145=(a132*a1);
  a146=(a118*a122);
  a147=(a121*a119);
  a146=(a146+a147);
  a147=(a120*a124);
  a148=(a130*a129);
  a147=(a147+a148);
  a148=(a146*a147);
  a149=(a146*a124);
  a150=(a121*a122);
  a151=(a118*a119);
  a150=(a150-a151);
  a151=(a150*a129);
  a149=(a149+a151);
  a151=(a120*a149);
  a148=(a148-a151);
  a148=(a148-a131);
  a131=(a127*a148);
  a151=(a130*a149);
  a152=(a150*a147);
  a151=(a151-a152);
  a151=(a151+a136);
  a136=(a135*a151);
  a131=(a131+a136);
  a131=(a131/a139);
  a136=(a131*a2);
  a145=(a145+a136);
  a29=(a145<=a29);
  a136=fabs(a145);
  a136=(a136/a32);
  a136=(a24-a136);
  a152=fabs(a145);
  a152=(a152/a32);
  a152=(a24+a152);
  a136=(a136/a152);
  a153=(a29?a136:0);
  a154=(!a29);
  a155=(a36*a145);
  a155=(a155/a32);
  a155=(a155/a38);
  a155=(a24-a155);
  a156=(a145/a32);
  a156=(a156/a38);
  a156=(a24-a156);
  a155=(a155/a156);
  a38=(a154?a155:0);
  a153=(a153+a38);
  a38=(a26*a153);
  a19=(a19<a134);
  a134=(a134/a21);
  a134=(a134-a24);
  a134=(a32*a134);
  a134=exp(a134);
  a21=(a134-a24);
  a21=(a21/a42);
  a21=(a19?a21:0);
  a38=(a38+a21);
  a21=5.5180586452188241e+00;
  a21=(a145/a21);
  a21=(a22*a21);
  a38=(a38+a21);
  a21=(a141*a38);
  a42=(a132*a21);
  a44=(a44+a42);
  a42=sin(a72);
  a157=cos(a72);
  a158=9.8100000000000005e+00;
  a159=cos(a6);
  a159=(a158*a159);
  a160=(a157*a159);
  a161=sin(a6);
  a161=(a158*a161);
  a162=(a42*a161);
  a160=(a160-a162);
  a162=(a79*a1);
  a163=(a157*a162);
  a164=(a163*a2);
  a160=(a160+a164);
  a164=(a1+a2);
  a165=(a164*a163);
  a160=(a160-a165);
  a165=(a42*a160);
  a166=(a42*a162);
  a167=(a164*a166);
  a168=(a157*a161);
  a169=(a42*a159);
  a168=(a168+a169);
  a169=(a166*a2);
  a168=(a168+a169);
  a167=(a167-a168);
  a168=(a157*a167);
  a165=(a165+a168);
  a165=(a79*a165);
  a44=(a44+a165);
  a165=(a4*a44);
  a168=9.6278838983177639e-01;
  a169=(a87*a99);
  a170=(a131*a21);
  a169=(a169+a170);
  a170=(a168*a169);
  a165=(a165+a170);
  a170=6.9253199970355839e-01;
  a165=(a165/a170);
  a170=(a3*a165);
  a171=9.6278838983177628e-01;
  a44=(a171*a44);
  a172=2.7025639012821789e-01;
  a169=(a172*a169);
  a44=(a44+a169);
  a170=(a170-a44);
  a44=3.7001900289039211e+00;
  a170=(a170/a44);
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
  a44=(a170*a13);
  a40=(a40*a44);
  if (res[1]!=0) res[1][1]=a40;
  a40=(a0*a50);
  a169=(a70*a40);
  if (res[1]!=0) res[1][2]=a169;
  a169=(a170*a50);
  a70=(a70*a169);
  if (res[1]!=0) res[1][3]=a70;
  a70=-3.9024390243902440e-01;
  a173=(a70*a87);
  a174=(a0*a88);
  a173=(a173+a174);
  a174=(a116*a173);
  if (res[1]!=0) res[1][4]=a174;
  a174=1.3902439024390245e+00;
  a175=(a174*a87);
  a176=(a170*a88);
  a175=(a175+a176);
  a116=(a116*a175);
  if (res[1]!=0) res[1][5]=a116;
  a116=(a70*a131);
  a176=(a0*a132);
  a116=(a116+a176);
  a176=(a38*a116);
  if (res[1]!=0) res[1][6]=a176;
  a176=(a174*a131);
  a177=(a170*a132);
  a176=(a176+a177);
  a38=(a38*a176);
  if (res[1]!=0) res[1][7]=a38;
  a38=cos(a6);
  a177=(a70*a21);
  a178=1.8122315551438725e-01;
  a116=(a141*a116);
  a179=(a22*a116);
  a179=(a178*a179);
  a180=-1.2121212121212121e+01;
  a181=(a26*a116);
  a155=(a155/a156);
  a182=(a181*a155);
  a182=(a180*a182);
  a182=(a22*a182);
  a182=(a154?a182:0);
  a179=(a179+a182);
  a182=(a181/a156);
  a182=(a180*a182);
  a182=(a22*a182);
  a182=(a36*a182);
  a182=(-a182);
  a182=(a154?a182:0);
  a179=(a179+a182);
  a136=(a136/a152);
  a182=(a181*a136);
  a182=(a22*a182);
  a183=casadi_sign(a145);
  a182=(a182*a183);
  a182=(-a182);
  a182=(a29?a182:0);
  a179=(a179+a182);
  a181=(a181/a152);
  a181=(a22*a181);
  a145=casadi_sign(a145);
  a181=(a181*a145);
  a181=(-a181);
  a181=(a29?a181:0);
  a179=(a179+a181);
  a181=(a2*a179);
  a177=(a177+a181);
  a181=(a177/a139);
  a182=(a135*a181);
  a184=(a130*a182);
  a185=(a127*a181);
  a186=(a120*a185);
  a184=(a184-a186);
  a186=(a124*a184);
  a187=(a147*a185);
  a186=(a186+a187);
  a187=(a121*a186);
  a188=(a129*a184);
  a189=(a147*a182);
  a188=(a188-a189);
  a189=(a118*a188);
  a187=(a187-a189);
  a189=(a0*a21);
  a190=(a1*a179);
  a189=(a189+a190);
  a190=(a189/a139);
  a191=(a135*a190);
  a192=(a125*a191);
  a187=(a187+a192);
  a192=(a149*a182);
  a193=(a146*a185);
  a194=(a150*a182);
  a193=(a193-a194);
  a194=(a129*a193);
  a192=(a192+a194);
  a133=(a133+a133);
  a194=1.1394939273245490e+00;
  a195=1.8122315551438726e+00;
  a196=6.7836549063042314e-03;
  a197=(a196*a116);
  a197=(a197*a134);
  a197=(a32*a197);
  a197=(a195*a197);
  a197=(a19?a197:0);
  a143=(a143+a143);
  a198=2.2222222222222223e+00;
  a116=(a153*a116);
  a116=(a142*a116);
  a116=(a144*a116);
  a116=(a198*a116);
  a116=(a143*a116);
  a116=(a195*a116);
  a197=(a197-a116);
  a197=(a194*a197);
  a126=(a126+a126);
  a197=(a197/a126);
  a116=(a133*a197);
  a140=(a140+a140);
  a199=(a131/a139);
  a177=(a199*a177);
  a200=(a132/a139);
  a189=(a200*a189);
  a177=(a177+a189);
  a189=(a139+a139);
  a177=(a177/a189);
  a201=(a140*a177);
  a202=(a116-a201);
  a203=(a151*a181);
  a204=(a137*a190);
  a203=(a203+a204);
  a202=(a202+a203);
  a204=(a5*a202);
  a192=(a192-a204);
  a204=(a127*a190);
  a205=(a185+a204);
  a206=(a5*a205);
  a192=(a192+a206);
  a206=(a121*a192);
  a187=(a187+a206);
  a123=(a123+a123);
  a197=(a123*a197);
  a138=(a138+a138);
  a177=(a138*a177);
  a206=(a197-a177);
  a181=(a148*a181);
  a190=(a128*a190);
  a181=(a181+a190);
  a206=(a206+a181);
  a206=(a125*a206);
  a187=(a187+a206);
  a206=(a146*a184);
  a190=(a120*a193);
  a206=(a206+a190);
  a182=(a182+a191);
  a206=(a206-a182);
  a197=(a197-a177);
  a197=(a197+a181);
  a206=(a206-a197);
  a206=(a79*a206);
  a187=(a187+a206);
  a206=(a124*a193);
  a185=(a149*a185);
  a206=(a206-a185);
  a182=(a5*a182);
  a206=(a206+a182);
  a197=(a5*a197);
  a206=(a206+a197);
  a197=(a118*a206);
  a187=(a187+a197);
  a187=(a38*a187);
  a197=cos(a6);
  a182=4.8780487804878025e-01;
  a185=(a182*a157);
  a181=(a157*a185);
  a177=(a182*a42);
  a191=(a42*a177);
  a181=(a181+a191);
  a181=(a158*a181);
  a181=(a197*a181);
  a191=sin(a6);
  a190=(a157*a177);
  a207=(a42*a185);
  a190=(a190-a207);
  a190=(a158*a190);
  a190=(a191*a190);
  a181=(a181+a190);
  a190=sin(a6);
  a207=(a121*a188);
  a208=(a118*a186);
  a207=(a207+a208);
  a201=(a201-a116);
  a201=(a201-a203);
  a201=(a125*a201);
  a207=(a207+a201);
  a201=(a118*a192);
  a207=(a207+a201);
  a184=(a150*a184);
  a193=(a130*a193);
  a184=(a184+a193);
  a184=(a184+a202);
  a184=(a184-a205);
  a184=(a79*a184);
  a207=(a207+a184);
  a204=(a125*a204);
  a207=(a207+a204);
  a204=(a121*a206);
  a207=(a207-a204);
  a207=(a190*a207);
  a181=(a181+a207);
  a187=(a187-a181);
  a181=sin(a6);
  a70=(a70*a99);
  a207=2.4999997933387313e-01;
  a173=(a97*a173);
  a204=(a22*a173);
  a204=(a207*a204);
  a184=(a102*a173);
  a114=(a114/a115);
  a205=(a184*a114);
  a205=(a180*a205);
  a205=(a22*a205);
  a205=(a113?a205:0);
  a204=(a204+a205);
  a205=(a184/a115);
  a205=(a180*a205);
  a205=(a22*a205);
  a205=(a36*a205);
  a205=(-a205);
  a205=(a113?a205:0);
  a204=(a204+a205);
  a110=(a110/a111);
  a205=(a184*a110);
  a205=(a22*a205);
  a202=casadi_sign(a103);
  a205=(a205*a202);
  a205=(-a205);
  a205=(a92?a205:0);
  a204=(a204+a205);
  a184=(a184/a111);
  a184=(a22*a184);
  a103=casadi_sign(a103);
  a184=(a184*a103);
  a184=(-a184);
  a184=(a92?a184:0);
  a204=(a204+a184);
  a184=(a2*a204);
  a70=(a70+a184);
  a184=(a70/a95);
  a205=(a91*a184);
  a193=(a86*a205);
  a201=(a83*a184);
  a203=(a75*a201);
  a193=(a193-a203);
  a203=(a85*a193);
  a116=(a105*a205);
  a203=(a203-a116);
  a116=(a76*a203);
  a208=(a80*a193);
  a209=(a105*a201);
  a208=(a208+a209);
  a209=(a73*a208);
  a116=(a116+a209);
  a96=(a96+a96);
  a209=(a87/a95);
  a70=(a209*a70);
  a210=(a88/a95);
  a211=(a0*a99);
  a212=(a1*a204);
  a211=(a211+a212);
  a212=(a210*a211);
  a70=(a70+a212);
  a212=(a95+a95);
  a70=(a70/a212);
  a213=(a96*a70);
  a89=(a89+a89);
  a214=2.4999997933387310e+00;
  a215=(a196*a173);
  a215=(a215*a90);
  a215=(a32*a215);
  a215=(a214*a215);
  a215=(a117?a215:0);
  a100=(a100+a100);
  a173=(a112*a173);
  a173=(a98*a173);
  a173=(a101*a173);
  a173=(a198*a173);
  a173=(a100*a173);
  a173=(a214*a173);
  a215=(a215-a173);
  a215=(a194*a215);
  a82=(a82+a82);
  a215=(a215/a82);
  a173=(a89*a215);
  a216=(a213-a173);
  a217=(a109*a184);
  a211=(a211/a95);
  a218=(a93*a211);
  a217=(a217+a218);
  a216=(a216-a217);
  a216=(a81*a216);
  a116=(a116+a216);
  a216=(a107*a205);
  a218=(a104*a201);
  a219=(a108*a205);
  a218=(a218-a219);
  a219=(a85*a218);
  a216=(a216+a219);
  a173=(a173-a213);
  a173=(a173+a217);
  a217=(a5*a173);
  a216=(a216-a217);
  a217=(a83*a211);
  a213=(a201+a217);
  a219=(a5*a213);
  a216=(a216+a219);
  a219=(a73*a216);
  a116=(a116+a219);
  a219=(a108*a193);
  a220=(a86*a218);
  a219=(a219+a220);
  a219=(a219+a173);
  a219=(a219-a213);
  a219=(a79*a219);
  a116=(a116+a219);
  a217=(a81*a217);
  a116=(a116+a217);
  a217=(a80*a218);
  a201=(a107*a201);
  a217=(a217-a201);
  a201=(a91*a211);
  a205=(a205+a201);
  a219=(a5*a205);
  a217=(a217+a219);
  a78=(a78+a78);
  a215=(a78*a215);
  a94=(a94+a94);
  a70=(a94*a70);
  a219=(a215-a70);
  a184=(a106*a184);
  a211=(a84*a211);
  a184=(a184+a211);
  a219=(a219+a184);
  a211=(a5*a219);
  a217=(a217+a211);
  a211=(a76*a217);
  a116=(a116-a211);
  a116=(a181*a116);
  a187=(a187-a116);
  a116=cos(a6);
  a211=(a76*a208);
  a213=(a73*a203);
  a211=(a211-a213);
  a201=(a81*a201);
  a211=(a211+a201);
  a201=(a76*a216);
  a211=(a211+a201);
  a215=(a215-a70);
  a215=(a215+a184);
  a215=(a81*a215);
  a211=(a211+a215);
  a193=(a104*a193);
  a218=(a75*a218);
  a193=(a193+a218);
  a193=(a193-a205);
  a193=(a193-a219);
  a193=(a79*a193);
  a211=(a211+a193);
  a193=(a73*a217);
  a211=(a211+a193);
  a211=(a116*a211);
  a187=(a187+a211);
  a211=sin(a6);
  a193=1.8890487445150401e+00;
  a40=(a54*a40);
  a219=(a196*a40);
  a219=(a219*a57);
  a219=(a32*a219);
  a219=(a193*a219);
  a219=(a71?a219:0);
  a59=(a59+a59);
  a205=(a66*a40);
  a205=(a55*a205);
  a205=(a60*a205);
  a205=(a198*a205);
  a205=(a59*a205);
  a205=(a193*a205);
  a219=(a219-a205);
  a219=(a194*a219);
  a56=(a56+a56);
  a219=(a219/a56);
  a205=(a48*a219);
  a218=(a50/a53);
  a215=(a0*a58);
  a184=1.8890487445150400e-01;
  a70=(a22*a40);
  a70=(a184*a70);
  a40=(a61*a40);
  a68=(a68/a69);
  a201=(a40*a68);
  a201=(a180*a201);
  a201=(a22*a201);
  a201=(a67?a201:0);
  a70=(a70+a201);
  a201=(a40/a69);
  a201=(a180*a201);
  a201=(a22*a201);
  a201=(a36*a201);
  a201=(-a201);
  a201=(a67?a201:0);
  a70=(a70+a201);
  a64=(a64/a65);
  a201=(a40*a64);
  a201=(a22*a201);
  a213=casadi_sign(a62);
  a201=(a201*a213);
  a201=(-a201);
  a201=(a63?a201:0);
  a70=(a70+a201);
  a40=(a40/a65);
  a40=(a22*a40);
  a62=casadi_sign(a62);
  a40=(a40*a62);
  a40=(-a40);
  a40=(a63?a40:0);
  a70=(a70+a40);
  a40=(a1*a70);
  a215=(a215+a40);
  a40=(a218*a215);
  a201=(a53+a53);
  a40=(a40/a201);
  a173=(a48*a40);
  a205=(a205-a173);
  a215=(a215/a53);
  a173=(a45*a215);
  a205=(a205-a173);
  a205=(a5*a205);
  a173=(a47*a215);
  a173=(a5*a173);
  a205=(a205+a173);
  a205=(a211*a205);
  a187=(a187-a205);
  a205=cos(a6);
  a46=(a46+a46);
  a219=(a46*a219);
  a52=(a52+a52);
  a40=(a52*a40);
  a219=(a219-a40);
  a40=(a49*a215);
  a219=(a219+a40);
  a219=(a5*a219);
  a215=(a51*a215);
  a215=(a5*a215);
  a219=(a219-a215);
  a219=(a205*a219);
  a187=(a187+a219);
  a219=sin(a6);
  a215=2.4999995987780066e+00;
  a165=(a17*a165);
  a40=(a196*a165);
  a40=(a40*a20);
  a40=(a32*a40);
  a40=(a215*a40);
  a40=(a41?a40:0);
  a23=(a23+a23);
  a173=(a34*a165);
  a173=(a18*a173);
  a173=(a25*a173);
  a173=(a198*a173);
  a173=(a23*a173);
  a173=(a215*a173);
  a40=(a40-a173);
  a40=(a194*a40);
  a9=(a9+a9);
  a40=(a40/a9);
  a173=(a11*a40);
  a220=(a13/a16);
  a0=(a0*a43);
  a221=2.4999995987780066e-01;
  a222=(a22*a165);
  a222=(a221*a222);
  a165=(a27*a165);
  a37=(a37/a39);
  a223=(a165*a37);
  a223=(a180*a223);
  a223=(a22*a223);
  a223=(a35?a223:0);
  a222=(a222+a223);
  a223=(a165/a39);
  a223=(a180*a223);
  a223=(a22*a223);
  a223=(a36*a223);
  a223=(-a223);
  a223=(a35?a223:0);
  a222=(a222+a223);
  a31=(a31/a33);
  a223=(a165*a31);
  a223=(a22*a223);
  a224=casadi_sign(a28);
  a223=(a223*a224);
  a223=(-a223);
  a223=(a30?a223:0);
  a222=(a222+a223);
  a165=(a165/a33);
  a165=(a22*a165);
  a28=casadi_sign(a28);
  a165=(a165*a28);
  a165=(-a165);
  a165=(a30?a165:0);
  a222=(a222+a165);
  a165=(a1*a222);
  a0=(a0+a165);
  a165=(a220*a0);
  a223=(a16+a16);
  a165=(a165/a223);
  a225=(a11*a165);
  a173=(a173-a225);
  a0=(a0/a16);
  a225=(a7*a0);
  a173=(a173-a225);
  a173=(a5*a173);
  a225=(a10*a0);
  a225=(a5*a225);
  a173=(a173+a225);
  a173=(a219*a173);
  a187=(a187-a173);
  a6=cos(a6);
  a8=(a8+a8);
  a40=(a8*a40);
  a15=(a15+a15);
  a165=(a15*a165);
  a40=(a40-a165);
  a165=(a12*a0);
  a40=(a40+a165);
  a40=(a5*a40);
  a0=(a14*a0);
  a0=(a5*a0);
  a40=(a40-a0);
  a40=(a6*a40);
  a187=(a187+a40);
  if (res[1]!=0) res[1][8]=a187;
  a187=(a174*a21);
  a176=(a141*a176);
  a40=(a22*a176);
  a178=(a178*a40);
  a26=(a26*a176);
  a155=(a26*a155);
  a155=(a180*a155);
  a155=(a22*a155);
  a155=(a154?a155:0);
  a178=(a178+a155);
  a156=(a26/a156);
  a156=(a180*a156);
  a156=(a22*a156);
  a156=(a36*a156);
  a156=(-a156);
  a154=(a154?a156:0);
  a178=(a178+a154);
  a136=(a26*a136);
  a136=(a22*a136);
  a136=(a136*a183);
  a136=(-a136);
  a136=(a29?a136:0);
  a178=(a178+a136);
  a26=(a26/a152);
  a26=(a22*a26);
  a26=(a26*a145);
  a26=(-a26);
  a29=(a29?a26:0);
  a178=(a178+a29);
  a29=(a2*a178);
  a187=(a187+a29);
  a29=(a187/a139);
  a26=(a135*a29);
  a145=(a130*a26);
  a152=(a127*a29);
  a136=(a120*a152);
  a145=(a145-a136);
  a136=(a124*a145);
  a183=(a147*a152);
  a136=(a136+a183);
  a183=(a121*a136);
  a154=(a129*a145);
  a147=(a147*a26);
  a154=(a154-a147);
  a147=(a118*a154);
  a183=(a183-a147);
  a21=(a170*a21);
  a147=(a1*a178);
  a21=(a21+a147);
  a139=(a21/a139);
  a135=(a135*a139);
  a147=(a125*a135);
  a183=(a183+a147);
  a147=(a149*a26);
  a156=(a146*a152);
  a155=(a150*a26);
  a156=(a156-a155);
  a129=(a129*a156);
  a147=(a147+a129);
  a129=(a196*a176);
  a129=(a129*a134);
  a129=(a32*a129);
  a129=(a195*a129);
  a19=(a19?a129:0);
  a176=(a153*a176);
  a142=(a142*a176);
  a142=(a144*a142);
  a142=(a198*a142);
  a143=(a143*a142);
  a195=(a195*a143);
  a19=(a19-a195);
  a19=(a194*a19);
  a19=(a19/a126);
  a133=(a133*a19);
  a199=(a199*a187);
  a200=(a200*a21);
  a199=(a199+a200);
  a199=(a199/a189);
  a140=(a140*a199);
  a189=(a133-a140);
  a151=(a151*a29);
  a137=(a137*a139);
  a151=(a151+a137);
  a189=(a189+a151);
  a137=(a5*a189);
  a147=(a147-a137);
  a127=(a127*a139);
  a137=(a152+a127);
  a200=(a5*a137);
  a147=(a147+a200);
  a200=(a121*a147);
  a183=(a183+a200);
  a123=(a123*a19);
  a138=(a138*a199);
  a199=(a123-a138);
  a148=(a148*a29);
  a128=(a128*a139);
  a148=(a148+a128);
  a199=(a199+a148);
  a199=(a125*a199);
  a183=(a183+a199);
  a146=(a146*a145);
  a120=(a120*a156);
  a146=(a146+a120);
  a26=(a26+a135);
  a146=(a146-a26);
  a123=(a123-a138);
  a123=(a123+a148);
  a146=(a146-a123);
  a146=(a79*a146);
  a183=(a183+a146);
  a124=(a124*a156);
  a149=(a149*a152);
  a124=(a124-a149);
  a26=(a5*a26);
  a124=(a124+a26);
  a123=(a5*a123);
  a124=(a124+a123);
  a123=(a118*a124);
  a183=(a183+a123);
  a38=(a38*a183);
  a183=-4.8780487804877992e-01;
  a123=(a183*a157);
  a26=(a157*a123);
  a149=(a183*a42);
  a152=(a42*a149);
  a26=(a26+a152);
  a26=(a158*a26);
  a197=(a197*a26);
  a26=(a157*a149);
  a152=(a42*a123);
  a26=(a26-a152);
  a158=(a158*a26);
  a191=(a191*a158);
  a197=(a197+a191);
  a191=(a121*a154);
  a158=(a118*a136);
  a191=(a191+a158);
  a140=(a140-a133);
  a140=(a140-a151);
  a140=(a125*a140);
  a191=(a191+a140);
  a118=(a118*a147);
  a191=(a191+a118);
  a150=(a150*a145);
  a130=(a130*a156);
  a150=(a150+a130);
  a150=(a150+a189);
  a150=(a150-a137);
  a150=(a79*a150);
  a191=(a191+a150);
  a125=(a125*a127);
  a191=(a191+a125);
  a121=(a121*a124);
  a191=(a191-a121);
  a190=(a190*a191);
  a197=(a197+a190);
  a38=(a38-a197);
  a174=(a174*a99);
  a175=(a97*a175);
  a197=(a22*a175);
  a207=(a207*a197);
  a102=(a102*a175);
  a114=(a102*a114);
  a114=(a180*a114);
  a114=(a22*a114);
  a114=(a113?a114:0);
  a207=(a207+a114);
  a115=(a102/a115);
  a115=(a180*a115);
  a115=(a22*a115);
  a115=(a36*a115);
  a115=(-a115);
  a113=(a113?a115:0);
  a207=(a207+a113);
  a110=(a102*a110);
  a110=(a22*a110);
  a110=(a110*a202);
  a110=(-a110);
  a110=(a92?a110:0);
  a207=(a207+a110);
  a102=(a102/a111);
  a102=(a22*a102);
  a102=(a102*a103);
  a102=(-a102);
  a92=(a92?a102:0);
  a207=(a207+a92);
  a92=(a2*a207);
  a174=(a174+a92);
  a92=(a174/a95);
  a102=(a91*a92);
  a103=(a86*a102);
  a111=(a83*a92);
  a110=(a75*a111);
  a103=(a103-a110);
  a110=(a85*a103);
  a202=(a105*a102);
  a110=(a110-a202);
  a202=(a76*a110);
  a113=(a80*a103);
  a105=(a105*a111);
  a113=(a113+a105);
  a105=(a73*a113);
  a202=(a202+a105);
  a209=(a209*a174);
  a99=(a170*a99);
  a174=(a1*a207);
  a99=(a99+a174);
  a210=(a210*a99);
  a209=(a209+a210);
  a209=(a209/a212);
  a96=(a96*a209);
  a212=(a196*a175);
  a212=(a212*a90);
  a212=(a32*a212);
  a212=(a214*a212);
  a117=(a117?a212:0);
  a175=(a112*a175);
  a98=(a98*a175);
  a98=(a101*a98);
  a98=(a198*a98);
  a100=(a100*a98);
  a214=(a214*a100);
  a117=(a117-a214);
  a117=(a194*a117);
  a117=(a117/a82);
  a89=(a89*a117);
  a82=(a96-a89);
  a109=(a109*a92);
  a99=(a99/a95);
  a93=(a93*a99);
  a109=(a109+a93);
  a82=(a82-a109);
  a82=(a81*a82);
  a202=(a202+a82);
  a82=(a107*a102);
  a93=(a104*a111);
  a95=(a108*a102);
  a93=(a93-a95);
  a85=(a85*a93);
  a82=(a82+a85);
  a89=(a89-a96);
  a89=(a89+a109);
  a109=(a5*a89);
  a82=(a82-a109);
  a83=(a83*a99);
  a109=(a111+a83);
  a96=(a5*a109);
  a82=(a82+a96);
  a96=(a73*a82);
  a202=(a202+a96);
  a108=(a108*a103);
  a86=(a86*a93);
  a108=(a108+a86);
  a108=(a108+a89);
  a108=(a108-a109);
  a108=(a79*a108);
  a202=(a202+a108);
  a83=(a81*a83);
  a202=(a202+a83);
  a80=(a80*a93);
  a107=(a107*a111);
  a80=(a80-a107);
  a91=(a91*a99);
  a102=(a102+a91);
  a107=(a5*a102);
  a80=(a80+a107);
  a78=(a78*a117);
  a94=(a94*a209);
  a209=(a78-a94);
  a106=(a106*a92);
  a84=(a84*a99);
  a106=(a106+a84);
  a209=(a209+a106);
  a84=(a5*a209);
  a80=(a80+a84);
  a84=(a76*a80);
  a202=(a202-a84);
  a181=(a181*a202);
  a38=(a38-a181);
  a181=(a76*a113);
  a202=(a73*a110);
  a181=(a181-a202);
  a91=(a81*a91);
  a181=(a181+a91);
  a76=(a76*a82);
  a181=(a181+a76);
  a78=(a78-a94);
  a78=(a78+a106);
  a81=(a81*a78);
  a181=(a181+a81);
  a104=(a104*a103);
  a75=(a75*a93);
  a104=(a104+a75);
  a104=(a104-a102);
  a104=(a104-a209);
  a104=(a79*a104);
  a181=(a181+a104);
  a73=(a73*a80);
  a181=(a181+a73);
  a116=(a116*a181);
  a38=(a38+a116);
  a169=(a54*a169);
  a116=(a196*a169);
  a116=(a116*a57);
  a116=(a32*a116);
  a116=(a193*a116);
  a71=(a71?a116:0);
  a116=(a66*a169);
  a55=(a55*a116);
  a55=(a60*a55);
  a55=(a198*a55);
  a59=(a59*a55);
  a193=(a193*a59);
  a71=(a71-a193);
  a71=(a194*a71);
  a71=(a71/a56);
  a56=(a48*a71);
  a58=(a170*a58);
  a193=(a22*a169);
  a184=(a184*a193);
  a61=(a61*a169);
  a68=(a61*a68);
  a68=(a180*a68);
  a68=(a22*a68);
  a68=(a67?a68:0);
  a184=(a184+a68);
  a69=(a61/a69);
  a69=(a180*a69);
  a69=(a22*a69);
  a69=(a36*a69);
  a69=(-a69);
  a67=(a67?a69:0);
  a184=(a184+a67);
  a64=(a61*a64);
  a64=(a22*a64);
  a64=(a64*a213);
  a64=(-a64);
  a64=(a63?a64:0);
  a184=(a184+a64);
  a61=(a61/a65);
  a61=(a22*a61);
  a61=(a61*a62);
  a61=(-a61);
  a63=(a63?a61:0);
  a184=(a184+a63);
  a63=(a1*a184);
  a58=(a58+a63);
  a218=(a218*a58);
  a218=(a218/a201);
  a48=(a48*a218);
  a56=(a56-a48);
  a58=(a58/a53);
  a45=(a45*a58);
  a56=(a56-a45);
  a56=(a5*a56);
  a47=(a47*a58);
  a47=(a5*a47);
  a56=(a56+a47);
  a211=(a211*a56);
  a38=(a38-a211);
  a46=(a46*a71);
  a52=(a52*a218);
  a46=(a46-a52);
  a49=(a49*a58);
  a46=(a46+a49);
  a46=(a5*a46);
  a51=(a51*a58);
  a51=(a5*a51);
  a46=(a46-a51);
  a205=(a205*a46);
  a38=(a38+a205);
  a44=(a17*a44);
  a196=(a196*a44);
  a196=(a196*a20);
  a32=(a32*a196);
  a32=(a215*a32);
  a41=(a41?a32:0);
  a32=(a34*a44);
  a18=(a18*a32);
  a18=(a25*a18);
  a198=(a198*a18);
  a23=(a23*a198);
  a215=(a215*a23);
  a41=(a41-a215);
  a194=(a194*a41);
  a194=(a194/a9);
  a9=(a11*a194);
  a170=(a170*a43);
  a43=(a22*a44);
  a221=(a221*a43);
  a27=(a27*a44);
  a37=(a27*a37);
  a37=(a180*a37);
  a37=(a22*a37);
  a37=(a35?a37:0);
  a221=(a221+a37);
  a39=(a27/a39);
  a180=(a180*a39);
  a180=(a22*a180);
  a36=(a36*a180);
  a36=(-a36);
  a35=(a35?a36:0);
  a221=(a221+a35);
  a31=(a27*a31);
  a31=(a22*a31);
  a31=(a31*a224);
  a31=(-a31);
  a31=(a30?a31:0);
  a221=(a221+a31);
  a27=(a27/a33);
  a22=(a22*a27);
  a22=(a22*a28);
  a22=(-a22);
  a30=(a30?a22:0);
  a221=(a221+a30);
  a1=(a1*a221);
  a170=(a170+a1);
  a220=(a220*a170);
  a220=(a220/a223);
  a11=(a11*a220);
  a9=(a9-a11);
  a170=(a170/a16);
  a7=(a7*a170);
  a9=(a9-a7);
  a9=(a5*a9);
  a10=(a10*a170);
  a10=(a5*a10);
  a9=(a9+a10);
  a219=(a219*a9);
  a38=(a38-a219);
  a8=(a8*a194);
  a15=(a15*a220);
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
  a38=cos(a72);
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
  a5=sin(a72);
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
  a182=sin(a72);
  a170=(a122*a188);
  a12=(a119*a186);
  a170=(a170+a12);
  a12=(a119*a192);
  a170=(a170+a12);
  a12=(a122*a206);
  a170=(a170-a12);
  a170=(a182*a170);
  a8=(a8-a170);
  a170=cos(a72);
  a186=(a122*a186);
  a188=(a119*a188);
  a186=(a186-a188);
  a192=(a122*a192);
  a186=(a186+a192);
  a206=(a119*a206);
  a186=(a186+a206);
  a186=(a170*a186);
  a8=(a8+a186);
  a186=sin(a72);
  a206=(a77*a203);
  a192=(a74*a208);
  a206=(a206+a192);
  a192=(a74*a216);
  a206=(a206+a192);
  a192=(a77*a217);
  a206=(a206-a192);
  a206=(a186*a206);
  a8=(a8-a206);
  a72=cos(a72);
  a208=(a77*a208);
  a203=(a74*a203);
  a208=(a208-a203);
  a216=(a77*a216);
  a208=(a208+a216);
  a217=(a74*a217);
  a208=(a208+a217);
  a208=(a72*a208);
  a8=(a8+a208);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a164*a123);
  a208=(a2*a123);
  a8=(a8-a208);
  a208=(a162*a8);
  a217=(a159*a123);
  a208=(a208-a217);
  a160=(a183*a160);
  a208=(a208+a160);
  a160=(a161*a149);
  a208=(a208-a160);
  a38=(a38*a208);
  a183=(a183*a167);
  a161=(a161*a123);
  a183=(a183-a161);
  a2=(a2*a149);
  a164=(a164*a149);
  a2=(a2-a164);
  a162=(a162*a2);
  a183=(a183+a162);
  a159=(a159*a149);
  a183=(a183+a159);
  a5=(a5*a183);
  a38=(a38-a5);
  a5=(a122*a154);
  a183=(a119*a136);
  a5=(a5+a183);
  a183=(a119*a147);
  a5=(a5+a183);
  a183=(a122*a124);
  a5=(a5-a183);
  a182=(a182*a5);
  a38=(a38-a182);
  a136=(a122*a136);
  a154=(a119*a154);
  a136=(a136-a154);
  a122=(a122*a147);
  a136=(a136+a122);
  a119=(a119*a124);
  a136=(a136+a119);
  a170=(a170*a136);
  a38=(a38+a170);
  a170=(a77*a110);
  a136=(a74*a113);
  a170=(a170+a136);
  a136=(a74*a82);
  a170=(a170+a136);
  a136=(a77*a80);
  a170=(a170-a136);
  a186=(a186*a170);
  a38=(a38-a186);
  a113=(a77*a113);
  a110=(a74*a110);
  a113=(a113-a110);
  a77=(a77*a82);
  a113=(a113+a77);
  a74=(a74*a80);
  a113=(a113+a74);
  a72=(a72*a113);
  a38=(a38+a72);
  if (res[1]!=0) res[1][11]=a38;
  a38=-1.;
  if (res[1]!=0) res[1][12]=a38;
  a72=(a166*a185);
  a113=(a163*a177);
  a72=(a72-a113);
  a6=(a42*a6);
  a14=(a157*a14);
  a6=(a6+a14);
  a6=(a79*a6);
  a6=(a72+a6);
  a14=(a132*a179);
  a6=(a6+a14);
  a14=(a88*a204);
  a6=(a6+a14);
  a70=(a50*a70);
  a6=(a6+a70);
  a222=(a13*a222);
  a6=(a6+a222);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a166*a123);
  a222=(a163*a149);
  a6=(a6-a222);
  a42=(a42*a8);
  a157=(a157*a2);
  a42=(a42+a157);
  a79=(a79*a42);
  a79=(a6+a79);
  a42=(a132*a178);
  a79=(a79+a42);
  a42=(a88*a207);
  a79=(a79+a42);
  a184=(a50*a184);
  a79=(a79+a184);
  a221=(a13*a221);
  a79=(a79+a221);
  if (res[1]!=0) res[1][14]=a79;
  if (res[1]!=0) res[1][15]=a38;
  a185=(a166*a185);
  a72=(a72-a185);
  a177=(a163*a177);
  a72=(a72+a177);
  a179=(a131*a179);
  a72=(a72+a179);
  a204=(a87*a204);
  a72=(a72+a204);
  if (res[1]!=0) res[1][16]=a72;
  a166=(a166*a123);
  a6=(a6-a166);
  a163=(a163*a149);
  a6=(a6+a163);
  a178=(a131*a178);
  a6=(a6+a178);
  a207=(a87*a207);
  a6=(a6+a207);
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
  a66=(a66*a60);
  a54=(a54*a66);
  a50=(a50*a54);
  a54=(a4*a50);
  a54=(a24*a54);
  a66=(a3*a54);
  a50=(a171*a50);
  a66=(a66-a50);
  a66=(a172*a66);
  a66=(-a66);
  if (res[3]!=0) res[3][2]=a66;
  if (res[3]!=0) res[3][3]=a54;
  a112=(a112*a101);
  a97=(a97*a112);
  a88=(a88*a97);
  a112=(a4*a88);
  a87=(a87*a97);
  a97=(a168*a87);
  a112=(a112+a97);
  a112=(a24*a112);
  a97=(a3*a112);
  a88=(a171*a88);
  a87=(a172*a87);
  a88=(a88+a87);
  a97=(a97-a88);
  a97=(a172*a97);
  a97=(-a97);
  if (res[3]!=0) res[3][4]=a97;
  if (res[3]!=0) res[3][5]=a112;
  a153=(a153*a144);
  a141=(a141*a153);
  a132=(a132*a141);
  a4=(a4*a132);
  a131=(a131*a141);
  a168=(a168*a131);
  a4=(a4+a168);
  a24=(a24*a4);
  a3=(a3*a24);
  a171=(a171*a132);
  a131=(a172*a131);
  a171=(a171+a131);
  a3=(a3-a171);
  a172=(a172*a3);
  a172=(-a172);
  if (res[3]!=0) res[3][6]=a172;
  if (res[3]!=0) res[3][7]=a24;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    case 4: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10302646_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
