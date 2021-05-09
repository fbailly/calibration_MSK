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
  #define CASADI_PREFIX(ID) model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_ ## ID
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

/* model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z:(i0[8],i1[8],i2[4],i3[],i4[])->(o0[8],o1[8x8,18nz],o2[8x8,8nz],o3[8x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a194, a195, a196, a197, a198, a199, a2, a20, a200, a201, a202, a203, a204, a205, a206, a207, a208, a209, a21, a210, a211, a212, a213, a214, a215, a216, a217, a218, a219, a22, a220, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a22=8.4999934524596654e-01;
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
  a43=8.4999934524596661e+00;
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
  a58=4.0000059675727639e-01;
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
  a58=4.0000059675727639e+00;
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
  a99=8.4999960239807870e-01;
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
  a99=8.4999960239807866e+00;
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
  a21=8.4999980481589466e-01;
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
  a21=8.4999980481589468e+00;
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
  a4=(a4*a44);
  a165=9.6278838983177639e-01;
  a168=(a87*a99);
  a169=(a131*a21);
  a168=(a168+a169);
  a165=(a165*a168);
  a4=(a4+a165);
  a165=6.9253199970355839e-01;
  a4=(a4/a165);
  a3=(a3*a4);
  a165=9.6278838983177628e-01;
  a165=(a165*a44);
  a44=2.7025639012821789e-01;
  a44=(a44*a168);
  a165=(a165+a44);
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
  a3=(a40*a4);
  if (res[1]!=0) res[1][0]=a3;
  a3=-3.9024390243902396e-01;
  a165=(a3*a13);
  a40=(a40*a165);
  if (res[1]!=0) res[1][1]=a40;
  a40=(a0*a50);
  a44=(a70*a40);
  if (res[1]!=0) res[1][2]=a44;
  a44=(a3*a50);
  a70=(a70*a44);
  if (res[1]!=0) res[1][3]=a70;
  a70=-3.9024390243902440e-01;
  a168=(a70*a87);
  a169=(a0*a88);
  a168=(a168+a169);
  a169=(a116*a168);
  if (res[1]!=0) res[1][4]=a169;
  a169=1.3902439024390245e+00;
  a170=(a169*a87);
  a171=(a3*a88);
  a170=(a170+a171);
  a116=(a116*a170);
  if (res[1]!=0) res[1][5]=a116;
  a116=(a70*a131);
  a171=(a0*a132);
  a116=(a116+a171);
  a171=(a38*a116);
  if (res[1]!=0) res[1][6]=a171;
  a171=(a169*a131);
  a172=(a3*a132);
  a171=(a171+a172);
  a38=(a38*a171);
  if (res[1]!=0) res[1][7]=a38;
  a38=cos(a6);
  a172=(a70*a21);
  a173=1.1764708583863670e-01;
  a116=(a141*a116);
  a174=(a22*a116);
  a174=(a173*a174);
  a175=-1.2121212121212121e+01;
  a176=(a26*a116);
  a155=(a155/a156);
  a177=(a176*a155);
  a177=(a175*a177);
  a177=(a22*a177);
  a177=(a154?a177:0);
  a174=(a174+a177);
  a177=(a176/a156);
  a177=(a175*a177);
  a177=(a22*a177);
  a177=(a36*a177);
  a177=(-a177);
  a177=(a154?a177:0);
  a174=(a174+a177);
  a136=(a136/a152);
  a177=(a176*a136);
  a177=(a22*a177);
  a178=casadi_sign(a145);
  a177=(a177*a178);
  a177=(-a177);
  a177=(a29?a177:0);
  a174=(a174+a177);
  a176=(a176/a152);
  a176=(a22*a176);
  a145=casadi_sign(a145);
  a176=(a176*a145);
  a176=(-a176);
  a176=(a29?a176:0);
  a174=(a174+a176);
  a176=(a2*a174);
  a172=(a172+a176);
  a176=(a172/a139);
  a177=(a135*a176);
  a179=(a130*a177);
  a180=(a127*a176);
  a181=(a120*a180);
  a179=(a179-a181);
  a181=(a124*a179);
  a182=(a147*a180);
  a181=(a181+a182);
  a182=(a121*a181);
  a183=(a129*a179);
  a184=(a147*a177);
  a183=(a183-a184);
  a184=(a118*a183);
  a182=(a182-a184);
  a184=(a0*a21);
  a185=(a1*a174);
  a184=(a184+a185);
  a185=(a184/a139);
  a186=(a135*a185);
  a187=(a125*a186);
  a182=(a182+a187);
  a187=(a149*a177);
  a188=(a146*a180);
  a189=(a150*a177);
  a188=(a188-a189);
  a189=(a129*a188);
  a187=(a187+a189);
  a133=(a133+a133);
  a189=1.1394939273245490e+00;
  a190=1.1764708583863670e+00;
  a191=6.7836549063042314e-03;
  a192=(a191*a116);
  a192=(a192*a134);
  a192=(a32*a192);
  a192=(a190*a192);
  a192=(a19?a192:0);
  a143=(a143+a143);
  a193=2.2222222222222223e+00;
  a116=(a153*a116);
  a116=(a142*a116);
  a116=(a144*a116);
  a116=(a193*a116);
  a116=(a143*a116);
  a116=(a190*a116);
  a192=(a192-a116);
  a192=(a189*a192);
  a126=(a126+a126);
  a192=(a192/a126);
  a116=(a133*a192);
  a140=(a140+a140);
  a194=(a131/a139);
  a172=(a194*a172);
  a195=(a132/a139);
  a184=(a195*a184);
  a172=(a172+a184);
  a184=(a139+a139);
  a172=(a172/a184);
  a196=(a140*a172);
  a197=(a116-a196);
  a198=(a151*a176);
  a199=(a137*a185);
  a198=(a198+a199);
  a197=(a197+a198);
  a199=(a5*a197);
  a187=(a187-a199);
  a199=(a127*a185);
  a200=(a180+a199);
  a201=(a5*a200);
  a187=(a187+a201);
  a201=(a121*a187);
  a182=(a182+a201);
  a123=(a123+a123);
  a192=(a123*a192);
  a138=(a138+a138);
  a172=(a138*a172);
  a201=(a192-a172);
  a176=(a148*a176);
  a185=(a128*a185);
  a176=(a176+a185);
  a201=(a201+a176);
  a201=(a125*a201);
  a182=(a182+a201);
  a201=(a146*a179);
  a185=(a120*a188);
  a201=(a201+a185);
  a177=(a177+a186);
  a201=(a201-a177);
  a192=(a192-a172);
  a192=(a192+a176);
  a201=(a201-a192);
  a201=(a79*a201);
  a182=(a182+a201);
  a201=(a124*a188);
  a180=(a149*a180);
  a201=(a201-a180);
  a177=(a5*a177);
  a201=(a201+a177);
  a192=(a5*a192);
  a201=(a201+a192);
  a192=(a118*a201);
  a182=(a182+a192);
  a182=(a38*a182);
  a192=cos(a6);
  a177=4.8780487804878025e-01;
  a180=(a177*a157);
  a176=(a157*a180);
  a172=(a177*a42);
  a186=(a42*a172);
  a176=(a176+a186);
  a176=(a158*a176);
  a176=(a192*a176);
  a186=sin(a6);
  a185=(a157*a172);
  a202=(a42*a180);
  a185=(a185-a202);
  a185=(a158*a185);
  a185=(a186*a185);
  a176=(a176+a185);
  a185=sin(a6);
  a202=(a121*a183);
  a203=(a118*a181);
  a202=(a202+a203);
  a196=(a196-a116);
  a196=(a196-a198);
  a196=(a125*a196);
  a202=(a202+a196);
  a196=(a118*a187);
  a202=(a202+a196);
  a179=(a150*a179);
  a188=(a130*a188);
  a179=(a179+a188);
  a179=(a179+a197);
  a179=(a179-a200);
  a179=(a79*a179);
  a202=(a202+a179);
  a199=(a125*a199);
  a202=(a202+a199);
  a199=(a121*a201);
  a202=(a202-a199);
  a202=(a185*a202);
  a176=(a176+a202);
  a182=(a182-a176);
  a176=sin(a6);
  a70=(a70*a99);
  a202=1.1764711385496295e-01;
  a168=(a97*a168);
  a199=(a22*a168);
  a199=(a202*a199);
  a179=(a102*a168);
  a114=(a114/a115);
  a200=(a179*a114);
  a200=(a175*a200);
  a200=(a22*a200);
  a200=(a113?a200:0);
  a199=(a199+a200);
  a200=(a179/a115);
  a200=(a175*a200);
  a200=(a22*a200);
  a200=(a36*a200);
  a200=(-a200);
  a200=(a113?a200:0);
  a199=(a199+a200);
  a110=(a110/a111);
  a200=(a179*a110);
  a200=(a22*a200);
  a197=casadi_sign(a103);
  a200=(a200*a197);
  a200=(-a200);
  a200=(a92?a200:0);
  a199=(a199+a200);
  a179=(a179/a111);
  a179=(a22*a179);
  a103=casadi_sign(a103);
  a179=(a179*a103);
  a179=(-a179);
  a179=(a92?a179:0);
  a199=(a199+a179);
  a179=(a2*a199);
  a70=(a70+a179);
  a179=(a70/a95);
  a200=(a91*a179);
  a188=(a86*a200);
  a196=(a83*a179);
  a198=(a75*a196);
  a188=(a188-a198);
  a198=(a85*a188);
  a116=(a105*a200);
  a198=(a198-a116);
  a116=(a76*a198);
  a203=(a80*a188);
  a204=(a105*a196);
  a203=(a203+a204);
  a204=(a73*a203);
  a116=(a116+a204);
  a96=(a96+a96);
  a204=(a87/a95);
  a70=(a204*a70);
  a205=(a88/a95);
  a206=(a0*a99);
  a207=(a1*a199);
  a206=(a206+a207);
  a207=(a205*a206);
  a70=(a70+a207);
  a207=(a95+a95);
  a70=(a70/a207);
  a208=(a96*a70);
  a89=(a89+a89);
  a209=1.1764711385496294e+00;
  a210=(a191*a168);
  a210=(a210*a90);
  a210=(a32*a210);
  a210=(a209*a210);
  a210=(a117?a210:0);
  a100=(a100+a100);
  a168=(a112*a168);
  a168=(a98*a168);
  a168=(a101*a168);
  a168=(a193*a168);
  a168=(a100*a168);
  a168=(a209*a168);
  a210=(a210-a168);
  a210=(a189*a210);
  a82=(a82+a82);
  a210=(a210/a82);
  a168=(a89*a210);
  a211=(a208-a168);
  a212=(a109*a179);
  a206=(a206/a95);
  a213=(a93*a206);
  a212=(a212+a213);
  a211=(a211-a212);
  a211=(a81*a211);
  a116=(a116+a211);
  a211=(a107*a200);
  a213=(a104*a196);
  a214=(a108*a200);
  a213=(a213-a214);
  a214=(a85*a213);
  a211=(a211+a214);
  a168=(a168-a208);
  a168=(a168+a212);
  a212=(a5*a168);
  a211=(a211-a212);
  a212=(a83*a206);
  a208=(a196+a212);
  a214=(a5*a208);
  a211=(a211+a214);
  a214=(a73*a211);
  a116=(a116+a214);
  a214=(a108*a188);
  a215=(a86*a213);
  a214=(a214+a215);
  a214=(a214+a168);
  a214=(a214-a208);
  a214=(a79*a214);
  a116=(a116+a214);
  a212=(a81*a212);
  a116=(a116+a212);
  a212=(a80*a213);
  a196=(a107*a196);
  a212=(a212-a196);
  a196=(a91*a206);
  a200=(a200+a196);
  a214=(a5*a200);
  a212=(a212+a214);
  a78=(a78+a78);
  a210=(a78*a210);
  a94=(a94+a94);
  a70=(a94*a70);
  a214=(a210-a70);
  a179=(a106*a179);
  a206=(a84*a206);
  a179=(a179+a206);
  a214=(a214+a179);
  a206=(a5*a214);
  a212=(a212+a206);
  a206=(a76*a212);
  a116=(a116-a206);
  a116=(a176*a116);
  a182=(a182-a116);
  a116=cos(a6);
  a206=(a76*a203);
  a208=(a73*a198);
  a206=(a206-a208);
  a196=(a81*a196);
  a206=(a206+a196);
  a196=(a76*a211);
  a206=(a206+a196);
  a210=(a210-a70);
  a210=(a210+a179);
  a210=(a81*a210);
  a206=(a206+a210);
  a188=(a104*a188);
  a213=(a75*a213);
  a188=(a188+a213);
  a188=(a188-a200);
  a188=(a188-a214);
  a188=(a79*a188);
  a206=(a206+a188);
  a188=(a73*a212);
  a206=(a206+a188);
  a206=(a116*a206);
  a182=(a182+a206);
  a206=sin(a6);
  a188=2.4999962702725869e+00;
  a40=(a54*a40);
  a214=(a191*a40);
  a214=(a214*a57);
  a214=(a32*a214);
  a214=(a188*a214);
  a214=(a71?a214:0);
  a59=(a59+a59);
  a200=(a66*a40);
  a200=(a55*a200);
  a200=(a60*a200);
  a200=(a193*a200);
  a200=(a59*a200);
  a200=(a188*a200);
  a214=(a214-a200);
  a214=(a189*a214);
  a56=(a56+a56);
  a214=(a214/a56);
  a200=(a48*a214);
  a213=(a50/a53);
  a210=(a0*a58);
  a179=2.4999962702725870e-01;
  a70=(a22*a40);
  a70=(a179*a70);
  a40=(a61*a40);
  a68=(a68/a69);
  a196=(a40*a68);
  a196=(a175*a196);
  a196=(a22*a196);
  a196=(a67?a196:0);
  a70=(a70+a196);
  a196=(a40/a69);
  a196=(a175*a196);
  a196=(a22*a196);
  a196=(a36*a196);
  a196=(-a196);
  a196=(a67?a196:0);
  a70=(a70+a196);
  a64=(a64/a65);
  a196=(a40*a64);
  a196=(a22*a196);
  a208=casadi_sign(a62);
  a196=(a196*a208);
  a196=(-a196);
  a196=(a63?a196:0);
  a70=(a70+a196);
  a40=(a40/a65);
  a40=(a22*a40);
  a62=casadi_sign(a62);
  a40=(a40*a62);
  a40=(-a40);
  a40=(a63?a40:0);
  a70=(a70+a40);
  a40=(a1*a70);
  a210=(a210+a40);
  a40=(a213*a210);
  a196=(a53+a53);
  a40=(a40/a196);
  a168=(a48*a40);
  a200=(a200-a168);
  a210=(a210/a53);
  a168=(a45*a210);
  a200=(a200-a168);
  a200=(a5*a200);
  a168=(a47*a210);
  a168=(a5*a168);
  a200=(a200+a168);
  a200=(a206*a200);
  a182=(a182-a200);
  a200=cos(a6);
  a46=(a46+a46);
  a214=(a46*a214);
  a52=(a52+a52);
  a40=(a52*a40);
  a214=(a214-a40);
  a40=(a49*a210);
  a214=(a214+a40);
  a214=(a5*a214);
  a210=(a51*a210);
  a210=(a5*a210);
  a214=(a214-a210);
  a214=(a200*a214);
  a182=(a182+a214);
  a214=sin(a6);
  a210=1.1764714944699486e+00;
  a4=(a17*a4);
  a40=(a191*a4);
  a40=(a40*a20);
  a40=(a32*a40);
  a40=(a210*a40);
  a40=(a41?a40:0);
  a23=(a23+a23);
  a168=(a34*a4);
  a168=(a18*a168);
  a168=(a25*a168);
  a168=(a193*a168);
  a168=(a23*a168);
  a168=(a210*a168);
  a40=(a40-a168);
  a40=(a189*a40);
  a9=(a9+a9);
  a40=(a40/a9);
  a168=(a11*a40);
  a215=(a13/a16);
  a0=(a0*a43);
  a216=1.1764714944699485e-01;
  a217=(a22*a4);
  a217=(a216*a217);
  a4=(a27*a4);
  a37=(a37/a39);
  a218=(a4*a37);
  a218=(a175*a218);
  a218=(a22*a218);
  a218=(a35?a218:0);
  a217=(a217+a218);
  a218=(a4/a39);
  a218=(a175*a218);
  a218=(a22*a218);
  a218=(a36*a218);
  a218=(-a218);
  a218=(a35?a218:0);
  a217=(a217+a218);
  a31=(a31/a33);
  a218=(a4*a31);
  a218=(a22*a218);
  a219=casadi_sign(a28);
  a218=(a218*a219);
  a218=(-a218);
  a218=(a30?a218:0);
  a217=(a217+a218);
  a4=(a4/a33);
  a4=(a22*a4);
  a28=casadi_sign(a28);
  a4=(a4*a28);
  a4=(-a4);
  a4=(a30?a4:0);
  a217=(a217+a4);
  a4=(a1*a217);
  a0=(a0+a4);
  a4=(a215*a0);
  a218=(a16+a16);
  a4=(a4/a218);
  a220=(a11*a4);
  a168=(a168-a220);
  a0=(a0/a16);
  a220=(a7*a0);
  a168=(a168-a220);
  a168=(a5*a168);
  a220=(a10*a0);
  a220=(a5*a220);
  a168=(a168+a220);
  a168=(a214*a168);
  a182=(a182-a168);
  a6=cos(a6);
  a8=(a8+a8);
  a40=(a8*a40);
  a15=(a15+a15);
  a4=(a15*a4);
  a40=(a40-a4);
  a4=(a12*a0);
  a40=(a40+a4);
  a40=(a5*a40);
  a0=(a14*a0);
  a0=(a5*a0);
  a40=(a40-a0);
  a40=(a6*a40);
  a182=(a182+a40);
  if (res[1]!=0) res[1][8]=a182;
  a182=(a169*a21);
  a141=(a141*a171);
  a171=(a22*a141);
  a173=(a173*a171);
  a26=(a26*a141);
  a155=(a26*a155);
  a155=(a175*a155);
  a155=(a22*a155);
  a155=(a154?a155:0);
  a173=(a173+a155);
  a156=(a26/a156);
  a156=(a175*a156);
  a156=(a22*a156);
  a156=(a36*a156);
  a156=(-a156);
  a154=(a154?a156:0);
  a173=(a173+a154);
  a136=(a26*a136);
  a136=(a22*a136);
  a136=(a136*a178);
  a136=(-a136);
  a136=(a29?a136:0);
  a173=(a173+a136);
  a26=(a26/a152);
  a26=(a22*a26);
  a26=(a26*a145);
  a26=(-a26);
  a29=(a29?a26:0);
  a173=(a173+a29);
  a29=(a2*a173);
  a182=(a182+a29);
  a29=(a182/a139);
  a26=(a135*a29);
  a145=(a130*a26);
  a152=(a127*a29);
  a136=(a120*a152);
  a145=(a145-a136);
  a136=(a124*a145);
  a178=(a147*a152);
  a136=(a136+a178);
  a178=(a121*a136);
  a154=(a129*a145);
  a147=(a147*a26);
  a154=(a154-a147);
  a147=(a118*a154);
  a178=(a178-a147);
  a21=(a3*a21);
  a147=(a1*a173);
  a21=(a21+a147);
  a139=(a21/a139);
  a135=(a135*a139);
  a147=(a125*a135);
  a178=(a178+a147);
  a147=(a149*a26);
  a156=(a146*a152);
  a155=(a150*a26);
  a156=(a156-a155);
  a129=(a129*a156);
  a147=(a147+a129);
  a129=(a191*a141);
  a129=(a129*a134);
  a129=(a32*a129);
  a129=(a190*a129);
  a19=(a19?a129:0);
  a153=(a153*a141);
  a142=(a142*a153);
  a144=(a144*a142);
  a144=(a193*a144);
  a143=(a143*a144);
  a190=(a190*a143);
  a19=(a19-a190);
  a19=(a189*a19);
  a19=(a19/a126);
  a133=(a133*a19);
  a194=(a194*a182);
  a195=(a195*a21);
  a194=(a194+a195);
  a194=(a194/a184);
  a140=(a140*a194);
  a184=(a133-a140);
  a151=(a151*a29);
  a137=(a137*a139);
  a151=(a151+a137);
  a184=(a184+a151);
  a137=(a5*a184);
  a147=(a147-a137);
  a127=(a127*a139);
  a137=(a152+a127);
  a195=(a5*a137);
  a147=(a147+a195);
  a195=(a121*a147);
  a178=(a178+a195);
  a123=(a123*a19);
  a138=(a138*a194);
  a194=(a123-a138);
  a148=(a148*a29);
  a128=(a128*a139);
  a148=(a148+a128);
  a194=(a194+a148);
  a194=(a125*a194);
  a178=(a178+a194);
  a146=(a146*a145);
  a120=(a120*a156);
  a146=(a146+a120);
  a26=(a26+a135);
  a146=(a146-a26);
  a123=(a123-a138);
  a123=(a123+a148);
  a146=(a146-a123);
  a146=(a79*a146);
  a178=(a178+a146);
  a124=(a124*a156);
  a149=(a149*a152);
  a124=(a124-a149);
  a26=(a5*a26);
  a124=(a124+a26);
  a123=(a5*a123);
  a124=(a124+a123);
  a123=(a118*a124);
  a178=(a178+a123);
  a38=(a38*a178);
  a178=-4.8780487804877992e-01;
  a123=(a178*a157);
  a26=(a157*a123);
  a149=(a178*a42);
  a152=(a42*a149);
  a26=(a26+a152);
  a26=(a158*a26);
  a192=(a192*a26);
  a26=(a157*a149);
  a152=(a42*a123);
  a26=(a26-a152);
  a158=(a158*a26);
  a186=(a186*a158);
  a192=(a192+a186);
  a186=(a121*a154);
  a158=(a118*a136);
  a186=(a186+a158);
  a140=(a140-a133);
  a140=(a140-a151);
  a140=(a125*a140);
  a186=(a186+a140);
  a118=(a118*a147);
  a186=(a186+a118);
  a150=(a150*a145);
  a130=(a130*a156);
  a150=(a150+a130);
  a150=(a150+a184);
  a150=(a150-a137);
  a150=(a79*a150);
  a186=(a186+a150);
  a125=(a125*a127);
  a186=(a186+a125);
  a121=(a121*a124);
  a186=(a186-a121);
  a185=(a185*a186);
  a192=(a192+a185);
  a38=(a38-a192);
  a169=(a169*a99);
  a97=(a97*a170);
  a170=(a22*a97);
  a202=(a202*a170);
  a102=(a102*a97);
  a114=(a102*a114);
  a114=(a175*a114);
  a114=(a22*a114);
  a114=(a113?a114:0);
  a202=(a202+a114);
  a115=(a102/a115);
  a115=(a175*a115);
  a115=(a22*a115);
  a115=(a36*a115);
  a115=(-a115);
  a113=(a113?a115:0);
  a202=(a202+a113);
  a110=(a102*a110);
  a110=(a22*a110);
  a110=(a110*a197);
  a110=(-a110);
  a110=(a92?a110:0);
  a202=(a202+a110);
  a102=(a102/a111);
  a102=(a22*a102);
  a102=(a102*a103);
  a102=(-a102);
  a92=(a92?a102:0);
  a202=(a202+a92);
  a92=(a2*a202);
  a169=(a169+a92);
  a92=(a169/a95);
  a102=(a91*a92);
  a103=(a86*a102);
  a111=(a83*a92);
  a110=(a75*a111);
  a103=(a103-a110);
  a110=(a85*a103);
  a197=(a105*a102);
  a110=(a110-a197);
  a197=(a76*a110);
  a113=(a80*a103);
  a105=(a105*a111);
  a113=(a113+a105);
  a105=(a73*a113);
  a197=(a197+a105);
  a204=(a204*a169);
  a99=(a3*a99);
  a169=(a1*a202);
  a99=(a99+a169);
  a205=(a205*a99);
  a204=(a204+a205);
  a204=(a204/a207);
  a96=(a96*a204);
  a207=(a191*a97);
  a207=(a207*a90);
  a207=(a32*a207);
  a207=(a209*a207);
  a117=(a117?a207:0);
  a112=(a112*a97);
  a98=(a98*a112);
  a101=(a101*a98);
  a101=(a193*a101);
  a100=(a100*a101);
  a209=(a209*a100);
  a117=(a117-a209);
  a117=(a189*a117);
  a117=(a117/a82);
  a89=(a89*a117);
  a82=(a96-a89);
  a109=(a109*a92);
  a99=(a99/a95);
  a93=(a93*a99);
  a109=(a109+a93);
  a82=(a82-a109);
  a82=(a81*a82);
  a197=(a197+a82);
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
  a197=(a197+a96);
  a108=(a108*a103);
  a86=(a86*a93);
  a108=(a108+a86);
  a108=(a108+a89);
  a108=(a108-a109);
  a108=(a79*a108);
  a197=(a197+a108);
  a83=(a81*a83);
  a197=(a197+a83);
  a80=(a80*a93);
  a107=(a107*a111);
  a80=(a80-a107);
  a91=(a91*a99);
  a102=(a102+a91);
  a107=(a5*a102);
  a80=(a80+a107);
  a78=(a78*a117);
  a94=(a94*a204);
  a204=(a78-a94);
  a106=(a106*a92);
  a84=(a84*a99);
  a106=(a106+a84);
  a204=(a204+a106);
  a84=(a5*a204);
  a80=(a80+a84);
  a84=(a76*a80);
  a197=(a197-a84);
  a176=(a176*a197);
  a38=(a38-a176);
  a176=(a76*a113);
  a197=(a73*a110);
  a176=(a176-a197);
  a91=(a81*a91);
  a176=(a176+a91);
  a76=(a76*a82);
  a176=(a176+a76);
  a78=(a78-a94);
  a78=(a78+a106);
  a81=(a81*a78);
  a176=(a176+a81);
  a104=(a104*a103);
  a75=(a75*a93);
  a104=(a104+a75);
  a104=(a104-a102);
  a104=(a104-a204);
  a104=(a79*a104);
  a176=(a176+a104);
  a73=(a73*a80);
  a176=(a176+a73);
  a116=(a116*a176);
  a38=(a38+a116);
  a54=(a54*a44);
  a44=(a191*a54);
  a44=(a44*a57);
  a44=(a32*a44);
  a44=(a188*a44);
  a71=(a71?a44:0);
  a66=(a66*a54);
  a55=(a55*a66);
  a60=(a60*a55);
  a60=(a193*a60);
  a59=(a59*a60);
  a188=(a188*a59);
  a71=(a71-a188);
  a71=(a189*a71);
  a71=(a71/a56);
  a56=(a48*a71);
  a58=(a3*a58);
  a188=(a22*a54);
  a179=(a179*a188);
  a61=(a61*a54);
  a68=(a61*a68);
  a68=(a175*a68);
  a68=(a22*a68);
  a68=(a67?a68:0);
  a179=(a179+a68);
  a69=(a61/a69);
  a69=(a175*a69);
  a69=(a22*a69);
  a69=(a36*a69);
  a69=(-a69);
  a67=(a67?a69:0);
  a179=(a179+a67);
  a64=(a61*a64);
  a64=(a22*a64);
  a64=(a64*a208);
  a64=(-a64);
  a64=(a63?a64:0);
  a179=(a179+a64);
  a61=(a61/a65);
  a61=(a22*a61);
  a61=(a61*a62);
  a61=(-a61);
  a63=(a63?a61:0);
  a179=(a179+a63);
  a63=(a1*a179);
  a58=(a58+a63);
  a213=(a213*a58);
  a213=(a213/a196);
  a48=(a48*a213);
  a56=(a56-a48);
  a58=(a58/a53);
  a45=(a45*a58);
  a56=(a56-a45);
  a56=(a5*a56);
  a47=(a47*a58);
  a47=(a5*a47);
  a56=(a56+a47);
  a206=(a206*a56);
  a38=(a38-a206);
  a46=(a46*a71);
  a52=(a52*a213);
  a46=(a46-a52);
  a49=(a49*a58);
  a46=(a46+a49);
  a46=(a5*a46);
  a51=(a51*a58);
  a51=(a5*a51);
  a46=(a46-a51);
  a200=(a200*a46);
  a38=(a38+a200);
  a17=(a17*a165);
  a191=(a191*a17);
  a191=(a191*a20);
  a32=(a32*a191);
  a32=(a210*a32);
  a41=(a41?a32:0);
  a34=(a34*a17);
  a18=(a18*a34);
  a25=(a25*a18);
  a193=(a193*a25);
  a23=(a23*a193);
  a210=(a210*a23);
  a41=(a41-a210);
  a189=(a189*a41);
  a189=(a189/a9);
  a9=(a11*a189);
  a3=(a3*a43);
  a43=(a22*a17);
  a216=(a216*a43);
  a27=(a27*a17);
  a37=(a27*a37);
  a37=(a175*a37);
  a37=(a22*a37);
  a37=(a35?a37:0);
  a216=(a216+a37);
  a39=(a27/a39);
  a175=(a175*a39);
  a175=(a22*a175);
  a36=(a36*a175);
  a36=(-a36);
  a35=(a35?a36:0);
  a216=(a216+a35);
  a31=(a27*a31);
  a31=(a22*a31);
  a31=(a31*a219);
  a31=(-a31);
  a31=(a30?a31:0);
  a216=(a216+a31);
  a27=(a27/a33);
  a22=(a22*a27);
  a22=(a22*a28);
  a22=(-a22);
  a30=(a30?a22:0);
  a216=(a216+a30);
  a1=(a1*a216);
  a3=(a3+a1);
  a215=(a215*a3);
  a215=(a215/a218);
  a11=(a11*a215);
  a9=(a9-a11);
  a3=(a3/a16);
  a7=(a7*a3);
  a9=(a9-a7);
  a9=(a5*a9);
  a10=(a10*a3);
  a10=(a5*a10);
  a9=(a9+a10);
  a214=(a214*a9);
  a38=(a38-a214);
  a8=(a8*a189);
  a15=(a15*a215);
  a8=(a8-a15);
  a12=(a12*a3);
  a8=(a8+a12);
  a8=(a5*a8);
  a14=(a14*a3);
  a5=(a5*a14);
  a8=(a8-a5);
  a6=(a6*a8);
  a38=(a38+a6);
  if (res[1]!=0) res[1][9]=a38;
  a38=cos(a72);
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
  a8=(a38*a8);
  a5=sin(a72);
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
  a177=sin(a72);
  a3=(a122*a183);
  a12=(a119*a181);
  a3=(a3+a12);
  a12=(a119*a187);
  a3=(a3+a12);
  a12=(a122*a201);
  a3=(a3-a12);
  a3=(a177*a3);
  a8=(a8-a3);
  a3=cos(a72);
  a181=(a122*a181);
  a183=(a119*a183);
  a181=(a181-a183);
  a187=(a122*a187);
  a181=(a181+a187);
  a201=(a119*a201);
  a181=(a181+a201);
  a181=(a3*a181);
  a8=(a8+a181);
  a181=sin(a72);
  a201=(a77*a198);
  a187=(a74*a203);
  a201=(a201+a187);
  a187=(a74*a211);
  a201=(a201+a187);
  a187=(a77*a212);
  a201=(a201-a187);
  a201=(a181*a201);
  a8=(a8-a201);
  a72=cos(a72);
  a203=(a77*a203);
  a198=(a74*a198);
  a203=(a203-a198);
  a211=(a77*a211);
  a203=(a203+a211);
  a212=(a74*a212);
  a203=(a203+a212);
  a203=(a72*a203);
  a8=(a8+a203);
  if (res[1]!=0) res[1][10]=a8;
  a8=(a164*a123);
  a203=(a2*a123);
  a8=(a8-a203);
  a203=(a162*a8);
  a212=(a159*a123);
  a203=(a203-a212);
  a160=(a178*a160);
  a203=(a203+a160);
  a160=(a161*a149);
  a203=(a203-a160);
  a38=(a38*a203);
  a178=(a178*a167);
  a161=(a161*a123);
  a178=(a178-a161);
  a2=(a2*a149);
  a164=(a164*a149);
  a2=(a2-a164);
  a162=(a162*a2);
  a178=(a178+a162);
  a159=(a159*a149);
  a178=(a178+a159);
  a5=(a5*a178);
  a38=(a38-a5);
  a5=(a122*a154);
  a178=(a119*a136);
  a5=(a5+a178);
  a178=(a119*a147);
  a5=(a5+a178);
  a178=(a122*a124);
  a5=(a5-a178);
  a177=(a177*a5);
  a38=(a38-a177);
  a136=(a122*a136);
  a154=(a119*a154);
  a136=(a136-a154);
  a122=(a122*a147);
  a136=(a136+a122);
  a119=(a119*a124);
  a136=(a136+a119);
  a3=(a3*a136);
  a38=(a38+a3);
  a3=(a77*a110);
  a136=(a74*a113);
  a3=(a3+a136);
  a136=(a74*a82);
  a3=(a3+a136);
  a136=(a77*a80);
  a3=(a3-a136);
  a181=(a181*a3);
  a38=(a38-a181);
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
  a72=(a166*a180);
  a113=(a163*a172);
  a72=(a72-a113);
  a6=(a42*a6);
  a14=(a157*a14);
  a6=(a6+a14);
  a6=(a79*a6);
  a6=(a72+a6);
  a14=(a132*a174);
  a6=(a6+a14);
  a14=(a88*a199);
  a6=(a6+a14);
  a70=(a50*a70);
  a6=(a6+a70);
  a217=(a13*a217);
  a6=(a6+a217);
  if (res[1]!=0) res[1][13]=a6;
  a6=(a166*a123);
  a217=(a163*a149);
  a6=(a6-a217);
  a42=(a42*a8);
  a157=(a157*a2);
  a42=(a42+a157);
  a79=(a79*a42);
  a79=(a6+a79);
  a132=(a132*a173);
  a79=(a79+a132);
  a88=(a88*a202);
  a79=(a79+a88);
  a50=(a50*a179);
  a79=(a79+a50);
  a13=(a13*a216);
  a79=(a79+a13);
  if (res[1]!=0) res[1][14]=a79;
  if (res[1]!=0) res[1][15]=a38;
  a180=(a166*a180);
  a72=(a72-a180);
  a172=(a163*a172);
  a72=(a72+a172);
  a174=(a131*a174);
  a72=(a72+a174);
  a199=(a87*a199);
  a72=(a72+a199);
  if (res[1]!=0) res[1][16]=a72;
  a166=(a166*a123);
  a6=(a6-a166);
  a163=(a163*a149);
  a6=(a6+a163);
  a131=(a131*a173);
  a6=(a6+a131);
  a87=(a87*a202);
  a6=(a6+a87);
  if (res[1]!=0) res[1][17]=a6;
  if (res[2]!=0) res[2][0]=a24;
  if (res[2]!=0) res[2][1]=a24;
  if (res[2]!=0) res[2][2]=a24;
  if (res[2]!=0) res[2][3]=a24;
  if (res[2]!=0) res[2][4]=a24;
  if (res[2]!=0) res[2][5]=a24;
  if (res[2]!=0) res[2][6]=a24;
  if (res[2]!=0) res[2][7]=a24;
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int model_2021_03_22_10301943_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
