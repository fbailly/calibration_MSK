import numpy as np
import matplotlib.pyplot as plt

# model constants
C = np.matrix([0.2, 0.995, 0.25])
kt = 35.0
B = np.matrix([[0.815, 1.055, 0.162, 0.063], [0.433, 0.717, -0.03, 0.2], [0.1, 1.0, 0.354, 0.0]]).T
kpe = 4.0
e0 = 0.6
D = np.matrix([-0.318, -8.149, -0.374, 0.886])

def plot_muscle_char(f_tendon, f_act, f_pas, f_v):
    vm_n = np.expand_dims(np.linspace(-1, 1, num=1000), 1)
    lm_n = np.expand_dims(np.linspace(0.2, 1.7, num=1000), 1)
    lt_n = np.expand_dims(np.linspace(1, 1.05, num=1000), 1)

    # f_act normalization f_act(1) should be 1?
    # print(f_act(1))

    plt.subplot(131)
    plt.plot(lt_n, f_tendon(lt_n))
    plt.xlabel('norm tendon len')
    plt.ylabel('tendon force')
    plt.subplot(132)
    plt.plot(lm_n, f_act(lm_n), label='active')
    plt.plot(lm_n, f_pas(lm_n), label='passive')
    plt.xlabel('norm muscle len')
    plt.ylabel('muscle force')
    plt.legend()
    plt.subplot(133)
    plt.plot(vm_n, f_v(vm_n))
    plt.xlabel('norm muscle vel')
    plt.ylabel('muscle force')
    plt.suptitle('Hill-Degroote muscle characteristics')
    plt.tight_layout()
    plt.show()