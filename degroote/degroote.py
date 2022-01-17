import numpy as np

c1 = 0.2
c2 = 0.995
c3 = 0.25
kt = 35.0
B = np.matrix([[0.815, 1.055, 0.162, 0.063], [0.433, 0.717, -0.03, 0.2], [0.1, 1.0, 0.353, 0.0]]).T
kpe = 4.0
e0 = 0.6
D = np.matrix([-0.318, -8.149, -0.374, 0.886])


def f_tendon(lt_n):
    return c1*np.exp(kt*(lt_n-c2))-c3


def f_act(lm_n):
    return np.sum(np.multiply(B[0, :], np.exp(np.divide(-0.5*np.square(lm_n-B[1, :]), (B[2, :] + B[3, :]*lm_n)))))


def f_pas(lm_n):
    return (np.exp(kpe*(lm_n-1)/e0)-1)/(np.exp(kpe)-1)


def f_v(vm_n):
    return D[0, 0]*np.log(D[0, 1]*vm_n+D[0, 2]+np.sqrt((D[0, 1]*vm_n+D[0, 2])+1))+D[0, 3]


print(f_tendon(0))
print(f_act(0))
print(f_pas(0))
print(f_v(0))
