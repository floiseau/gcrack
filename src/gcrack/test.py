import numpy as np
import matplotlib.pyplot as plt
from optimization_solvers import F_AL92


def fF11(phi):
    m_ = phi / np.pi  # m_ in [-1,1] ==> phi in [-180,+180]
    return (
        1
        - ((3 * np.pi**2) / 8) * m_**2
        + (np.pi**2 - (5 * np.pi**4) / 128) * m_**4
        + (np.pi**2 / 9 - (11 * np.pi**4) / 72 + (119 * np.pi**6 / 15360))
        * m_**6
        + 5.07790 * m_**8
        - 2.88312 * m_**10
        - 0.0925 * m_**12
        + 2.996 * m_**14
        - 4.059 * m_**16
        + 1.63 * m_**18
        + 4.1 * m_**20
    )


def fF12(phi):
    m_ = phi / np.pi  # m_ in [-1,1] ==> phi in [-180,+180]
    return (
        (-3 * np.pi) / 2 * m_
        + ((10 * np.pi) / 3 + np.pi**3 / 16) * m_**3
        + (-2 * np.pi - (133 * np.pi**3) / 180 + (59 * np.pi**5) / 1280) * m_**5
        + 12.313906 * m_**7
        - 7.32433 * m_**9
        + 1.5793 * m_**11
        + 4.0216 * m_**13
        - 6.915 * m_**15
        + 4.21 * m_**17
        + 4.56 * m_**19
    )


def fF21(phi):
    m_ = phi / np.pi  # m_ in [-1,1] ==> phi in [-180,+180]
    return (
        (np.pi / 2) * m_
        - ((4 * np.pi) / 3 + np.pi**3 / 48) * m_**3
        + ((-2 * np.pi) / 3 + (13 * np.pi**3) / 30 - (59 * np.pi**5) / 3840)
        * m_**5
        - 6.176023 * m_**7
        + 4.44112 * m_**9
        - 1.5340 * m_**11
        - 2.07 * m_**13
        + 4.684 * m_**15
        - 3.95 * m_**17
        - 1.32 * m_**19
    )


def fF22(phi):
    m_ = phi / np.pi  # m_ in [-1,1] ==> phi in [-180,+180]
    return (
        1
        - (4 + (3 * np.pi**2) / 8) * m_**2
        + (8 / 3 + 29 * (np.pi**2) / 18 - (5 * np.pi**4) / 128) * m_**4
        + (
            -32 / 15
            - (4 * np.pi**2) / 9
            - (1159 * np.pi**4) / 7200
            + (119 * np.pi**6) / 15360
        )
        * m_**6
        + 10.58254 * m_**8
        - 4.78511 * m_**10
        - 1.8804 * m_**12
        + 7.28 * m_**14
        - 7.591 * m_**16
        + 0.25 * m_**18
        + 12.5 * m_**20
    )


phi = np.linspace(0, np.pi - np.pi / 100000)
f_fl = np.array(list(map(F_AL92, phi)))

f_dr = np.empty((len(phi), 2, 2))
f_dr[:, 0, 0] = fF11(phi)
f_dr[:, 0, 1] = fF12(phi)
f_dr[:, 1, 0] = fF21(phi)
f_dr[:, 1, 1] = fF22(phi)

fig, axs = plt.subplots(2, 2, sharex=True)
plt.suptitle(r"Amestoy-Leblond function $F_{pq}$")

axs[0, 0].plot(phi, f_fl[:, 0, 0], "r", label=r"FL")
axs[0, 0].plot(phi, f_dr[:, 0, 0], ":b", label=r"DR")
axs[0, 0].set_ylabel(r"$F_{11}$")
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(phi, f_fl[:, 0, 1], "r")
axs[0, 1].plot(phi, f_dr[:, 0, 1], ":b")
axs[0, 1].set_ylabel(r"$F_{12}$")
axs[0, 1].grid()

axs[1, 0].plot(phi, f_fl[:, 1, 0], "r")
axs[1, 0].plot(phi, f_dr[:, 1, 0], ":b")
axs[1, 0].set_xlabel(r"Kink angle $\varphi$")
axs[1, 0].set_ylabel(r"$F_{21}$")
axs[1, 0].grid()

axs[1, 1].plot(phi, f_fl[:, 1, 1], "r")
axs[1, 1].plot(phi, f_dr[:, 1, 1], ":b")
axs[1, 1].set_xlabel(r"Kink angle $\varphi$")
axs[1, 1].set_ylabel(r"$F_{22}$")
axs[1, 1].grid()

plt.show()
