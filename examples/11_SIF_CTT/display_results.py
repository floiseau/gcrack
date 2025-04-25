import numpy as np
import matplotlib.pyplot as plt

# Read the FEM results (Williams series interpolation)
res_w = np.loadtxt("sif_williams.csv", skiprows=1, delimiter=",")
# Extract FEM data (Williams series interpolation)
alpha_w = res_w[:, 0]
KI_w = res_w[:, 1]
KII_w = res_w[:, 2]
B_w = res_w[:, 3]

# Read the FEM results (i-integral)
res_i = np.loadtxt("sif_i-integral.csv", skiprows=1, delimiter=",")
# Extract FEM data (i-integral)
alpha_i = res_i[:, 0]
KI_i = res_i[:, 1]
KII_i = res_i[:, 2]
B_i = res_i[:, 3]


# Compute analytical solution
alpha_ana = np.linspace(0, 85, 128)
alp = np.deg2rad(alpha_ana)
KI_ana = np.cos(alp) ** 2
KII_ana = np.cos(alp) * np.sin(alp)
B_ana = np.cos(alp) * (np.tan(alp) ** 2 - 1)

# Display KI and KII
plt.figure()
plt.scatter(
    alpha_i, KI_i, c="tab:blue", marker="o", label=r"$K_{I}$ - FEM - I-integral"
)
plt.scatter(
    alpha_w, KI_w, c="tab:orange", marker="s", label=r"$K_{I}$ - FEM - Williams"
)
plt.plot(alpha_ana, KI_ana, c="tab:red", label=r"$K_{I}$ - ANA")
plt.scatter(
    alpha_i, KII_i, c="tab:blue", marker="o", label=r"$K_{II}$ - FEM - I-integral"
)
plt.scatter(
    alpha_w, KII_w, c="tab:orange", marker="s", label=r"$K_{II}$ - FEM - Williams"
)
plt.plot(alpha_ana, KII_ana, c="tab:red", label=r"$K_{II}$ - ANA")
plt.xlabel(r"Crack angle $\alpha$ (°)")
plt.ylabel(r"$\frac{K_p}{\sigma \sqrt{\pi a}}$")
plt.grid()
plt.legend()
plt.savefig("KI_KII.pdf")
plt.show()

# Display T-stress
plt.figure()
plt.scatter(
    alpha_i[:-1],
    B_i[:-1],
    c="tab:blue",
    marker="o",
    label=r"$B$ - FEM - I-integral",
)
plt.scatter(
    alpha_w[:-1],
    B_w[:-1],
    c="tab:orange",
    marker="s",
    label=r"$B$ - FEM - Williams",
)
plt.plot(alpha_ana, B_ana, c="tab:red", label=r"$B$ - ANA")
plt.xlabel(r"Crack angle $\alpha$ (°)")
plt.ylabel(r"$B = \frac{T \sqrt{\pi a}}{\sqrt{K_I^2 + K_{II}^2}}$")
plt.grid()
plt.legend()
plt.savefig("T_stress.pdf")
plt.show()
