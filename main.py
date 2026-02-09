import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit
from data import *

def damped_sine(t, A, gamma, omega, phi, C):
    """Damped sine with constant offset"""
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi) + C

def plot_offset_symmetry(data, dt, C, title="Offset symmetry check"):
    data = np.asarray(data, dtype=float)
    t = np.arange(len(data)) * dt

    # Remove offset
    centered = data - C

    # Flipped version
    flipped = -centered

    plt.figure()
    plt.plot(t, centered, 'o', label="Data − offset")
    plt.plot(t, flipped, 'x', label="Flipped (−1 ×)")
    plt.axhline(0, color='k', linewidth=0.8, linestyle='--')

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (offset removed)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_with_damped_fit(
    data,
    dt,
    xlabel="Time (s)",
    ylabel="Displacement (cm)",
    title=None
):
    data = np.asarray(data, dtype=float)
    t = np.arange(len(data)) * dt

    # ---- initial parameter guesses ----
    A0 = 0.5 * (np.max(data) - np.min(data))
    gamma0 = 1e-4
    omega0 = 2 * np.pi * 5 / t[-1]   # assume a few oscillations
    phi0 = 0.0
    C0 = np.mean(data)

    p0 = [A0, gamma0, omega0, phi0, C0]

    # ---- curve fit ----
    params, covariance = curve_fit(
        damped_sine,
        t,
        data,
        p0=p0,
        maxfev=50000
    )
    A, gamma, omega, phi, C = params

    print("\nFitted parameters:")
    print(f"Amplitude A     = {A:.6f}")
    print(f"Damping gamma   = {gamma:.6e}")
    print(f"Angular freq ω  = {omega:.6f}")
    print(f"Phase φ         = {phi:.6f}")
    print(f"Offset C        = {C:.6f}")

    #Errors ==============================
    A_err = np.sqrt(np.diag(covariance)[0])
    gamma_err = np.sqrt(np.diag(covariance)[1])
    omega_err = np.sqrt(np.diag(covariance)[2])
    phi_err = np.sqrt(np.diag(covariance)[3])
    C_err = np.sqrt(np.diag(covariance)[4])

    print("\nErrors of Fitted parameters:")
    print(f"Error: Amplitude A     = {A_err:.6f}")
    print(f"Error: Damping gamma   = {gamma_err:.6e}")
    print(f"Error: Angular freq ω  = {omega_err:.6f}")
    print(f"Error: Phase φ         = {phi_err:.6f}")
    print(f"Error: Offset C        = {C_err:.6f}")
    # ---- smooth curve for plotting ----
    t_fit = np.linspace(t[0], t[-1], 2000)
    y_fit = damped_sine(t_fit, *params)

    # ---- plot ----
    plt.figure()
    plt.plot(t, data, 'x', label="Data")
    plt.plot(t_fit, y_fit, label="Damped sine fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plot_offset_symmetry(data, dt, C)

    return params


def plot_dataset(
    data,
    dt,
    xlabel="Time (s)",
    ylabel="Displacement (cm)",
    title=None
):
    data = np.asarray(data, dtype=float)
    t = np.arange(len(data)) * dt

    plt.figure()
    plt.plot(t, data, 'x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def data_averager(l1, l2):
    if len(l1) != len(l2):
        return IOError
    else:
        return [int(((i+j)/2)*1000)/1000 for i, j in zip(l1, l2)]

def error_propagation(a, da,y, dy, d,dd, m2, dm2, L, dL, T, dT):
   G= 6.67430 * 10**-11
   return G * np.sqrt((2 * da/a)**2 + (dy/y)**2 + (dd/d)**2 + (dm2/m2)**2 + (dL/L)**2 + (2*dT/T)**2)

def calculate_G2(ca,cy,cd,cm2,cL,cT):
    cr = r
    theta = 1/2*np.arctan(cy/cL)
    s = cd*np.sin(theta)
    beta = (ca*(ca-s)**2)/((ca**2+(4*(cd**2)))**(3/2))
    numerator = 4*(np.pi**2)*(ca-s)**2*theta
    denominator = cd*(1-beta)*cT**2*cm2
    ex = (cd**2+(2/5)*cr**2)
    return (numerator/denominator)*ex
if __name__ == '__main__':
    plot_with_damped_fit(i2,15)

    names = ["Daniel", "Ariv", "Rumen", "Idris"]
    G2_s = []
    DG2_s = []
    for i in range(4):
        current_G2_error = error_propagation(
            a[i], da[i],
            y[i], dy[i],
            d, dd,
            m2, dm2,
            L[i], dL[i],
            T[i], dT[i]
        )
        current_G2 = calculate_G2( a[i], y[i], d, m2,L[i],T[i])

        print("=" * 50)
        print(f"{names[i]:^50}")  # centered name
        print("=" * 50)

        print("Input values:")
        print(f"  a   = {a[i]:>10}   ± {da[i]}")
        print(f"  y   = {y[i]:>10}   ± {dy[i]}")
        print(f"  d   = {d:>10}   ± {dd}")
        print(f"  m2  = {m2:>10}   ± {dm2}")
        print(f"  L   = {L[i]:>10}   ± {dL[i]}")
        print(f"  T   = {T[i]:>10}   ± {dT[i]}")

        print("\nResult:")
        print(f"  Absolute error = {current_G2_error:.6e}")
        print("\nBIG G:")
        print(f"  G_2 = {current_G2:.6e}")

        #Save
        G2_s.append(current_G2)
        DG2_s.append(current_G2_error)
    print("=" * 50)
    print(f"{"weighted errors:":^50}")
    G_avg = 0


    sum_DG2 = 0
    for i in range (4):

        G_avg += G2_s[i]/DG2_s[i]**2
        sum_DG2 += 1/DG2_s[i]**2
    G_avg = G_avg/sum_DG2
    SG = np.sqrt(1/sum_DG2)
    print(f"  G2   = {G_avg:>10}   ± {SG}")
