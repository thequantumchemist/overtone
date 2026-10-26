import numpy as np

H = 6.62607015e-34  # Planck constant, J s
C_CM = 2.99792458e10  # speed of light, cm/s
AMU = 1.66053906660e-27  # kg

# ase.vibrations.Infrared intensities, (D/Angstrom)^2/amu -> km/mol
ASE_IR_TO_KM_PER_MOL = 42.255


def rotational_constants(moments):
    """Rotational constants (cm^-1) from principal moments of inertia (amu Angstrom^2)."""
    return H / (8 * np.pi**2 * C_CM * np.asarray(moments) * AMU * 1e-20)
