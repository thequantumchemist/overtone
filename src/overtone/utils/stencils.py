
def get_stencil(stencil):
    if stencil == 0:
        displacements = [-1, 1]
        coefficients = [-1, 1]
        dfac = 2
    elif stencil == 1:  # Equivalent to the one Bowman uses.
        displacements = [2, 1, -1, -2]
        coefficients = [-1, 8, -8, 1]
        dfac = 12
    elif stencil == 2:
        displacements = [-3, -2, -1, 1, 2, 3]
        coefficients = [-1, 9, -45, 45, -9, 1]
        dfac = 60
    elif stencil == 3:
        displacements = [-4, -3, -2, -1, 1, 2, 3, 4]
        coefficients = [3, -32, 168, -672, 672, -168, 32, -3]
        dfac = 840
    return displacements, coefficients, dfac