from typing import List, Optional

class AtomConstraints:
    """Manages constrained and unconstrained atom indices for harmonic calculations."""

    def __init__(
        self, num_atoms: int, constrain_index: Optional[List[int]] = None
    ):
        self.num_atoms = num_atoms
        self.constrain_index = constrain_index or []

        if constrain_index is not None:
            self.num_constrained = len(constrain_index)
            self.constrained = True
        else:
            self.constrained = False
            self.num_constrained = 0

        self.unconstrained_index = [
            i for i in range(num_atoms) if i not in self.constrain_index
        ]
        self.num_eff_atoms = len(self.unconstrained_index)

        assert self.num_eff_atoms == num_atoms - self.num_constrained

    @property
    def force_indices(self) -> List[int]:
        """Generate flat indices for forces on unconstrained atoms (3 per atom)."""
        indices = []
        for i in self.unconstrained_index:
            for d in range(3):
                indices.append(i * 3 + d)
        return indices
