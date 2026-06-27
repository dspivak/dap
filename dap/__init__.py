"""dap: dynamic-algebra-potentials, executable core.

A Python implementation of the principal constructions of the paper
``dynamic-algebra-potentials.tex``: reactive vector spaces, smooth adaptive
arrangements, the smooth polynomial interpretation, and the two dynamics
functors ``Phiconf`` (configuration / descent) and ``Phiphase`` (phase / Hamilton)
of cor.functor, both built from a shared interpretation plus an integrator.
"""

from .rvect import ReactiveVectorSpace
from .arrangement import SmoothArrangement
from .polynomial import Yon, Cot, DirichletProduct, PolyMap
from .org import OrgMorphism
from .integrator import (
    Integrator,
    Integrator2,
    IntegratorK,
    configuration_integrator,
    damped_phase_integrator,
    gyro_phase_integrator,
    phase_integrator,
)
from .interpretation import smooth_interpretation
from .functors import (
    Phi, Phiconf, Phidamped, Phigyro, Phiphase, Phirk4, Phirk4gyro, cot_object, cot_map,
)
from .org2 import OrgMorphism2, org2_from_integrator
from .orgK import OrgMorphismK, orgK_from_integrator
from .leapfrog import Phileap, leapfrog_integrator
from .rk4 import rk4_gyro_integrator, rk4_integrator
from . import functors
from . import wiring
from . import learning

# Extensions (beyond the paper): reuse the functorial core above but add content
# not in the paper's formal development. See the README "Extensions" section and
# each module's header. ``Phidamped`` / ``damped_phase_integrator`` are extensions too.
from . import system_id
from . import pinn
from . import gyroscope
from . import gyroscope_faithful

__all__ = [
    "ReactiveVectorSpace",
    "SmoothArrangement",
    "Yon",
    "Cot",
    "DirichletProduct",
    "PolyMap",
    "OrgMorphism",
    "Integrator",
    "Integrator2",
    "IntegratorK",
    "configuration_integrator",
    "damped_phase_integrator",
    "gyro_phase_integrator",
    "phase_integrator",
    "smooth_interpretation",
    "Phi",
    "Phiconf",
    "Phidamped",
    "Phigyro",
    "Phiphase",
    "Phirk4",
    "Phirk4gyro",
    "Phileap",
    "leapfrog_integrator",
    "rk4_integrator",
    "rk4_gyro_integrator",
    "OrgMorphism2",
    "org2_from_integrator",
    "OrgMorphismK",
    "orgK_from_integrator",
    "cot_object",
    "cot_map",
    "functors",
    "wiring",
    "learning",
    # extensions (beyond the paper):
    "system_id",
    "pinn",
    "gyroscope",
    "gyroscope_faithful",
]
