from .PlanarFlow import InvertedPlanarFlow
from .RadialFlow import InvertedRadialFlow
from .IdentityFlow import IdentityFlow

FLOWS = {
    'planar': InvertedPlanarFlow,
    'radial': InvertedRadialFlow,
    'identity': IdentityFlow,
}

