from .PlanarFlow import InvertedPlanarFlow
from .RadialFlow import InvertedRadialFlow
from .IdentityFlow import IdentityFlow
from .AffineFlow import AffineFlow

FLOWS = {
    'planar': InvertedPlanarFlow,
    'radial': InvertedRadialFlow,
    'identity': IdentityFlow,
    'affine': AffineFlow
}

