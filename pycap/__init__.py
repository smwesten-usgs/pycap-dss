from .pycap_exceptions import PycapException
from .solutions import (
    ALL_DD_METHODS,
    ALL_DEPL_METHODS,
    CFD2GPM,
    GPM2CFD,
    SEC2DAY,
    _calc_deltaQ,
    glover_depletion,
    hunt_03_depletion,
    hunt_99_depletion,
    hunt_99_drawdown,
    sdf,
    theis_drawdown,
    walton_depletion,
    dudley_ward_lough_depletion,
    dudley_ward_lough_drawdown,
)
from .utilities import (
    Q2ts,
    create_timeseries_template,
)
from .wells import Well, WellResponse

# from . import _version
# __version__ = _version.get_versions()['version']
