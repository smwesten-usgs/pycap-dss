import sys
import warnings
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.special as sps
from scipy.special import gammaln

from pycap.pycap_exceptions import PycapException


def _time_dist_error(funcname):
    """Function for trying to call both time and distance
    as arrays in a function
    """
    raise PycapException(
        "cannot have both time and distance as arrays\n"
        + f"in the {funcname} method.  Need to externally loop\n"
        + "over one of the arrays and pass the other"
    )


def _make_arrays(a):
    """private function to force values to
    arrays from lists or scalars
    """
    if isinstance(a, np.ndarray):
        return a.astype(float)
    else:
        return np.atleast_1d(a).astype(float)


# suppress divide by zero errors
np.seterr(divide="ignore", invalid="ignore")

""" File of drawdown and stream depletion analytical solutions
    as part of the pycap suite.

"""


def _check_nones(all_vars, var_dict):
    """Function to check if any of the required parameters are
    set to None (default) value. Call to this function
    is added to all solutions that require parameters in
    addition to T, S, time, dist and Q.

    Parameters
    ----------
    all_vars: dictionary
        dictionary of variable values passed from calling
        routine, can be generated using locals()
    var_dict: dictionary
        key is the function name and value is a list of
        required parameter names.
    """
    fxn_name = list(var_dict.keys())[0]

    nonevars = {
        k: v
        for k, v in all_vars.items()
        if (k in var_dict[fxn_name]) & (v is None)
    }
    if len(nonevars) > 0:
        raise PycapException(
            f"The function: {fxn_name} requires the following\n"
            + "additional arguments which were missing\n"
            + "in the function call:\n"
            + ", ".join(nonevars.keys())
        )


# define drawdown methods here
def theis_drawdown(T, S, time, dist, Q, **kwargs):
    """Function to calculate Theis drawdown. Units are not specified, but
        should be consistent length and time.

        Calculates the Theis drawdown solution at specified times
        or distances from a pumping well.

        https://pubs.usgs.gov/publication/70198446

    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    S: float
        storage [dimensionless]
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float, optionally np.array or list
        distance at which to calculate results in [L]
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs: included to all drawdown methods for extra values required
        in some calls

    Returns
    -------
    drawdown: float or array of floats
        drawdown values at input parameter times/distances [L]
    """

    time = _make_arrays(time)
    dist = _make_arrays(dist)
    if len(dist) > 1 and len(time) > 1:
        _time_dist_error("theis_drawdown")

    # construct the well function argument
    # is dist is zero, then function does not exist
    # trap for dist==0 and set to small value

    # sort out geometry of solution
    if len(dist) >= len(time):
        if time[0] == 0:
            return 0
        else:
            ddn = np.zeros_like(dist)
            u = dist**2.0 * S / (4.0 * T * time)
            ddn[dist != 0] = (Q / (4.0 * np.pi * T)) * sps.exp1(u[dist != 0])
            return ddn
    elif len(time) > len(dist):
        ddn = np.zeros_like(time)
        u = np.zeros_like(time)
        u[time != 0] = dist**2.0 * S / (4.0 * T * time[time != 0])
        ddn[time != 0] = (Q / (4.0 * np.pi * T)) * sps.exp1(u[time != 0])
        return ddn


def hunt_99_drawdown(
    T, S, time, dist, Q, streambed_conductance=None, x=None, y=None, **kwargs
):
    """Function to calculate drawdown in an aquifer with a partially
        penetrating stream including streambed resistance (Hunt, 1999).
        Units are not specified, but should be consistent length and time.

        The solution becomes the Theis solution if streambed conductance
        is zero, and approaches an image-well solution from Theis or Glover
        and Balmer (1954) as streambed conductance gets very large.
        Note that the well is located at the location x,y = (l, 0)
        and the stream is aligned with y-axis at x=0.

        x and y locations provided are the points at which drawdown is
        calculated and reported. It is possible to provide x and y
        ndarrays generated with `numpy.meshgrid`.

        Hunt, B., 1999, Unsteady streamflow depletion from ground
        water pumping: Groundwater, v. 37, no. 1, pgs. 98-102,
        https://doi.org/10.1111/j.1745-6584.1999.tb00962.x


    Parameters
    ----------
    T: float
        Transmissivity of aquifer [L**2/T]
    S: float
        Storativity of aquifer [dimensionless]
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float
        distance between well and stream in [L]
    Q : float
        pumping rate (+ is extraction) [L**3/T]
    streambed_conductance: float
        streambed conductance [L/T] (lambda in the paper)
    x: float, optionally ndarray
        x locations at which to report calculated drawdown.
    y: float, optionally ndarray
        y locations at which to report calculated drawdown.
    **kwargs:  included to all drawdown methods for extra values required
        in some calls

    Returns
    -------
    drawdown: float
        single value, meshgrid of drawdowns, or np.array with shape
        (ntimes, meshgridxx, meshgridyy)
        depending on input form of x, y, and ntimes [L]
    """
    _check_nones(
        locals(), {"hunt_99_drawdown": ["streambed_conductance", "x", "y"]}
    )

    # turn lists into np.array so they get handled correctly,
    # check if time or space is an array
    time = _make_arrays(time)
    dist = _make_arrays(dist)

    if len(dist) > 1:
        PycapException(
            "hunt_99_drawdown can only accept a single distance argument"
        )

    spacescalar = True
    if isinstance(x, np.ndarray):
        spacescalar = False

    # compute a single x, y point at a given time
    if len(time) == 1 and spacescalar:
        # handle zero time
        if time[0] == 0:
            return 0
        else:
            warnings.filterwarnings(
                "ignore", category=integrate.IntegrationWarning
            )
            [strmintegral, err] = integrate.quad(
                _ddwn2,
                0.0,
                np.inf,
                args=(dist[0], x, y, T, streambed_conductance, time[0], S),
            )
            return (Q / (4.0 * np.pi * T)) * (
                _ddwn1(dist[0], x, y, T, streambed_conductance, time[0], S)
                - strmintegral
            )

    # compute a vector of times for a given point
    if len(time) > 1 and spacescalar:
        drawdowns = np.zeros_like(time)
        for i, tm in enumerate(time):
            # special case for zero time
            if tm != 0:
                warnings.filterwarnings(
                    "ignore", category=integrate.IntegrationWarning
                )
                [strmintegral, err] = integrate.quad(
                    _ddwn2,
                    0.0,
                    np.inf,
                    args=(dist[0], x, y, T, streambed_conductance, tm, S),
                )
                drawdowns[i] = (Q / (4.0 * np.pi * T)) * (
                    _ddwn1(dist[0], x, y, T, streambed_conductance, tm, S)
                    - strmintegral
                )
        return drawdowns

    # if meshgrid is passed, return an np.array with dimensions
    # ntimes, num_x, num_y
    if not spacescalar:
        numrow = np.shape(x)[0]
        numcol = np.shape(x)[1]
        drawdowns = np.zeros(shape=(len(time), numrow, numcol))
        for time_idx in range(0, len(time)):
            for i in range(0, numrow):
                for j in range(0, numcol):
                    # special case for zero time
                    if time[time_idx] == 0:
                        drawdowns[time_idx, i, j] = 0
                    else:
                        warnings.filterwarnings(
                            "ignore", category=integrate.IntegrationWarning
                        )
                        [strmintegral, err] = integrate.quad(
                            _ddwn2,
                            0.0,
                            np.inf,
                            args=(
                                dist[0],
                                x[i, j],
                                y[i, j],
                                T,
                                streambed_conductance,
                                time[time_idx],
                                S,
                            ),
                        )
                        drawdowns[time_idx, i, j] = (Q / (4.0 * np.pi * T)) * (
                            _ddwn1(
                                dist[0],
                                x[i, j],
                                y[i, j],
                                T,
                                streambed_conductance,
                                time[time_idx],
                                S,
                            )
                            - strmintegral
                        )
        return drawdowns


def _ddwn1(dist, x, y, T, streambed, time, S):
    """Internal method to calculate Theis drawdown function for a point (x,y)

    Used in computing Hunt, 1999 estimate of drawdown.  Equation 30 from
    the paper.  Variables described in hunt_99_drawdown function.
    """
    # construct the well function argument
    # if (l-x) is zero, then function does not exist
    # trap for (l-x)==0 and set to small value
    dist = dist - x
    
    if dist == 0.0:
        dist = 0.001

    u1 = ((dist) ** 2 + y**2) / (4.0 * T * time / S)

    return sps.exp1(u1)


def _ddwn2(theta, dist, x, y, T, streambed, time, S):
    """Internal method to calculate function that gets integrated
        in the Hunt (1999) solution

    Equations 29 and 30 in the paper, theta is the constant
    of integration and the rest of the variables described in the
    hunt_99_drawdown function.
    """
    if streambed == 0.0:
        return 0.0
    u2 = ((dist + np.abs(x) + 2 * T * theta / streambed) ** 2 + y**2) / (
        4.0 * T * time / S
    )
    return np.exp(-theta) * sps.exp1(u2)


def dudley_ward_lough_drawdown(
    T1,
    S1,
    time,
    dist,
    Q,
    T2=None,
    S2=None,
    width=None,
    streambed_thick=None,
    streambed_K=None,
    aquitard_thick=None,
    aquitard_K=None,
    x=None,
    y=None,
    NSteh1=2,
    NSteh2=2,
    **kwargs,
):
    """Compute drawdown using Dudley Ward and Lough (2011) solution

        Dudley Ward and Lough (2011) presented a solution for streamflow depletion
        by a pumping well in a layered aquifer system.  The stream
        is in the upper aquifer, and the pumping well is in a lower
        aquifer that is separated from the upper aquifer by a
        semi-confining aquitard layer.

        Dudley Ward, N.,and Lough, H., 2011, Stream depletion from pumping a
        semiconfined aquifer in a two-layer leaky aquifer system (technical note):
        Journal of Hydrologic Engineering ASCE, v. 16, no. 11, pgs. 955-959,
        https://doi.org/10.1061/(ASCE)HE.1943-5584.0000382.

    Parameters
    ----------
    T: float
        Transmissivity  in the upper aquifer [L**2/T]
        (K*D or T1 in the original paper)
    S: float
        Specific yield for upper aquifer [unitless]
        (S1 in the original paper)
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: Distance between pumping well and stream [L]
        (L in the original paper)
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs:  included to all drawdown methods for extra values required
        in some calls
    Returns
    -------
    ddwn float, 2-column ndarray
        drawdown at specified location [L]
        in the shallow aquifer (column 0)
        and the deeper aquifer (column 1)

    Other Parameters
    ----------------
    T2: float
        Transmissivity of deeper system
    S2: float
        Storativity of
    streambed_thick: float
        thickness of streambed
    streambed_K: float
        hydraulic conductivity of streambed, [L/T]
    aquitard_thick: float
        thickness of intervening leaky aquitard, [L]
    aquitard_K: float
        hydraulic conductivity of intervening leaky aquifer, [L/T]
    x: float
        x-coordinate of drawdown location
        (with origin being x=0 at stream location) [L]
    y: float
        y-coordinate of drawdown location
        (with origin being y=0 at pumping well location) [L]
    NSteh1: int
        Number of Stehfest series levels - algorithmic tuning parameter.
        Defaults to 2.
    NStehl2: int
        Number of Stehfest series levels - algorithmic tuning parameter.
        Defaults to 2.
    width: float
        stream width (b in paper) [L]

    """
    time = _make_arrays(time)
    dist = _make_arrays(dist)
    if len(dist) > 1:
        PycapException(
            "dudley_ward_lough_drawdown can only accept a single distance argument"
        )
    _check_nones(
        locals(),
        {
            "dudley_ward_lough_drawdown": [
                "T2",
                "S2",
                "width",
                "streambed_thick",
                "streambed_K",
                "aquitard_thick",
                "aquitard_K",
                "x",
                "y",
            ]
        },
    )

    # first nondimensionalize all the parameters
    x, y, t, T1, S1, K, lambd = _DudleyWardLoughNonDimensionalize(
        T1,
        T2,
        S1,
        S2,
        width,
        Q,
        dist[0],
        streambed_thick,
        streambed_K,
        aquitard_thick,
        aquitard_K,
        time,
        x,
        y,
    )

    # Initialize output arrays
    s1 = np.zeros_like(t)
    s2 = np.zeros_like(t)

    # Inverse Fourier transform
    for ii in range(len(t)):
        # special case for zero time
        if t[ii] == 0:
            s1[ii] = 0
            s2[ii] = 0
        else:
            try:
                s1[ii] = _StehfestCoeff(1, NSteh1) * _if1(
                    T1, S1, K, lambd, x, y, np.log(2) / t[ii]
                )
                for jj in range(2, NSteh1 + 1):
                    s1[ii] += _StehfestCoeff(jj, NSteh1) * _if1(
                        T1, S1, K, lambd, x, y, jj * np.log(2) / t[ii]
                    )
                s1[ii] *= np.log(2) / t[ii]
            except OverflowError as e:
                print(f"Overflow error in s1 calculation at index {ii}: {e}")
                s1[ii] = np.nan  # Assign NaN if there's an overflow

            try:
                s2[ii] = _StehfestCoeff(1, NSteh2) * _if2(
                    T1, S1, K, lambd, x, y, np.log(2) / t[ii]
                )
                for jj in range(2, NSteh2 + 1):
                    s2[ii] += _StehfestCoeff(jj, NSteh2) * _if2(
                        T1, S1, K, lambd, x, y, jj * np.log(2) / t[ii]
                    )
                s2[ii] *= np.log(2) / t[ii]
            except OverflowError as e:
                print(f"Overflow error in s2 calculation at index {ii}: {e}")
                s2[ii] = np.nan  # Assign NaN if there's an overflow

    return np.array(list(zip(s1 * Q / T2, s2 * Q / T2)))  # re-dimensionalize


# define stream depletion methods here
def glover_depletion(T, S, time, dist, Q, **kwargs):
    """
    Calculate Glover and Balmer (1954) solution for stream depletion

    Depletion solution for a well near a river where the river fully
    penetrates the aquifer and there is no streambed resistance.

    Glover, R.E. and Balmer, G.G., 1954, River depletion from pumping
    a well near a river, Eos Transactions of the American Geophysical Union,
    v. 35, no. 3, pg. 468-470, https://doi.org/10.1029/TR035i003p00468.

    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    S: float
        storage [unitless]
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float, optionally np.array or list
        distance at which to calculate results in [ft]
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs: included to all depletion methods for extra values required in some calls

    Returns
    -------
    drawdown: float
        depletion values at at input parameter times/distances


    """
    # turn lists into np.array so they get handled correctly
    time = _make_arrays(time)
    dist = _make_arrays(dist)
    if len(dist) > 1 and len(time) > 1:
        _time_dist_error("glover_depletion")

    if len(time) == 1 and len(dist) == 1:
        if time == 0:
            return 0
        else:
            return Q * sps.erfc(dist[0] / np.sqrt(4 * (T / S) * time[0]))

    elif len(time) == 1 and len(dist) > 1:
        # handle zero time condition for list-like times
        if time[0] == 0:
            return np.zeros_like(dist)
        else:
            z = dist / np.sqrt(4 * (T / S) * time)
            return Q * sps.erfc(z)
    elif len(time) > 1 and len(dist) == 1:
        # handle zero time condition for list-like times
        z = np.zeros_like(time)
        z[time != 0] = dist / np.sqrt(4 * (T / S) * time[time != 0])
        depl = np.zeros_like(time)
        depl[time != 0] = Q * sps.erfc(z[time != 0])
        return depl


def sdf(T, S, dist, **kwargs):
    """
    internal function for Stream Depletion Factor

    Stream Depletion Factor was defined by Jenkins (1968) and described
    in Jenkins as the time when the volume of stream depletion is
    28 percent of the net volume pumped from the well.
    SDF = dist**2 * S/T.

    Jenkins, C.T., Computation of rate and volume of stream depletion
    by wells: U.S. Geological Survey Techniques of Water-Resources
    Investigations, Chapter D1, Book 4, https://pubs.usgs.gov/twri/twri4d1/.

    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    S: float
        storage [unitless]
    dist: float, optionally np.array or list
        distance at which to calculate results in [L]
    **kwargs: included to all depletion methods for extra values required in some calls

    Returns
    -------
    SDF: float
        Stream depletion factor [T]
    """
    if isinstance(dist, list):
        dist = np.array(dist)
    return dist**2 * S / T


def walton_depletion(T, S, time, dist, Q, **kwargs):
    """
    Calculate depletion using Walton (1987) PT-8 BASIC program logic

    Provides the Glover and Balmer (Jenkins) solution.

    Walton, W.C., Groundwater Pumping Tests:  Lewis Publishers, Chelsea,
    Michigan, 201 p.

    Note that unlike the other depletion functions, this Walton function
    is unit-specific, using feet and days as dimensions.

    Parameters
    ----------
    T: float
        transmissivity [gal per d per ft]
    S: float
        storage [unitless]
    time: float, optionally np.array or list
        time at which to calculate results [d]
    dist: float, optionally np.array or list
        distance at which to calculate results in [ft]
    Q: float
        pumping rate (+ is extraction) [ft**3/d]
    **kwargs: included to all depletion methods for extra values required in some calls

    Returns
    -------
    drawdown: float
        depletion values at at input parameter times/distances

    """
    # turn lists into np.array so they get handled correctly
    time = _make_arrays(time)
    dist = _make_arrays(dist)
    if len(dist) > 1 and len(time) > 1:
        _time_dist_error("walton_depletion")

    if len(time) == 1:
        if time[0] == 0:
            return 0
        else:
            # avoid divide by zero for time==0
            # time = time.values
            G = dist / np.sqrt((0.535 * time * T / S))
    elif len(dist) == 1:
        G = np.zeros_like(time).astype(float)
        G[time != 0] = dist / np.sqrt((0.535 * time[time != 0] * T / S))
        I = (
            1
            + 0.0705230784 * G
            + 0.0422820123 * (G**2)
            + 9.2705272e-03 * (G**3)
        )
        J = (
            I
            + 1.52014e-04 * (G**4)
            + 2.76567e-04 * (G**5)
            + 4.30638e-05 * (G**6)
        ) ** 16
        depl = Q * (1 / J)
        # handle zero time condition
        depl[time == 0] = 0.0
        if len(depl) == 1:
            return depl[0]
        else:
            return depl


def hunt_99_depletion(
    T, S, time, dist, Q, streambed_conductance=None, **kwargs
):
    """Function for Hunt (1999) solution for streamflow depletion by a pumping well.

        Computes streamflow depletion by a pumping well for a partially penetrating
        stream with streambed resistance.

        Hunt, B., 1999, Unsteady streamflow depletion from ground
        water pumping: Groundwater, v. 37, no. 1, pgs. 98-102,
        https://doi.org/10.1111/j.1745-6584.1999.tb00962.x

    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    S: float
        storage [unitless]
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float, optionally np.array or list
        distance at which to calculate results in [L]
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs: included to all depletion methods for extra values required in some calls

    Returns
    -------
    Qs: float
        streamflow depletion rate, optionally np.array or list
        depending on input of time and dist [L**3/T]

    Other Parameters
    ----------------
    streambed_conductance: float
        streambed_conductance conductance [L/T] (lambda in the paper)
    """
    _check_nones(locals(), {"hunt_99_depletion": ["streambed_conductance"]})
    # turn lists into np.array so they get handled correctly
    time = _make_arrays(time)
    dist = _make_arrays(dist)
    if len(dist) > 1 and len(time) > 1:
        _time_dist_error("hunt_99_depletion")

    elif len(time) == 1:
        if time[0] == 0:
            return 0
        else:
            a = np.sqrt(S * dist**2 / (4.0 * T * time))
            b = (streambed_conductance**2 * time) / (4 * S * T)
            c = (streambed_conductance * dist) / (2.0 * T)
            y = np.sqrt(b) + a
            t1 = sps.erfcx(y)
            t2 = np.exp(b + c - y**2)
            depl = sps.erfc(a) - (t1 * t2)
            if len(dist) > 1:
                return depl * Q
            else:
                return depl[0] * Q

    elif len(dist) == 1:
        a = np.zeros_like(time)
        a[time != 0] = np.sqrt(S * dist**2 / (4.0 * T * time[time != 0]))
        b = (streambed_conductance**2 * time) / (4 * S * T)
        c = (streambed_conductance * dist) / (2.0 * T)
        y = np.sqrt(b) + a
        t1 = sps.erfcx(y)
        t2 = np.exp(b + c - y**2)
        depl = np.zeros_like(a)
        depl[time != 0] = sps.erfc(a[time != 0]) - (
            t1[time != 0] * t2[time != 0]
        )
        return depl * Q


def hunt_03_depletion(
    T,
    S,
    time,
    dist,
    Q,
    Bprime=None,
    Bdouble=None,
    aquitard_K=None,
    sigma=None,
    width=None,
    streambed_conductance=None,
    n_quad=100,
    **kwargs,
):
    """Function for Hunt (2003) solution for streamflow depletion by a pumping well.

        Computes streamflow depletion by a pumping well in a semiconfined aquifer
        for a partially penetrating stream.  The stream is in an upper semi-confining
        aquifer and pumping is in a lower aquifer.

        Hunt, B., 2003, Unsteady streamflow depletion when pumping
        from semiconfined aquifer: Journal of Hydrologic Engineering,
        v.8, no. 1, pgs 12-19. https://doi.org/10.1061/(ASCE)1084-0699(2003)8:1(12)


    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    S: float
        storage [unitless]
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float,
        distance at which to calculate results in [L]
        Note, because of computation demand, only a single value
        for distance can be computed in a call
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs: included to all depletion methods for extra values required in some calls

    Returns
    -------
    Qs: float
        streamflow depletion rate, optionally np.array or list
        depending on input of time and dist [L**3/T]

    Other Parameters
    ----------------
    Bprime: float
        saturated thickness of semiconfining layer containing stream, [L]
    Bdouble: float
        distance from bottom of stream to bottom of semiconfining layer,
        [L] (aquitard thickness beneath the stream)
    aquitard_K: float
        hydraulic conductivity of semiconfining layer [L/T]
    sigma: float
        porosity of semiconfining layer
    width: float
        stream width (b in paper) [L]
    streambed_conductance: float
        streambed conductance [L/T] (lambda in the paper),
        only used if K is less than 1e-10
    n_quad: int, optional
        number of Gauss-Legendre quadrature points for numerical integration
        (default: 100)
    """
    _check_nones(
        locals(),
        {
            "hunt_03_depletion": [
                "Bprime",
                "Bdouble",
                "aquitard_K",
                "sigma",
                "width",
                "streambed_conductance",
            ]
        },
    )

    # turn lists into np.array so they get handled correctly
    time = _make_arrays(time)
    dist = _make_arrays(dist)

    if len(dist) > 1 and len(time) > 1:
        _time_dist_error("hunt_03_depletion")

    if len(dist) > 1:
        raise PycapException(
            "cannot have distance as an array\n"
            "in the hunt_03_depletion method. Need to externally loop\n"
            "over distance"
        )

    dist = dist[0]

    # make dimensionless group used in equations
    dtime = (T * time) / (S * np.power(dist, 2))

    # if K is really small, set streambed conductance to a value
    # so solution collapses to Hunt 1999 (confined aquifer solution)
    if aquitard_K < 1.0e-10:
        lam = streambed_conductance
    else:
        lam = aquitard_K * width / Bdouble

    dlam = lam * dist / T
    epsilon = S / sigma
    dK = (aquitard_K / Bprime) * np.power(dist, 2) / T

    # Compute correction terms using vectorized quadrature
    correction = _compute_correction(dtime, dlam, epsilon, dK, n_quad=n_quad)

    # terms for depletion, similar to Hunt (1999) but repeated
    # here so it matches the 2003 paper.
    # note correcting for zero time
    a = np.zeros_like(dtime)
    nonzero_mask = dtime != 0
    a[nonzero_mask] = 1.0 / (2.0 * np.sqrt(dtime[nonzero_mask]))

    b = dlam / 2.0 + (dtime * np.power(dlam, 2) / 4.0)
    c = a + (dlam * np.sqrt(dtime) / 2.0)

    # use erfcx() function from scipy (see hunt_99_depletion above)
    # for erf(b)*erfc(c) term
    t1 = sps.erfcx(c)
    t2 = np.exp(b - c**2)

    # note correcting for zero time
    depl = np.zeros_like(dtime)
    depl[nonzero_mask] = sps.erfc(a[nonzero_mask]) - (
        t1[nonzero_mask] * t2[nonzero_mask]
    )

    # corrected depletion for storage of upper semiconfining unit
    if len(depl) == 1:
        return Q * (depl[0] - correction[0])
    else:
        return Q * (depl - correction)
# Pre-compute binomial coefficients for n = 0 to 59
# binom(2n, n) for the series in _G
_MAX_N = 60
_N_RANGE = np.arange(_MAX_N, dtype=np.float64)
_N2_RANGE = 2 * _N_RANGE

# Pre-compute log(binom(2n, n)) using log-gamma for numerical stability
# log(binom(2n, n)) = log((2n)!) - 2*log(n!)
#                   = gammaln(2n+1) - 2*gammaln(n+1)
_LOG_BINOM_2N_N = gammaln(_N2_RANGE + 1) - 2 * gammaln(_N_RANGE + 1)

# Pre-compute binom(2n, n) directly for small n
_BINOM_2N_N = sps.binom(_N2_RANGE, _N_RANGE)

# Pre-compute default Gauss-Legendre quadrature points and weights
# 100 points provides high accuracy for numerical integration
_DEFAULT_N_QUAD = 100
_x_gl_default, _w_gl_default = np.polynomial.legendre.leggauss(_DEFAULT_N_QUAD)
_ALPHA_DEFAULT = 0.5 * (_x_gl_default + 1)  # Transform [-1,1] to [0,1]
_WEIGHTS_DEFAULT = 0.5 * _w_gl_default

# Cache for non-default quadrature
_QUAD_CACHE = {_DEFAULT_N_QUAD: (_ALPHA_DEFAULT, _WEIGHTS_DEFAULT)}


def _get_gauss_legendre(n_quad):
    """Get cached Gauss-Legendre quadrature points and weights for [0, 1]."""
    if n_quad not in _QUAD_CACHE:
        x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
        # Transform from [-1, 1] to [0, 1]
        alpha_points = 0.5 * (x_gl + 1)
        weights = 0.5 * w_gl
        _QUAD_CACHE[n_quad] = (alpha_points, weights)
    return _QUAD_CACHE[n_quad]


def _compute_correction(dtime_array, dlam, epsilon, dK, n_quad=_DEFAULT_N_QUAD):
    """
    Compute the correction term for all time points using vectorized operations.

    Parameters
    ----------
    dtime_array : ndarray
        Array of dimensionless times
    dlam : float
        Dimensionless streambed conductance
    epsilon : float
        Dimensionless storage
    dK : float
        Dimensionless conductivity
    n_quad : int
        Number of quadrature points (default: 100)

    Returns
    -------
    ndarray
        Correction values for each time point
    """
    alpha_points, weights = _get_gauss_legendre(n_quad)

    corrections = np.zeros_like(dtime_array)

    for i, dt in enumerate(dtime_array):
        if dt == 0:
            corrections[i] = 0.0
        else:
            # Evaluate integrand at all quadrature points (vectorized)
            integrand_vals = _integrand(alpha_points, dlam, dt, epsilon, dK)
            # Compute weighted sum
            corrections[i] = dlam * np.dot(weights, integrand_vals)

    return corrections


def _G(alpha, epsilon, dK, dtime):
    """
    Vectorized G function from Hunt (2003) equation (46).
    
    This implementation eliminates the Python loop over n by using
    NumPy broadcasting to compute all 60 terms simultaneously.
    
    Parameters
    ----------
    alpha : float or ndarray
        Integration variable
    epsilon : float
        Dimensionless storage (storativity/porosity)
    dK : float
        Dimensionless conductivity
    dtime : float
        Dimensionless time
        
    Returns
    -------
    float or ndarray
        G function value(s)
    """
    # Handle scalar alpha
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    scalar_input = alpha.shape == (1,)
    
    # If dimensionless K is very small, return 0
    if dK < 1.0e-10:
        result = np.zeros_like(alpha)
        return result[0] if scalar_input else result
    
    alpha2 = alpha ** 2
    dKdtime = dK * dtime
    
    # Compute a and b for each alpha value
    # Shape: (n_alpha,)
    a = epsilon * dKdtime * (1.0 - alpha2)
    b = dKdtime * alpha2
    ab = a + b
    atb = a * b
    
    # Handle potential numerical issues
    # Where atb < 0, set to small positive value
    atb = np.maximum(atb, 1e-300)
    
    sqrt_atb = np.sqrt(atb)
    
    # term1 = exp(-ab) * I0(2*sqrt(atb)) when ab < 80, else 0
    term1 = np.where(ab < 80, np.exp(-ab) * sps.i0(2.0 * sqrt_atb), 0.0)
    
    # abterm = sqrt(atb) / ab
    # Protect against division by zero
    abterm = np.where(ab > 1e-300, sqrt_atb / ab, 0.0)
    
    # Now compute the sum over n = 0 to 59
    # For each alpha, we need: sum over n of binom(2n,n) * gammainc(2n+1, ab) * abterm^(2n)
    
    # Expand dimensions for broadcasting: (n_alpha, n_terms)
    ab_expanded = ab[:, np.newaxis]           # (n_alpha, 1)
    abterm_expanded = abterm[:, np.newaxis]   # (n_alpha, 1)
    
    # n values broadcast ready
    n2 = _N2_RANGE[np.newaxis, :]             # (1, 60)
    
    # Compute gammainc for all (alpha, n) combinations
    gammainc_vals = sps.gammainc(n2 + 1, ab_expanded)  # (n_alpha, 60)
    
    # Direct computation for small n (stable)
    direct_terms = _BINOM_2N_N * gammainc_vals * (abterm_expanded ** n2)  # (n_alpha, 60)
    
    # Log computation for large n (numerical stability)
    log_gammainc = np.where(
        gammainc_vals > 1e-300,
        np.log(gammainc_vals),
        -700  # Very negative log for essentially zero
    )
    log_abterm = np.where(
        abterm_expanded > 1e-300,
        np.log(abterm_expanded),
        -700
    )
    
    log_terms = _LOG_BINOM_2N_N + log_gammainc + n2 * log_abterm
    log_computed_terms = np.exp(log_terms)
    
    # Use direct computation for n <= 8, log computation for n > 8
    n_mask = (_N_RANGE <= 8)[np.newaxis, :]  # (1, 60)
    terms = np.where(n_mask, direct_terms, log_computed_terms)
    
    # Handle NaN and Inf
    terms = np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sum over n dimension
    sum1 = np.sum(terms, axis=1)  # (n_alpha,)
    
    # Compute equation 52
    ba_over_ab = np.where(np.abs(ab) > 1e-300, (b - a) / ab, 0.0)
    eqn52 = 0.5 * (1.0 - term1 + ba_over_ab * sum1)
    
    # Clamp to [0, 1]
    eqn52 = np.clip(eqn52, 0.0, 1.0)
    
    return eqn52[0] if scalar_input else eqn52


def _F(alpha, dlam, dtime):
    """Vectorized F function from Hunt (2003) equation (47).

    Parameters
    ----------
    alpha : float or ndarray
        Integration variable
    dlam : float
        Dimensionless streambed conductance
    dtime : float
        Dimensionless time

    Returns
    -------
    float or ndarray
        F function value(s)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    scalar_input = alpha.shape == (1,)

    # z = alpha * dlam * sqrt(dtime) / 2 + 1 / (2 * alpha * sqrt(dtime))
    sqrt_dtime = np.sqrt(dtime)
    z = alpha * dlam * sqrt_dtime / 2.0 + 1.0 / (2.0 * alpha * sqrt_dtime)

    # Compute both branches
    # Branch 1: |z| < 3
    a = dlam / 2.0 + (dtime * alpha**2 * dlam**2 / 4.0)
    t1_f1 = sps.erfcx(z)
    t2_f1 = np.exp(a - z**2)
    b_f1 = -1.0 / (4 * dtime * alpha**2)
    F_branch1 = np.exp(b_f1) * np.sqrt(dtime / np.pi) - (
        alpha * dtime * dlam
    ) / 2.0 * (t1_f1 * t2_f1)

    # Branch 2: |z| >= 3 (asymptotic expansion)
    t1_f2 = np.exp(-(1.0 / (4.0 * dtime * alpha**2))) / (
        2.0 * alpha * z * np.sqrt(np.pi)
    )
    t2_f2 = 2.0 / (dlam * (1.0 + (1.0 / (dlam * dtime * alpha**2)) ** 2))
    sumterm = (
        1
        - (3.0 / (2 * z**2))
        + (15.0 / (4.0 * z**4))
        - (105.0 / (8 * z**6))
    )
    F_branch2 = t1_f2 * (1.0 + t2_f2 * sumterm)

    # Select based on |z|
    F = np.where(np.abs(z) < 3.0, F_branch1, F_branch2)

    return F[0] if scalar_input else F


def _integrand(alpha, dlam, dtime, epsilon, dK):
    """Vectorized integrand F(alpha) * G(alpha) for numerical integration."""
    return _F(alpha, dlam, dtime) * _G(alpha, epsilon, dK, dtime)


def _calc_deltaQ(Q):
    """internal function to parse the Q time series to find changes and their associated times

    Parameters
    ----------
    Q: pandas Series
        time series of pumping

    Returns
    -------
    deltaQ: pandas Series)
        times and changes in Q over time
    """
    # find the differences in pumping
    dq = Q.copy()
    dq.iloc[1:] = np.diff(Q)
    # get the locations of changes
    deltaQ = dq.loc[dq != 0]
    # special case for starting with 0 pumping
    if Q.index[0] not in deltaQ.index:
        deltaQ.loc[Q.index[0]] = Q.iloc[0]
        deltaQ.sort_index(inplace=True)
    return deltaQ


def _DudleyWardLoughNonDimensionalize(
    T1,
    T2,
    S1,
    S2,
    width,
    Q,
    dist,
    streambed_thick,
    streambed_K,
    aquitard_thick,
    aquitard_K,
    t,
    x=0,
    y=0,
):
    """Internal function to make non-dimensional groups for Dudley Ward and Lough solution"""
    t = np.array(t)  # make sure not passing a list
    if x is not None:
        x /= dist
    if y is not None:
        y /= dist
    t = t * T2 / (S2 * (dist**2))
    T1 /= T2
    S1 /= S2
    K = ((aquitard_K / aquitard_thick) * (dist**2)) / T2
    lambd = ((streambed_K * width) / streambed_thick) * dist / T2
    return x, y, t, T1, S1, K, lambd


def dudley_ward_lough_depletion(
    T1,
    S1,
    time,
    dist,
    Q,
    T2=None,
    S2=None,
    width=None,
    streambed_thick=None,
    streambed_K=None,
    aquitard_thick=None,
    aquitard_K=None,
    NSteh1=2,
    **kwargs,
):
    """
    Compute streamflow depletion using Dudley Ward and Lough (2011) solution

    Dudley Ward and Lough (2011) presented a solution for streamflow depletion
    by a pumping well in a layered aquifer system.  The stream
    is in the upper aquifer, and the pumping well is in a lower
    aquifer that is separated from the upper aquifer by a
    semi-confining aquitard layer.

    Dudley Ward, N.,and Lough, H., 2011, Stream depletion from pumping a
    semiconfined aquifer in a two-layer leaky aquifer system (techical note):
    Journal of Hydrologic Engineering ASCE, v. 16, no. 11, pgs. 955-959,
    https://doi.org/10.1061/(ASCE)HE.1943-5584.0000382.

    Parameters
    ----------
    T: float
        transmissivity [L**2/T]
    storage [unitless]
        specific yield in the upper q aquifer
    time: float, optionally np.array or list
        time at which to calculate results [T]
    dist: float, optionally np.array or list
        distance at which to calculate results in [L]
    Q: float
        pumping rate (+ is extraction) [L**3/T]
    **kwargs: included to all depletion methods for extra values required in some calls


    Returns
    -------
    Qs: float
        streamflow depletion rate, optionally np.array or list
        depending on input of time and dist [L**3/T]

    Other Parameters
    ----------------
    T2: float
        Transmissivity of deeper system
    S2: float
        Storativity of
    streambed_thick: float
        thickness of streambed
    streambed_K: float
        hydraulic conductivity of streambed, [L/T]
    aquitard_thick: float
        thickness of intervening leaky aquitard, [L]
    aquitard_K: float
        hydraulic conductivity of intervening leaky aquifer, [L/T]
    NSteh1: int
        Number of Stehfest series levels - algorithmic tuning parameter.
        Defaults to 2.
    width: float
        stream width (b in paper) [L]

    """
    _check_nones(
        locals(),
        {
            "dudley_ward_lough_depletion": [
                "T2",
                "S2",
                "width",
                "streambed_thick",
                "streambed_K",
                "aquitard_thick",
                "aquitard_K",
            ]
        },
    )
    # first nondimensionalize all the parameters
    x, y, t, T1, S1, K, lambd = _DudleyWardLoughNonDimensionalize(
        T1,
        T2,
        S1,
        S2,
        width,
        Q,
        dist,
        streambed_thick,
        streambed_K,
        aquitard_thick,
        aquitard_K,
        time,
        0,
        0,
    )

    # Inverse Fourier transform
    # Handle scalar time values
    if isinstance(t, int) or isinstance(t, float):
        if t == 0:
            return 0
        # Compute for scalar time
        DeltaQ = _StehfestCoeff(1, NSteh1) * _if1_dQ(
            T1, S1, K, lambd, np.log(2) / t
        )
        for jj in range(2, NSteh1 + 1):
            DeltaQ += _StehfestCoeff(jj, NSteh1) * _if1_dQ(
                T1, S1, K, lambd, jj * np.log(2) / t
            )
        DeltaQ = 2 * np.pi * lambd * DeltaQ * np.log(2) / t
        return DeltaQ * Q  # redimensionalize

    # Handle array/list time values
    if isinstance(t, list):
        t = np.array(t)
    DeltaQ = np.zeros_like(t)
    DeltaQ[t != 0] = _StehfestCoeff(1, NSteh1) * _if1_dQ(
        T1, S1, K, lambd, np.log(2) / t[t != 0]
    )
    for jj in range(2, NSteh1 + 1):
        DeltaQ[t != 0] += _StehfestCoeff(jj, NSteh1) * _if1_dQ(
            T1, S1, K, lambd, jj * np.log(2) / t[t != 0]
        )
    DeltaQ[t != 0] = (
        2 * np.pi * lambd * DeltaQ[t != 0] * np.log(2) / t[t != 0]
    )

    return DeltaQ * Q  # redimensionalize


def _if1_dQ(T1, S1, K, lambda_, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    return _kernel1(T1, S1, K, lambda_, 0, 0, p) + _kernel2(
        T1, S1, K, lambda_, 0, 0, p
    )


def _if1(T1, S1, K, lambd, x, y, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    G = (
        lambda phi: 2
        * (
            _kernel1(T1, S1, K, lambd, x, np.tan(phi), p)
            + _kernel2(T1, S1, K, lambd, x, np.tan(phi), p)
        )
        * np.cos(np.tan(phi) * y)
        / np.cos(phi) ** 2
    )
    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
    s1InvFour, _ = integrate.quad(
        G, 0, np.pi / 2, epsrel=1e-1, epsabs=1e-1, limit=10000
    )
    return s1InvFour


def _if2(T1, S1, K, lambd, x, y, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    H = (
        lambda phi: 2
        * (
            _coeff_s1_1(T1, S1, K, lambd, np.tan(phi), p)
            * _kernel1(T1, S1, K, lambd, x, np.tan(phi), p)
            + _coeff_s1_2(T1, S1, K, lambd, np.tan(phi), p)
            * _kernel2(T1, S1, K, lambd, x, np.tan(phi), p)
        )
        * np.cos(np.tan(phi) * y)
        / np.cos(phi) ** 2
    )
    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
    s2InvFour, errbnd = integrate.quad(
        H, 0, np.pi / 2, epsrel=1e-1, epsabs=1e-1, limit=10000
    )
    return s2InvFour


def _coeff_s1_1(T1, S1, K, lambd, theta, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    b11, b12, b22, mu1, mu2, l1, l2, beta1, beta2, A1, A2 = _coeffs(
        T1, S1, K, lambd, theta, p
    )
    B1 = (mu1 * T1 - b11) / b12
    return B1


def _coeff_s1_2(T1, S1, K, lambd, theta, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    b11, b12, b22, mu1, mu2, l1, l2, beta1, beta2, A1, A2 = _coeffs(
        T1, S1, K, lambd, theta, p
    )
    B2 = (mu2 * T1 - b11) / b12
    return B2


def _kernel1(T1, S1, K, lambd, x, theta_or_y, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    b11, b12, b22, mu1, mu2, l1, l2, beta1, beta2, A1, A2 = _coeffs(
        T1, S1, K, lambd, theta_or_y, p
    )

    if x < 0:
        F1 = A1 * np.exp(x * np.sqrt(mu1))
    elif 0 <= x <= 1:
        F1 = A1 * np.exp(-x * np.sqrt(mu1)) + beta1 / (
            2 * np.sqrt(mu1) * l1
        ) * (np.exp((x - 1) * np.sqrt(mu1)) - np.exp(-(x + 1) * np.sqrt(mu1)))
    else:
        F1 = A1 * np.exp(-x * np.sqrt(mu1)) + beta1 / (
            2 * np.sqrt(mu1) * l1
        ) * (np.exp((1 - x) * np.sqrt(mu1)) - np.exp(-(x + 1) * np.sqrt(mu1)))
    return F1


def _kernel2(T1, S1, K, lambd, x, theta_or_y, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    b11, b12, b22, mu1, mu2, l1, l2, beta1, beta2, A1, A2 = _coeffs(
        T1, S1, K, lambd, theta_or_y, p
    )

    if x < 0:
        F2 = A2 * np.exp(x * np.sqrt(mu2))
    elif 0 <= x <= 1:
        F2 = A2 * np.exp(-x * np.sqrt(mu2)) + beta2 / (
            2 * np.sqrt(mu2) * l2
        ) * (np.exp((x - 1) * np.sqrt(mu2)) - np.exp(-(x + 1) * np.sqrt(mu2)))
    else:
        F2 = A2 * np.exp(-x * np.sqrt(mu2)) + beta2 / (
            2 * np.sqrt(mu2) * l2
        ) * (np.exp((1 - x) * np.sqrt(mu1)) - np.exp(-(x + 1) * np.sqrt(mu1)))
    return F2


def _coeffs(T1, S1, K, lambd, theta_or_y, p):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    b11 = T1 * theta_or_y**2 + S1 * p + K
    b12 = -K
    b22 = theta_or_y**2 + p + K

    mu1 = (b11 / T1 + b22) / 2 + np.sqrt(
        (b11 / T1 + b22) ** 2 / 4 + (b12**2 - b11 * b22) / T1
    )
    mu2 = (b11 / T1 + b22) / 2 - np.sqrt(
        (b11 / T1 + b22) ** 2 / 4 + (b12**2 - b11 * b22) / T1
    )
    l1 = T1 + ((mu1 * T1 - b11) / b12) ** 2
    l2 = T1 + ((mu2 * T1 - b11) / b12) ** 2

    beta1 = (mu1 * T1 - b11) / (b12 * 2 * np.pi * p)
    beta2 = (mu2 * T1 - b11) / (b12 * 2 * np.pi * p)

    Delta = 4 * np.sqrt(mu1 * mu2) + 2 * lambd * (
        np.sqrt(mu1) / l2 + np.sqrt(mu2) / l1
    )

    A1 = (
        (
            (lambd / l2 + 2 * np.sqrt(mu2)) * beta1 * np.exp(-np.sqrt(mu1))
            - lambd * beta2 / l2 * np.exp(-np.sqrt(mu2))
        )
        / Delta
        / l1
    )

    A2 = (
        (
            -lambd * beta1 / l1 * np.exp(-np.sqrt(mu1))
            + (lambd / l1 + 2 * np.sqrt(mu1)) * beta2 * np.exp(-np.sqrt(mu2))
        )
        / Delta
        / l2
    )

    return b11, b12, b22, mu1, mu2, l1, l2, beta1, beta2, A1, A2


def _safe_factorial(n):
    """Calculate factorial using logarithmic method to avoid overflow."""
    if n < 0:
        return float("inf")
    elif n < 2:
        return 1
    else:
        return np.exp(gammaln(n + 1))


def _StehfestCoeff(jj, N):
    """Internal function for Dudley Ward and Lough (2011) solution"""
    LowerLimit = (jj + 1) // 2
    UpperLimit = min(jj, N // 2)

    V = 0
    for kk in range(LowerLimit, UpperLimit + 1):
        denominator = (
            _safe_factorial(N // 2 - kk)
            * _safe_factorial(kk)
            * _safe_factorial(kk - 1)
            * _safe_factorial(jj - kk)
            * _safe_factorial(2 * kk - jj)
        )
        if denominator != 0:  # Prevent division by zero
            V += kk ** (N // 2) * _safe_factorial(2 * kk) / denominator

    V *= (-1) ** (N // 2 + jj)
    return V


# List drawdown and depletion methods so they can be called
# programatically
ALL_DD_METHODS = {
    "theis_drawdown": theis_drawdown,
    "hunt_99_drawdown": hunt_99_drawdown,
    "dudley_ward_lough_drawdown": dudley_ward_lough_drawdown,
}

ALL_DEPL_METHODS = {
    "glover_depletion": glover_depletion,
    "walton_depletion": walton_depletion,
    "hunt_99_depletion": hunt_99_depletion,
    "hunt_03_depletion": hunt_03_depletion,
    "dudley_ward_lough_depletion": dudley_ward_lough_depletion,
}

GPM2CFD = 60 * 24 / 7.48  # factor to convert from GPM to CFD
CFD2GPM = 1 / GPM2CFD  # factor to convert from CFD to GPM
SEC2DAY = 60 * 60 * 24  # factor to conver x/sec to x/day
