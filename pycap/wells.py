import sys

import numpy as np

import pycap


class WellResponse:
    """Class to facilitate depletion or drawdown calculations

    Objects from this class will have the required information needed to
    call any of the analysis methods in the package.  The intent is that
    users will not have to access analytical solutions directly, but
    can set up a WellResponse object.  The generation of WellResponse
    objects is generally done through an AnalysisProject object.

    """

    def __init__(
        self,
        name,
        response_type,
        T,
        S,
        dist,
        Q,
        stream_apportionment=None,
        dd_method="theis_drawdown",
        depl_method="glover_depletion",
        theis_time=-9999,
        depl_pump_time=-99999,
        streambed_conductance=None,
        Bprime=None,
        Bdouble=None,
        sigma=None,
        width=None,
        T2=None,
        S2=None,
        streambed_thick=None,
        streambed_K=None,
        aquitard_thick=None,
        aquitard_K=None,
        x=None,
        y=None,
    ) -> None:
        """Class to calculate a single response for a single pumping well.

        Parameters
        ----------
        name: string
            pumping well name
        response_type: string
            reserved for future implementation
        T: float
            Aquifer Transmissivity [L**2/T]
        S: float
            Aquifer Storage [unitless]
        dist: float
            Distance between well and response [L]
        Q: pandas series
            Pumping rate changes and times [L**3/T]
        stream_apportionment: dict of floats
                Dictionary with stream responses and fraction of depletion
                attributed to each. Defaults to None.
        dd_method: string, optional
            Method to be used for drawdown calculations. Defaults to 'theis_drawdown'.
        depl_method: string, optional
            Method to be used for depletion calculations. Defaults to 'glover_depletion'.
        theis_time: integer, optional
            Time at which drawdown calculation should be made [T].
            Defaults to -9999.
        depl_pump_time: integer, optional
            Length of time per year that pumping should be simulated for depletion
            calculations [T]. Not used if pumping time series is used.
            Defaults to -99999.
        streambed_conductance: float
            Streambed conductance for the hunt_99_depletion depletion method [L/T].
            Defaults to None

        Additional Parameters Used by Hunt and Dudley Ward/Lough Solutions
        -----------------------------------------------------------
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
            stream width [T]
        x: float
            x-coordinate of drawdown location
            (with origin being x=0 at stream location) [L]
        y: float
            y-coordinate of drawdown location
            (with origin being y=0 at pumping well location) [L]

        Additional Parameters Used by Dudley Ward/Lough Solutions
        --------------------------------------------------
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
        """
        self._drawdown = None
        self._depletion = None
        self.name = name  # name of response (stream, or drawdown response
        # (e.g. assessed well, lake, spring)) evaluated
        self.response_type = response_type  # might use this later to
        # sort out which response to return
        self.T = T
        self.T_gpd_ft = T * 7.48
        self.S = S
        self.dist = dist
        self.dd_method = dd_method
        self.depl_method = depl_method
        self.theis_time = theis_time
        self.depl_pump_time = depl_pump_time
        self.Q = Q
        self.stream_apportionment = stream_apportionment
        self.streambed_conductance = streambed_conductance
        self.Bprime = Bprime
        self.Bdouble = Bdouble
        self.sigma = sigma
        self.width = width
        self.T2 = T2
        self.S2 = S2
        self.streambed_thick = streambed_thick
        self.streambed_K = streambed_K
        self.aquitard_thick = aquitard_thick
        self.aquitard_K = aquitard_K
        self.x = x
        self.y = y
        self.extra_args = {
            "streambed_conductance": streambed_conductance,
            "Bprime": Bprime,
            "Bdouble": Bdouble,
            "sigma": sigma,
            "width": width,
            "T2": T2,
            "S2": S2,
            "streambed_thick": streambed_thick,
            "streambed_K": streambed_K,
            "aquitard_thick": aquitard_thick,
            "aquitard_K": aquitard_K,
            "x": x,
            "y": y,
        }

    def _calc_drawdown(self):
        """calculate drawdown at requested distance and
        time using solution given as attribute to the object"""
        dd_f = pycap.ALL_DD_METHODS[self.dd_method.lower()]
        # start with zero drawdown
        if "lough" not in self.dd_method.lower():
            dd = np.zeros(len(self.Q))
        else:
            dd = np.zeros((len(self.Q), 2))
        deltaQ = pycap._calc_deltaQ(self.Q.copy())
        # initialize with pumping at the first time being positive
        idx = deltaQ.index[0] - 1
        cQ = deltaQ.iloc[0]
        ct = list(range(idx, len(self.Q)))
        dd[idx:] = dd_f(self.T, self.S, ct, self.dist, cQ, **self.extra_args)
        if len(deltaQ) > 1:
            deltaQ = deltaQ.iloc[1:]
            for idx, cQ in zip(deltaQ.index, deltaQ.values):
                idx -= 2
                ct = list(range(len(self.Q) - idx))
                # note that by setting Q negative from the diff calculations, we always add
                # below for the image wells
                dd[idx:] += dd_f(
                    self.T, self.S, ct, self.dist, cQ, **self.extra_args
                )
        return dd

    def _calc_depletion(self):
        """Optimized depletion calculation.

        Computes a unit response (Q=1) once for the full time range,
        then applies superposition via array slicing and scaling.
        This assumes depletion is linear in Q, which holds for all
        analytical solutions in pycap (Glover, Hunt99, Hunt03, etc.).

        For intermittent pumping with many on/off transitions, this
        reduces hundreds of depletion function calls down to one.
        """
        depl_f = pycap.ALL_DEPL_METHODS[self.depl_method.lower()]
        depl = np.zeros(len(self.Q))

        deltaQ = pycap._calc_deltaQ(self.Q.copy())

        if self.depl_method.lower() == "walton_depletion":
            T = self.T_gpd_ft
        else:
            T = self.T

        # Compute unit response (Q=1) for the longest possible time array.
        # All superposition calls use time arrays that are subsets of this.
        max_len = len(self.Q)
        unit_time = list(range(max_len))
        unit_response = depl_f(
            T,
            self.S,
            unit_time,
            self.dist,
            1.0,
            **self.extra_args,
        )

        # Apply superposition using the precomputed unit response
        idx = deltaQ.index[0] - 1
        cQ = deltaQ.iloc[0]
        n = max_len - idx
        depl[idx:] = cQ * self.stream_apportionment * unit_response[:n]

        if len(deltaQ) > 1:
            for i_idx, cQ in zip(deltaQ.index[1:], deltaQ.values[1:]):
                idx = i_idx - 2
                n = max_len - idx
                depl[idx:] += cQ * self.stream_apportionment * unit_response[:n]

        return depl

    @property
    def drawdown(self):
        return self._calc_drawdown()

    @property
    def depletion(self):
        return self._calc_depletion()


class Well:
    """Object to evaluate a pending (or existing,
    or a couple other possibilities) well with all relevant impacts.
    Preprocessing makes unit conversions and calculates distances as needed
    """

    def __init__(
        self,
        well_status="pending",
        T=-9999,
        S=-99999,
        Q=-99999,
        depletion_years=5,
        theis_dd_days=-9999,
        depl_pump_time=-9999,
        stream_dist=None,
        drawdown_dist=None,
        stream_apportionment=None,
        depl_method="walton_depletion",
        drawdown_method="theis_drawdown",
        streambed_conductance=None,
        Bprime=None,
        Bdouble=None,
        sigma=None,
        width=None,
        T2=None,
        S2=None,
        streambed_thick=None,
        streambed_K=None,
        aquitard_thick=None,
        aquitard_K=None,
        x=None,
        y=None,
    ) -> None:
        """
        Object to evaluate a pending (or existing,
        or a couple other possibilities) well with all relevant impacts.
        Preprocessing makes unit conversions and calculates distances as needed

        Parameters
        ----------
        T: float
            Aquifer Transmissivity [L**2/T]
        S: float
            Aquifer Storage [unitless]
        Q: pandas series
            Pumping rate changes and times [L**3/T]
        depletion_years: int, optional
            Number of years over which to calculate depletion. Defaults to 4.
        theis_dd_days: int
            Number of days at which drawdown is calculated. Defaults to -9999.
        depl_pump_time: integer, optional
            Length of time per year that pumping should be simulated for depletion
            calculations [T]. Not used if pumping time series is used.
            Defaults to -99999.
        stream_apportionment: dict of floats
                Dictionary with stream responses and fraction of depletion
                attributed to each. Defaults to None.
        drawdown_dist: float
            Distance between well and drawdown calculation location. [L]
        stream_apportionment: dict of floats
                Dictionary with stream responses and fraction of depletion
                attributed to each. Defaults to None.
        depl_method: string, optional
            Method to be used for depletion calculations. Defaults to 'glover_depletion'.
        drawdown_method: string, optional
            Method to be used for drawdown calculations. Defaults to 'theis_drawdown'.
            Only 'theis_drawdown' is available right now in the Well() class, if
            Hunt (1999) or Dudley Ward and Lough (2014) are desired, the function
            must be called directly
        streambed_conductance: float
            Streambed conductance for the hunt_99_depletion depletion method [L/T].
            Defaults to None

        Additional Parameters Used by Hunt and Dudley Ward/Lough Solutions
        -----------------------------------------------------------
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
            stream width [T]
        x: float
            x-coordinate of drawdown location
            (with origin being x=0 at stream location) [L]
        y: float
            y-coordinate of drawdown location
            (with origin being y=0 at pumping well location) [L]

        Additional Parameters Used by Dudley Ward/Lough Solutions
        --------------------------------------------------
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
        """

        # placeholders for values returned with @property decorators
        self._depletion = None
        self._drawdown = None
        self._max_depletion = None
        self.depl_method = depl_method
        self.drawdown_method = drawdown_method
        self.stream_dist = stream_dist
        self.drawdown_dist = drawdown_dist
        self.T = T
        self.S = S
        self.depletion_years = int(depletion_years)
        self.theis_dd_days = int(theis_dd_days)
        self.depl_pump_time = depl_pump_time
        self.Q = Q
        self.stream_apportionment = stream_apportionment
        self.streambed_conductance = streambed_conductance
        self.Bprime = Bprime
        self.Bdouble = Bdouble
        self.sigma = sigma
        self.width = width
        self.T2 = T2
        self.S2 = S2
        self.streambed_thick = streambed_thick
        self.streambed_K = streambed_K
        self.aquitard_thick = aquitard_thick
        self.aquitard_K = aquitard_K
        self.x = x
        self.y = y
        self.stream_responses = {}  # dict of WellResponse objects
        # for this well with streams
        self.drawdown_responses = {}  # dict of WellResponse objects
        # for this well with drawdown responses
        self.well_status = well_status  # this is for the well object -
        # later used for aggregation and must be
        # {'existing', 'active', 'pending',
        # 'new_approved', 'inactive' }
        # make sure stream names consistent
        # between dist and apportionment

        if self.drawdown_method.lower() != "theis_drawdown":
            print(
                "'theis_drawdown' must be used as drawdown method in Well Class."
            )
            sys.exit()
        if stream_dist is not None and stream_apportionment is not None:
            assert (
                len(
                    set(self.stream_dist.keys())
                    - set(self.stream_apportionment.keys())
                )
                == 0
            )
        if stream_dist is not None and streambed_conductance is not None:
            assert (
                len(
                    set(self.stream_dist.keys())
                    - set(self.streambed_conductance.keys())
                )
                == 0
            )
        if self.stream_dist is not None:
            self.stream_response_names = list(self.stream_dist.keys())
        if self.drawdown_dist is not None:
            self.drawdown_response_names = list(self.drawdown_dist.keys())

        # now make all the WellResponse objects
        # first for streams
        extra_args = {
            "Bprime": self.Bprime,
            "Bdouble": self.Bdouble,
            "sigma": self.sigma,
            "width": self.width,
            "T2": self.T2,
            "S2": self.S2,
            "streambed_thick": self.streambed_thick,
            "streambed_K": self.streambed_K,
            "aquitard_thick": self.aquitard_thick,
            "aquitard_K": self.aquitard_K,
            "x": self.x,
            "y": self.y,
        }
        if self.stream_dist is not None:
            for cs, (cname, cdist) in enumerate(self.stream_dist.items()):
                if self.streambed_conductance is not None:
                    streambed_conductance_current = self.streambed_conductance[
                        cname
                    ]
                else:
                    streambed_conductance_current = None
                self.stream_responses[cs + 1] = WellResponse(
                    cname,
                    "stream",
                    T=self.T,
                    S=self.S,
                    dist=cdist,
                    depl_pump_time=self.depl_pump_time,
                    Q=self.Q,
                    stream_apportionment=self.stream_apportionment[cname],
                    depl_method=self.depl_method,
                    streambed_conductance=streambed_conductance_current,
                    **extra_args,
                )

        # next for drawdown responses
        if self.drawdown_dist is not None:
            for cw, (cname, cdist) in enumerate(self.drawdown_dist.items()):
                self.drawdown_responses[cw + 1] = WellResponse(
                    cname,
                    "well",
                    T=self.T,
                    S=self.S,
                    dist=cdist,
                    theis_time=self.theis_dd_days,
                    Q=self.Q,
                    dd_method=self.drawdown_method,
                    **extra_args,
                )

    @property
    def drawdown(self):
        if self._drawdown is None:
            self._drawdown = {}
            for cw, cwob in self.drawdown_responses.items():
                self._drawdown[cwob.name] = cwob.drawdown
        return self._drawdown

    @property
    def depletion(self):
        self._depletion = {}
        for _, cwob in self.stream_responses.items():
            self._depletion[cwob.name] = cwob.depletion

        return self._depletion

    @property
    def max_depletion(self):
        return {
            cwob.name: np.nanmax(cwob.depletion)
            for _, cwob in self.stream_responses.items()
        }
