from __future__ import annotations


class RegimeSwitchingModel:
    """Extension placeholder for cyclical-company regime switching.

    The v1 default follows the SRS by shipping the stable multivariate DLM first.
    """

    def fit(self, *_args: object, **_kwargs: object) -> "RegimeSwitchingModel":
        raise NotImplementedError("Regime switching is scaffolded for a future release")

