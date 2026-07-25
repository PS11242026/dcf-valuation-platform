from __future__ import annotations

import numpy as np

from dcf_project.models.state_space import FCFStateSpaceModel


def test_state_space_model_forecasts_shape() -> None:
    x = np.array(
        [
            [0.02, 0.30, 0.20, 0.08],
            [0.03, 0.31, 0.21, 0.08],
            [0.01, 0.29, 0.19, 0.07],
            [0.02, 0.30, 0.20, 0.08],
            [0.03, 0.30, 0.18, 0.08],
            [0.02, 0.31, 0.21, 0.08],
            [0.02, 0.30, 0.20, 0.08],
            [0.03, 0.31, 0.19, 0.08],
        ]
    )
    model = FCFStateSpaceModel().fit(x)
    states, covs = model.forecast_state(4)
    assert states.shape == (4, 4)
    assert covs.shape == (4, 4, 4)
    assert np.isfinite(states).all()
