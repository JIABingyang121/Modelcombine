import numpy as np
import pandas as pd

from src.models.implementations import ARIMAModel, CatBoostModel, LGBMModel, ProphetModel, XGBModel


class FakeProphetBackend:
    def predict(self, future):
        return pd.DataFrame(
            {
                "yhat": np.asarray([10.0, 11.0]),
                "yhat_lower": np.asarray([9.0, 9.5]),
                "yhat_upper": np.asarray([12.0, 13.0]),
            }
        )


def test_prophet_predict_interval_uses_backend_bounds():
    model = ProphetModel()
    model.model = FakeProphetBackend()
    model.train_end = pd.Timestamp("2026-01-01 00:00:00")
    model.freq = "h"
    X = pd.DataFrame(index=pd.date_range("2026-01-01 01:00:00", periods=2, freq="h"))

    yhat, lower, upper = model.predict_interval(X, alpha=0.2)

    assert np.allclose(yhat, [10.0, 11.0])
    assert np.allclose(lower, [9.0, 9.5])
    assert np.allclose(upper, [12.0, 13.0])


class FakeAutoARIMA:
    def predict(self, n_periods, return_conf_int=False, alpha=0.1):
        yhat = np.asarray([20.0, 21.0, 22.0])[:n_periods]
        if not return_conf_int:
            return yhat
        conf = np.column_stack([yhat - 2.0, yhat + 3.0])
        return yhat, conf


def test_arima_predict_interval_uses_auto_arima_conf_int():
    model = ARIMAModel()
    model.fitted_model = FakeAutoARIMA()
    model._model_family = "auto_arima"
    X = pd.DataFrame(index=range(3))

    yhat, lower, upper = model.predict_interval(X, alpha=0.2)

    assert np.allclose(yhat, [20.0, 21.0, 22.0])
    assert np.allclose(lower, [18.0, 19.0, 20.0])
    assert np.allclose(upper, [23.0, 24.0, 25.0])


class FakeForecast:
    predicted_mean = np.asarray([30.0, 31.0])

    def conf_int(self, alpha=0.1):
        return np.asarray([[28.0, 33.0], [29.0, 34.0]])


class FakeStatsModel:
    def get_forecast(self, steps):
        return FakeForecast()


def test_arima_predict_interval_uses_statsmodels_conf_int():
    model = ARIMAModel()
    model.fitted_model = FakeStatsModel()
    model._model_family = "arima"
    X = pd.DataFrame(index=range(2))

    yhat, lower, upper = model.predict_interval(X, alpha=0.2)

    assert np.allclose(yhat, [30.0, 31.0])
    assert np.allclose(lower, [28.0, 29.0])
    assert np.allclose(upper, [33.0, 34.0])


class FakeEstimator:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, X):
        return np.asarray([10.0, 20.0, 30.0])[: len(X)] + self.offset


class FakeTreeEnsemble:
    estimators_ = [FakeEstimator(-1.0), FakeEstimator(0.0), FakeEstimator(2.0)]

    def predict(self, X):
        return np.asarray([10.0, 20.0, 30.0])[: len(X)]


class FakePointTree:
    def predict(self, X):
        return np.asarray([5.0, 6.0])[: len(X)]


def _wrapper(cls, model):
    wrapper = cls.__new__(cls)
    wrapper.model = model
    return wrapper


def test_tree_wrappers_predict_interval_from_estimators():
    X = pd.DataFrame({"x": [1, 2, 3]})

    for cls in (XGBModel, LGBMModel, CatBoostModel):
        yhat, lower, upper = _wrapper(cls, FakeTreeEnsemble()).predict_interval(X, alpha=0.2)
        assert np.allclose(yhat, [10.0, 20.0, 30.0])
        assert np.all(lower < yhat)
        assert np.all(upper > yhat)


def test_tree_wrappers_predict_interval_falls_back_to_point_bounds():
    X = pd.DataFrame({"x": [1, 2]})

    yhat, lower, upper = _wrapper(XGBModel, FakePointTree()).predict_interval(X)

    assert np.allclose(yhat, [5.0, 6.0])
    assert np.allclose(lower, yhat)
    assert np.allclose(upper, yhat)


def _synthetic_regression(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-2.0, 2.0, n),
            "x2": rng.uniform(-2.0, 2.0, n),
            "x3": rng.normal(0.0, 1.0, n),
        }
    )
    noise = rng.normal(0.0, 1.0 + 0.5 * np.abs(X["x1"].values), n)
    y = pd.Series(3.0 * X["x1"].values - 2.0 * X["x2"].values + noise)
    return X, y


def test_real_tree_models_predict_interval_is_not_degenerate(monkeypatch):
    """真实 XGB/LGBM/CatBoost 上区间必须非零宽——防止只有假对象路径可用（历史缺陷）。"""
    monkeypatch.setenv("MODELCOMBINE_USE_GPU", "false")
    X, y = _synthetic_regression()
    X_train, y_train = X.iloc[:400], y.iloc[:400]
    X_test, y_test = X.iloc[400:], y.iloc[400:]

    for cls in (XGBModel, LGBMModel, CatBoostModel):
        model = cls(n_estimators=60, max_depth=3)
        model.fit(X_train, y_train)

        yhat, lower, upper = model.predict_interval(X_test, alpha=0.2)

        name = cls.__name__
        assert len(yhat) == len(X_test), name
        assert np.all(lower <= yhat + 1e-9), name
        assert np.all(yhat <= upper + 1e-9), name
        widths = upper - lower
        assert np.mean(widths > 1e-6) > 0.9, f"{name}: interval widths mostly zero (degenerate)"
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        assert coverage >= 0.5, f"{name}: nominal 80% interval covered only {coverage:.0%}"
