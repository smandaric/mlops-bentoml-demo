import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

xbgoost_regressor = bentoml.xgboost.get("xgb_winequality_regressor:latest").to_runner()

svc = bentoml.Service("wine_regressor", runners=[xbgoost_regressor])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = xbgoost_regressor.predict.run(input_series)
    return result
