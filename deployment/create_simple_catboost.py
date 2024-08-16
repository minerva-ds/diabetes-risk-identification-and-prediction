import numpy as np
from catboost import CatBoostClassifier
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Train a simple CatBoost model
model = CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, loss_function='Logloss')
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)
model.fit(X, y)

# Convert the CatBoost model to ONNX
initial_type = [('float_input', FloatTensorType([None, 10]))]  # 10 is the number of features
onnx_model = onnxmltools.convert.convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model
with open("catboost_model_onnxmltools.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
