import onnx
from onnx import helper

# Load the ONNX model
model = onnx.load("models/catboost_model_no_zipmap.onnx")

# Find the TreeEnsembleClassifier node
for node in model.graph.node:
    if node.op_type == "TreeEnsembleClassifier":
        # Connect the 'probabilities' output
        node.output.append('probabilities')

# Save the modified model
onnx.save(model, "models/catboost_model_fixed.onnx")
