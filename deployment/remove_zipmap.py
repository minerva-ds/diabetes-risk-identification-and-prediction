import onnx
from onnx import helper, shape_inference

# Load the ONNX model
model = onnx.load("models/catboost_model.onnx")

# Locate and remove the ZipMap node
new_nodes = []
for node in model.graph.node:
    if node.op_type != "ZipMap":
        new_nodes.append(node)

# Create a new graph without ZipMap
model.graph.ClearField('node')
model.graph.node.extend(new_nodes)

# Save the modified model
onnx.save(model, "models/catboost_model_no_zipmap.onnx")
