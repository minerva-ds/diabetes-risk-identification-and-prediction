import onnx

# Load the minimal model and catboost_model
minimal_model_path = "models/model.onnx"
catboost_model_path = "models/catboost_model.onnx"

# Load the models using onnx
minimal_model = onnx.load(minimal_model_path)
catboost_model = onnx.load(catboost_model_path)

# Get the basic details of both models for comparison
minimal_model_info = {
    "ir_version": minimal_model.ir_version,
    "opset_import": minimal_model.opset_import,
    "producer_name": minimal_model.producer_name,
    "graph_inputs": len(minimal_model.graph.input),
    "graph_outputs": len(minimal_model.graph.output),
    "graph_nodes": len(minimal_model.graph.node),
}

catboost_model_info = {
    "ir_version": catboost_model.ir_version,
    "opset_import": catboost_model.opset_import,
    "producer_name": catboost_model.producer_name,
    "graph_inputs": len(catboost_model.graph.input),
    "graph_outputs": len(catboost_model.graph.output),
    "graph_nodes": len(catboost_model.graph.node),
}

print("Minimal Model Info:", minimal_model_info)
print("CatBoost Model Info:", catboost_model_info)
