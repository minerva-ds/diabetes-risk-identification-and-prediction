import onnx

def print_model_info(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Check the model for errors
    onnx.checker.check_model(model)

    # Print the model's input names, shapes, and types
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"Shape: {input_shape}")
        print(f"Type: {input.type.tensor_type.elem_type}\n")

    # Print the model's output names, shapes, and types
    print("Model Outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}")
        output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Shape: {output_shape}")
        print(f"Type: {output.type.tensor_type.elem_type}\n")

    # Print the model's nodes (operations)
    print("Model Nodes:")
    for node in model.graph.node:
        print(f"Op Type: {node.op_type}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}\n")

if __name__ == "__main__":
    model_path = "models/catboost_model.onnx"  # Update this to your model's path
    print_model_info(model_path)


def inspect_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Check the model's inputs
    print("Model Inputs:")
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        data_type = input_tensor.type.tensor_type.elem_type
        print(f"Name: {name}, Shape: {shape}, Data Type: {onnx.TensorProto.DataType.Name(data_type)}")

    # Check the model's outputs (optional, for reference)
    print("\nModel Outputs:")
    for output_tensor in model.graph.output:
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        data_type = output_tensor.type.tensor_type.elem_type
        print(f"Name: {name}, Shape: {shape}, Data Type: {onnx.TensorProto.DataType.Name(data_type)}")

# Replace 'your_model.onnx' with the path to your ONNX model file
model_path = "models/catboost_model.onnx"  # Update this to your model's path
inspect_onnx_model(model_path)

