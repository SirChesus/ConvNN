print("Running export_to_onnx.py...")

import torch
from CNN import CNN

try:
    print("Loading model weights...")
    model = CNN(in_channels=3, num_classes=10)
    model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
    model.eval()

    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Exported model.onnx successfully")

except Exception as e:
    print("Error during export:", e)
