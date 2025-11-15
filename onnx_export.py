import os
import argparse
import torch

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)


def get_network(params):
    # Model selection based on arguments
    if params.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=3)
    elif params.network == "mobilenetv1":
        model = MobileNetV1(embedding_dim=512)
    elif params.network == "mobilenetv2":
        model = MobileNetV2(embedding_dim=512)
    elif params.network == "mobilenetv3_small":
        model = mobilenet_v3_small(embedding_dim=512)
    elif params.network == "mobilenetv3_large":
        model = mobilenet_v3_large(embedding_dim=512)
    else:
        raise ValueError("Unsupported network!")

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='ONNX Export')

    parser.add_argument(
        '-w', '--weights',
        default='./weights/mobilenetv2_mcp.pth',
        type=str,
        help='Trained state_dict file path to open'
    )
    parser.add_argument(
        '-n', '--network',
        type=str,
        default='mobilenetv2',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large',
            'sphere20', 'sphere36', 'sphere64',
            'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size and input dimensions for ONNX export'
    )

    return parser.parse_args()


@torch.no_grad()
def onnx_export(params):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_network(params)
    model.to(device)

    # Load weights
    state_dict = torch.load(params.weights, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Set model to evaluation mode
    model.eval()

    # Generate output filename
    fname = os.path.splitext(os.path.basename(params.weights))[0]
    onnx_model = f'{fname}.onnx'
    print(f"==> Exporting model to ONNX format at '{onnx_model}'")

    # Create dummy input tensor
    x = torch.randn(1, 3, 112, 112).to(device)

    # Prepare dynamic axes if --dynamic flag is enabled
    dynamic_axes = None
    if params.dynamic:
        dynamic_axes = {
            'input': {
                0: 'batch_size',
            },
            'output': {
                0: 'batch_size'
            },
        }
        print("Exporting model with dynamic input shapes.")

    else:
        print("Exporting model with fixed input size: (1, 3, 112, 112)")

    # Export model to ONNX
    torch.onnx.export(
        model,                # PyTorch Model
        x,                    # Model input
        onnx_model,           # Output file path
        export_params=True,    # Store the trained parameter weights inside the model file
        opset_version=16,      # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=['input'],     # Model's input names
        output_names=['output'],   # Model's output names
        dynamic_axes=dynamic_axes  # Use dynamic or static depending on flag
    )

    print(f"Model exported successfully to {onnx_model}")


if __name__ == '__main__':
    args = parse_arguments()
    onnx_export(args)
