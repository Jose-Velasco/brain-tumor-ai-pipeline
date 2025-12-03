import torch
from monai.networks.nets.segresnet import SegResNet
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Export MONAI SegResNe to ONNX format."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the export on: 'cuda', 'cpu', or a specific GPU like 'cuda:0'."
    )

    parser.add_argument(
        "--dummy_input",
        type=int,
        nargs=5,
        metavar=("B", "C", "D", "H", "W"),
        default=[1, 4, 128, 128, 128],
        help=(
            "Dummy input tensor size for ONNX export. "
            "Default: 1, 4, 128, 128, 128 (B C H W D). "
            "Example override: --dummy_input 1 4 128 128 128"
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to existing model weights (.pt or .pth)"
    )

    parser.add_argument(
        "--onnx_out",
        type=str,
        required=True,
        help="Output ONNX file path, e.g. 'segresnet_kd_teacher.onnx'."
    )

    return parser.parse_args()

def main():
    args  = get_args()
    device = args.device

    teacher = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)

    teacher.eval()
    print(len(args.dummy_input))
    print(args.dummy_input)

    dummy_input = torch.randn(args.dummy_input, device=device)

    teacher.load_state_dict(torch.load(args.weights, map_location=device))

    torch.onnx.export(
        teacher,
        dummy_input,
        args.onnx_out,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

if __name__ == "__main__":
    main()