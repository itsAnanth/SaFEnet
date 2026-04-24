import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.models.ResNet18 import get_resnet18
from src.models.proto import get_safenet
from src.models.Ladevic import get_ladevic
from src.models.MobileNetv2 import get_MobileNetV2
from src.models.Mulki import get_mulki

LABELS = {0: "REAL", 1: "FAKE"}

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("metrics", {}).get("config", {})
    model_name = config.get("model", None) if config else None

    if model_name is None:
        raise ValueError(
            "Checkpoint does not contain a model config. "
            "Pass --model explicitly."
        )

    if model_name == "safenet" or model_name == "safenet_spatial":
        branches = ['spatial'] if model_name == "safenet_spatial" else ('spatial', 'gradient', 'frequency')
        model = get_safenet(num_classes=2, branches=branches)
    elif model_name == "resnet18":
        model = get_resnet18(num_classes=2)
    elif model_name == "ladevic":
        model = get_ladevic(num_classes=2)
    elif model_name == "mobilenetv2":
        model = get_MobileNetV2(num_classes=2)
    elif model_name == "mulki":
        model = get_mulki(num_classes=2)
    else:
        raise ValueError(f"Unknown model type in checkpoint: '{model_name}'")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def predict(model, image_path: str, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()

    return pred, probs[pred].item(), probs


def resolve_checkpoint(path: str) -> str:
    """Accept either a directory (uses checkpoint_best.pt) or a .pt file."""
    if os.path.isdir(path):
        best = os.path.join(path, "checkpoint_best.pt")
        if not os.path.exists(best):
            raise FileNotFoundError(f"No checkpoint_best.pt found in {path}")
        return best
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint")
    parser.add_argument("checkpoint", type=str,
                        help="Path to checkpoint file (.pt) or checkpoint directory")
    parser.add_argument("images", nargs="+", metavar="IMAGE",
                        help="One or more image paths to classify")
    parser.add_argument("--model", type=str, default=None,
                        choices=["safenet", "safenet_spatial", "resnet18",
                                 "mobilenetv2", "ladevic", "mulki"],
                        help="Override model type (only needed if checkpoint lacks config)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    try:
        model, model_name = load_model(ckpt_path, device)
    except ValueError:
        if args.model is None:
            print("ERROR: Could not infer model type from checkpoint. Pass --model explicitly.")
            sys.exit(1)
        model_name = args.model
        # rebuild with explicit model name override
        ckpt = torch.load(ckpt_path, map_location=device)
        if model_name == "safenet":
            model = get_safenet(num_classes=2)
        elif model_name == "safenet_spatial":
            model = get_safenet(num_classes=2, branches=['spatial'])
        elif model_name == "resnet18":
            model = get_resnet18(num_classes=2)
        elif model_name == "ladevic":
            model = get_ladevic(num_classes=2)
        elif model_name == "mobilenetv2":
            model = get_MobileNetV2(num_classes=2)
        elif model_name == "mulki":
            model = get_mulki(num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

    print(f"Model: {model_name}  |  Device: {device}\n")
    print(f"{'Image':<45} {'Prediction':<8} {'Confidence':>10}  {'P(REAL)':>8}  {'P(FAKE)':>8}")
    print("-" * 85)

    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"{image_path:<45}  [FILE NOT FOUND]")
            continue
        pred, conf, probs = predict(model, image_path, device)
        name = os.path.basename(image_path)
        print(f"{name:<45} {LABELS[pred]:<8} {conf:>10.2%}  {probs[0].item():>8.2%}  {probs[1].item():>8.2%}")


if __name__ == "__main__":
    main()
