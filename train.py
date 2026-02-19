import argparse
from pathlib import Path
from ultralytics import YOLO


class YOLOv8Trainer:
    def __init__(
        self,
        model: str,
        data: str,
        epochs: int,
        imgsz: int,
        batch: int,
        device: str = "cuda",
        project: str = "runs/train",
        name: str = "exp",
        workers: int = 8,
        patience: int = 50,
        resume: bool = False,
    ):
        self.model_path = model
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project = project
        self.name = name
        self.workers = workers
        self.patience = patience
        self.resume = resume

        self._validate()
        self.model = YOLO(self.model_path)

    def _validate(self):
        if not self.model_path.endswith(".pt"):
            raise ValueError("Model must be a .pt file (e.g. yolov8l.pt)")

        if not Path(self.data).exists():
            raise FileNotFoundError(f"Dataset yaml not found: {self.data}")

    def train(self):
        self.model.train(
            data=self.data,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            project=self.project,
            name=self.name,
            workers=self.workers,
            patience=self.patience,
            resume=self.resume,
            pretrained=True,
            verbose=True,
            plots=True,          # enables TensorBoard plots
        )


def parse_args():
    parser = argparse.ArgumentParser("YOLOv8 Training Script")

    # REQUIRED (minimum parameters)
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)

    # OPTIONAL
    parser.add_argument("--model", type=str, default="yolov8l.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = YOLOv8Trainer(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
    )

    trainer.train()
