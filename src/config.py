from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-4
    num_workers: int = 4
    loss: str = "focal"  # Options: "focal", "bce", "ce"
    model: str = "safenet"  # Options: "safenet", "resnet"
