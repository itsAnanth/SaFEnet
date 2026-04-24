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
    aux_warmup: bool = False  # Whether to linearly warm up aux branch LRs for first 5 epochs (only applies to SaFENet)
    clip_grad: bool = False  # Whether to clip gradients to max norm of 1
    diff_lr: bool = False  # Whether to use differential learning rates for backbone and aux branches (only applies to SaFENet). If not set, all params use the same LR.