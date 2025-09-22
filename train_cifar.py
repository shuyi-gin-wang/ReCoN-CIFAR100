from dataclasses import dataclass

import torch
import torch.nn.functional as F

from train_common import get_loaders, resnet18_cifar100


@dataclass
class Config:
    data_dir: str = "./data"
    batch_size: int = 128
    epochs: int = 350
    warmup_epochs: int = 5
    lr: float = 0.1      # for batch_size 128 on single GPU
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    drop_rate: float = 0.0
    seed: int = 42
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, device, cfg, scaler):
    model.train()
    total, correct_class_count, correct_superclass_count, running_loss = 0, 0, 0, 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        
        target_class, target_superclass = targets
        target_class = target_class.to(device, non_blocking=True)
        target_superclass = target_superclass.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=cfg.amp):
            logits = model(images)
            loss_class = F.cross_entropy(logits["class"], target_class, label_smoothing=cfg.label_smoothing)
            loss_superclass = F.cross_entropy(logits["superclass"], target_superclass, label_smoothing=cfg.label_smoothing)
            loss = loss_class + loss_superclass

        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        predicted_class = logits["class"].argmax(dim=1)
        predicted_superclass = logits["superclass"].argmax(dim=1)
        correct_class_count += (predicted_class == target_class).sum().item()
        correct_superclass_count += (predicted_superclass == target_superclass).sum().item()
        total += images.size(0)


    avg_loss = running_loss / len(loader.dataset)
    
    accuracy_class = 100.0 * correct_class_count / max(total, 1)
    accuracy_superclass = 100.0 * correct_superclass_count / max(total, 1)

    return avg_loss, accuracy_class, accuracy_superclass


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct_class_count, correct_superclass_count, loss_sum = 0, 0, 0, 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        target_class, target_superclass = targets
        target_class = target_class.to(device, non_blocking=True)
        target_superclass = target_superclass.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits["class"], target_class) + F.cross_entropy(logits["superclass"], target_superclass)
        loss_sum += loss.item() * images.size(0)
        
        predicted_class = logits["class"].argmax(dim=1)
        predicted_superclass = logits["superclass"].argmax(dim=1)
        
        correct_class_count += (predicted_class == target_class).sum().item()
        correct_superclass_count += (predicted_superclass == target_superclass).sum().item()
        total += images.size(0)
        
    return loss_sum / total, 100 * correct_class_count / total, 100 * correct_superclass_count / total


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device
    print("Using device:", device)

    train_loader, test_loader = get_loaders(cfg.data_dir, cfg.batch_size)

    model = resnet18_cifar100(drop_rate=cfg.drop_rate).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    best_class_accuracy, best_superclass_accuracy = 0, 0

    for epoch in range(cfg.epochs):
        train_loss, train_class_accuracy, train_superclass_accuracy = train_one_epoch(model, train_loader, optimizer, device, cfg, scaler)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | train class accuracy {train_class_accuracy:.2f}% | train superclass accuracy {train_superclass_accuracy:.2f}%")

        eval_loss, eval_class_accuracy, eval_superclass_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | eval loss {eval_loss:.4f} | eval class accuracy {eval_class_accuracy:.2f}% | eval superclass accuracy {eval_superclass_accuracy:.2f}%")

        scheduler.step()

        if eval_class_accuracy > best_class_accuracy:
            best_class_accuracy = eval_class_accuracy
            best_superclass_accuracy = eval_superclass_accuracy
            torch.save({"model": model.state_dict(),
                        "acc": best_class_accuracy,
                        "epoch": epoch,
                        "cfg": cfg.__dict__},
                        "best_resnet18_cifar100_super.pt")
            print(f"*Saved checkpoint with acc {best_class_accuracy:.2f}%")

    print(f"Best class accuracy: {best_class_accuracy:.2f}% | Best superclass accuracy: {best_superclass_accuracy:.2f}%")


if __name__ == "__main__":
    main()
