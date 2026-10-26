"""Stage 1: LeJEPA pretraining before teacher-guided distillation."""

from train import main


if __name__ == "__main__":
    main(stage="pretrain")
