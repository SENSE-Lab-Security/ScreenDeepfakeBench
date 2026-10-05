"""Fine-tune released detector weights with a fresh optimizer."""

from train import main

if __name__ == "__main__":
    main(finetune=True)
