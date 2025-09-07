# Training Guide

## Training the GPT-2 Embedding Model

The CLI now supports training the model on TSV data containing sentence pairs and similarity labels.

### Basic Training Command

```bash
gptn-embedding train --train-data data_sets/train.tsv
```

### Full Training with Validation

```bash
gptn-embedding train \
  --train-data data_sets/train.tsv \
  --validation-data data_sets/dev.tsv \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --output-dir ./my_model \
  --checkpoint-every 5
```

### Training Options

- `--train-data`: Path to training TSV file (required)
- `--validation-data`: Optional validation TSV file  
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--loss`: Loss function - "contrastive", "cosine", or "mse" (default: contrastive)
- `--output-dir`: Directory for saving checkpoints (default: checkpoints)
- `--checkpoint-every`: Save checkpoint every N epochs (default: 1)
- `--resume-from`: Path to existing model to resume training

### TSV Data Format

The training data should be in TSV format with the following columns:

```
id	sentence1	sentence2	label
1	Hello world	Hi there	1
2	Good morning	Bad evening	0
```

Where:
- `id`: Unique identifier for the example
- `sentence1`: First sentence
- `sentence2`: Second sentence  
- `label`: Similarity label (1 = similar, 0 = dissimilar)

### Training Features

#### Visual Progress Monitoring
- Real-time progress bars showing training progress
- Live statistics including loss and accuracy metrics
- Estimated time remaining for each epoch

#### Graceful Interruption
- Press Ctrl+C at any time to stop training
- Automatically saves a checkpoint when interrupted
- Safe shutdown ensures no data loss

#### Model Checkpoints
- Automatic checkpoint saving at specified intervals
- Final trained model saved at completion
- Checkpoints can be used to resume training later

### Example Training Session

```bash
$ gptn-embedding train --train-data data_sets/train.tsv --validation-data data_sets/dev.tsv --epochs 5

🚀 Starting GPT-2 Embedding Model Training
==========================================
Loading training data from: data_sets/train.tsv
Loaded 1000 examples from data_sets/train.tsv
Dataset Statistics:
  Total examples: 1000
  Similar pairs (label=1): 500 (50.0%)
  Dissimilar pairs (label=0): 500 (50.0%)

Loading validation data from: data_sets/dev.tsv
Loaded 200 examples from data_sets/dev.tsv
Dataset Statistics:
  Total examples: 200
  Similar pairs (label=1): 100 (50.0%)
  Dissimilar pairs (label=0): 100 (50.0%)

Starting training with configuration:
  Learning rate: 0.0001
  Epochs: 5
  Batch size: 16
  Loss function: Contrastive
  Checkpoint every: 1 epochs

🟢 [00:01:23] [████████████████████████████████████████] 63/63 (00:00)
Epoch 1: train_loss=0.4521, train_acc=78.2%, val_loss=0.4892, val_acc=76.5%, time=1.4s
Checkpoint saved: checkpoints/checkpoint_epoch_1.bin

🟢 [00:02:48] [████████████████████████████████████████] 63/63 (00:00)  
Epoch 2: train_loss=0.3876, train_acc=82.1%, val_loss=0.4234, val_acc=79.8%, time=1.3s
Checkpoint saved: checkpoints/checkpoint_epoch_2.bin

...

🎉 Training completed!
Final model saved: checkpoints/final_model.bin
```

### Resuming Training

To continue training from a checkpoint:

```bash
gptn-embedding train \
  --train-data data_sets/train.tsv \
  --resume-from checkpoints/checkpoint_epoch_10.bin \
  --epochs 20
```

This allows you to:
- Resume training after interruption
- Extend training with more epochs
- Fine-tune with different parameters