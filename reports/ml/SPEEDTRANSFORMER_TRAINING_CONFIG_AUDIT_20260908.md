# SpeedTransformer Training Configuration Audit - 2026-09-08

## Reference
Commit:
2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959

## Reference Geolife replication configuration
Observed in run_training_experiments.sh:

- window_size: 200
- stride: 50
- batch_size: 512
- num_epochs: 50
- patience: 7
- d_model: 128
- nhead: 8
- num_layers: 4
- dropout: 0.1
- learning_rate: 2e-4
- weight_decay: 1e-4
- gradient_clip: 1.0
- kv_heads: 4
- AMP enabled

## Reference checkpoint policy
train.py:
- monitors Validation loss
- saves checkpoint when Validation loss improves
- early stopping is based on Validation loss
- replication patience = 7

## Previous CANOPY smoke configuration
- window_size: 200
- batch_size: 64
- epochs: 3
- d_model: 64
- nhead: 8
- num_layers: 2
- dropout: 0.1
- learning_rate: 2e-4
- kv_heads: 4

The previous run was intentionally lightweight.
It was NOT a reference reproduction.

## Important
The 3-epoch window screening selected the highest observed Validation
Macro F1 only for smoke-screening interpretation.

That must not be confused with the reference training policy,
which selects checkpoints using Validation loss.

## Next
Run a longer-training diagnostic while keeping the previous lightweight
architecture fixed, so training duration can be investigated before
changing model capacity.

Internal Test remains untouched.
