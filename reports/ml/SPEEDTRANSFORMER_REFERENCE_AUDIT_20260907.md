# SpeedTransformer Reference Code Audit - 2026-09-07

## Reference
- Commit: 2c0e7ac2aa52813c0899b2f8ba7c6394ace2e959

## Confirmed from reference code

### Window semantics
- window_size is configurable.
- DataProcessor default window_size is 100.
- Sliding windows operate on processed feature rows.
- When feature_columns=speed, window_size represents speed-feature rows.
- Therefore window_size=200 is not inherently 200 seconds.

### Short sequences
Sequences shorter than window_size are:
- zero padded
- accompanied by a padding mask

Therefore 200 real observations are not an architecture requirement.

### Geolife speed generation
Within each Trip_id:
1. sort by time
2. shift previous coordinate/time
3. compute Haversine distance
4. compute time difference
5. calculate speed
6. apply label-specific speed thresholds
7. retain filtered rows
8. feed retained speed rows into trajectory windows

Because filtering occurs after speed calculation and the model windowing uses
the retained row sequence, filtered rows can create non-uniform real-time
intervals inside the model sequence.

### Label-dependent filtering
Observed thresholds:
- bike: 0.5-80 km/h
- bus: 1-120 km/h
- car: 3-180 km/h
- train: 3-350 km/h
- walk: 0.1-15 km/h

These thresholds depend on the ground-truth transportation label.

They must not be copied directly into CANOPY inference preprocessing,
because the transportation mode is unknown at inference time.

### Split behavior
Reference DataProcessor randomly splits traj_id into train/val/test.
It does not enforce CANOPY UID-disjoint evaluation.

CANOPY experiments should preserve uid_split_v1 instead.

## Current decision
- Do not treat window_size=200 as a mandatory architecture constraint.
- Do not copy label-dependent Geolife speed filtering directly.
- Do not use the reference random traj_id split for CANOPY evaluation.
- Build a label-independent AI Hub adapter using the existing UID-disjoint split.

## Next
1. Define CANOPY label-independent speed preprocessing.
2. Build small AI Hub smoke-training dataset.
3. Run SpeedTransformer smoke training.
4. Only after that compare candidate window sizes.
