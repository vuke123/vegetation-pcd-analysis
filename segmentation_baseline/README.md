# Semantic Segmentation Baseline — RandLA-Net (Open3D-ML)

Local, research-oriented scaffold for 3D point-cloud semantic segmentation
of vineyard and orchard vegetation, built on **RandLA-Net** via **Open3D-ML
(PyTorch backend)**.

> **Status:** Baseline scaffold. No trained model, no labeled data yet.
> This module provides the code structure, dataset pipeline, and training
> loop so that labeling, training, and evaluation can begin immediately
> once labels are available.

---

## Why RandLA-Net?

RandLA-Net (Hu et al., 2020) was chosen as the first baseline because:

1. **Efficiency on large point clouds** — uses random sampling instead of
   expensive farthest-point-sampling, making it practical for million-point
   vineyard scenes on a single GPU.
2. **Local feature aggregation** — the Local Spatial Encoding + Attentive
   Pooling modules capture fine-grained 3D structure relevant to vine rows
   and tree canopies.
3. **Proven on outdoor LiDAR** — strong results on SemanticKITTI, Semantic3D,
   S3DIS, and Toronto3D — all outdoor or large-scale benchmarks.
4. **Available in Open3D-ML** — no need to implement from scratch.

## Why Open3D-ML?

- Provides a validated PyTorch implementation of RandLA-Net.
- Handles KNN, random sampling, and the encoder-decoder architecture.
- Allows focusing on data preparation and domain adaptation rather than
  re-implementing standard architectures.
- Easy to swap in other models later (e.g., KPConv, PointTransformer)
  within the same framework.

---

## Semantic classes

| ID | Class       | Description                         |
|----|-------------|-------------------------------------|
| 0  | unlabeled   | No label assigned (ignored in loss) |
| 1  | vine_row    | Grapevine row canopy                |
| 2  | olive_tree  | Olive tree canopy                   |
| 3  | other       | Ground, infrastructure, other veg.  |

These are starter classes. Refine after inspecting your data (you may split
`other` into `ground` + `infrastructure`, or add `inter-row` vegetation).

---

## Expected input features

The pipeline supports configurable feature sets via `config/randlanet_config.yaml`:

| Mode                          | Channels (beyond XYZ)                  |
|-------------------------------|----------------------------------------|
| `xyz`                         | coordinates only                       |
| `xyz_ndvi`                    | + NDVI                                 |
| `xyz_rgb_nir`                 | + Red, Green, Blue, NIR                |
| `xyz_rgb_nir_ndvi_intensity`  | + Red, Green, Blue, NIR, NDVI, Intensity |

Or define a custom `feature_list` in the config. Missing features are
zero-filled with a warning.

---

## Project structure

```
segmentation_baseline/
  README.md                        # this file
  requirements.txt                 # Python dependencies
  config/
    randlanet_config.yaml          # all hyperparameters and paths
  data/
    raw/                           # input LAS/LAZ files
    tiles/                         # prepared .npz tile blocks
    splits/                        # train.txt, val.txt, test.txt
    README_data_layout.md          # data format documentation
  scripts/
    prepare_dataset.py             # scan LAS -> validate -> tile
    train_randlanet.py             # training loop with validation
    infer_randlanet.py             # predict labels on new scenes
    evaluate_segmentation.py       # compute IoU, accuracy, confusion matrix
    dry_run_check.py               # verify environment + scaffold wiring
  src/
    __init__.py
    io_las.py                      # LAS/LAZ reading and writing
    features.py                    # per-point feature extraction
    tile_builder.py                # split scenes into overlapping tiles
    dataset.py                     # PyTorch Dataset for tiles
    model_wrapper.py               # RandLA-Net init, checkpoint I/O
    label_mapping.py               # class IDs, names, colours
    metrics.py                     # IoU, accuracy, confusion matrix
    postprocess.py                 # tile reassembly, spatial smoothing
    utils.py                       # config loading, logging, path helpers
  output/
    checkpoints/                   # saved model weights
    predictions/                   # inference output LAS files
    logs/                          # training logs
```

---

## Tile / block strategy

Large vineyard point clouds (100+ m extent, millions of points) are split
into overlapping rectangular tiles before training/inference:

- **Block size:** 10 m x 10 m (configurable)
- **Overlap:** 2 m between adjacent tiles
- **Max points per tile:** 65,536 (subsampled if exceeded)
- **Min points per tile:** 128 (tiles below this are discarded)

During inference, overlapping predictions are merged using nearest-centre
voting to avoid seam artifacts.

---

## What is already implemented

- [x] Config-driven pipeline (YAML)
- [x] LAS/LAZ I/O with safe dimension handling
- [x] Configurable feature extraction (xyz, ndvi, rgb, nir, intensity)
- [x] Scene-to-tile splitting with overlap and subsampling
- [x] Tile save/load (.npz format)
- [x] PyTorch Dataset with padding and augmentation
- [x] RandLA-Net initialization via Open3D-ML wrapper
- [x] Checkpoint save/load
- [x] Training loop with validation, early stopping, class weighting
- [x] Inference with tile-based prediction and reassembly
- [x] Evaluation: per-class IoU, mIoU, OA, confusion matrix
- [x] Graceful failure messages when labels/checkpoints are missing
- [x] Dry-run script to verify the scaffold without labels or GPU
- [x] Label mapping with colour definitions for visualization

## What is still TODO

- [ ] **Label a subset of point clouds** — the most critical next step
- [ ] Create train/val/test split files
- [ ] First training run and hyperparameter tuning
- [ ] Validate Open3D-ML RandLA-Net forward pass with real data shape
- [ ] Implement proper KD-tree based post-processing smoothing
- [ ] Add TensorBoard or wandb logging
- [ ] Test with different feature modes
- [ ] Compare against a second baseline (e.g., KPConv)
- [ ] Evaluate on held-out test scenes

---

## Quickstart

### 1. Install dependencies

```bash
cd segmentation_baseline

# Create a new venv (recommended — separate from the main pipeline venv)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify the scaffold

```bash
python scripts/dry_run_check.py
```

This checks imports, config parsing, feature extraction, tiling, and
dataset logic on synthetic data. No labels, GPU, or real data required.

### 3. Prepare data

Place your LAS files in `data/raw/`, then:

```bash
python scripts/prepare_dataset.py --config config/randlanet_config.yaml
```

This scans files, reports available features, and creates tiles in `data/tiles/`.

### 4. Label and split

Label your point clouds (e.g., in CloudCompare), then:
1. Re-run `prepare_dataset.py` to regenerate labeled tiles
2. Create `data/splits/train.txt` and `data/splits/val.txt`, each listing
   tile filenames (one per line)

### 5. Train

```bash
python scripts/train_randlanet.py --config config/randlanet_config.yaml
```

### 6. Inference

```bash
python scripts/infer_randlanet.py --config config/randlanet_config.yaml --input data/raw/new_scene.las
```

### 7. Evaluate

```bash
python scripts/evaluate_segmentation.py --config config/randlanet_config.yaml
```

---

## Next-step plan (for thesis work)

1. **Inspect data** — run `prepare_dataset.py` to see feature availability
   and point distributions across your vineyard/orchard scenes.
2. **Decide final class definitions** — start with the 3 classes above;
   refine after visual inspection. Consider whether `ground` should be
   separate from `other`.
3. **Label a small subset** — 5-10 scenes in CloudCompare. Focus on
   diversity: different vine row sizes, olive trees of varying maturity,
   and varied `other` content.
4. **Create train/val/test split** — 70/15/15 or similar. Keep entire
   scenes in the same split (no tile-level splitting across splits).
5. **Choose feature set** — start with `xyz_ndvi` since NDVI is already
   computed in the existing pipeline. Compare with `xyz` and
   `xyz_rgb_nir_ndvi_intensity` to see if extra features help.
6. **Fine-tune RandLA-Net** — start with default hyperparameters, then
   tune learning rate, block size, and class weights.
7. **Run inference on unseen scenes** — check predictions visually in
   CloudCompare before relying on metrics.
8. **Evaluate** — report per-class IoU and mIoU. Analyze failure cases
   (confused classes, edge effects, sparse regions).

---

## Integration with existing pipeline

This module is designed to work with the output of the existing vineyard
point-cloud processing pipeline in `scripts/`:

```
scripts/run_pipeline.sh
  -> SMRF ground classification
  -> Euclidean clustering
  -> PCD to NDVI LAS
  -> Merged cluster LAS files  ──>  segmentation_baseline/data/raw/
```

The cluster LAS files from the existing pipeline contain XYZ + NDVI and
can be placed directly in `data/raw/` for segmentation.

Alternatively, run segmentation on the **pre-ground-removal** point cloud
to let the network learn to distinguish ground from vegetation on its own.
This is a design decision for your thesis — both approaches have merit.

---

## Honest limitations

1. **No pretrained vine/olive model exists.** RandLA-Net has been trained
   on urban LiDAR (cars, buildings, roads). Transfer to agricultural
   vegetation is not guaranteed to work well without domain-specific
   fine-tuning.

2. **Labeling is still required.** This scaffold does not bypass the need
   for manual or semi-automatic point-cloud labeling. The quality of the
   segmentation is fundamentally limited by the quality and quantity of
   labeled training data.

3. **This is a scaffold, not a trained solution.** The scripts, model
   wrapper, and evaluation pipeline are real and functional, but no
   training has been performed. Results will only be meaningful after
   labeling and training.

4. **Single baseline architecture.** RandLA-Net is a good starting point
   but may not be the best model for this specific domain. Plan to
   compare with at least one other architecture (KPConv, PointTransformer)
   for a thorough thesis evaluation.

5. **Feature engineering is preliminary.** The optimal combination of
   input features (XYZ, NDVI, RGB, NIR, intensity, normals) for vineyard
   segmentation is an open research question that requires experimentation.

---

## References

- Hu, Q., et al. (2020). *RandLA-Net: Efficient Semantic Segmentation of
  Large-Scale Point Clouds.* CVPR 2020.
- Open3D-ML: http://www.open3d.org/docs/release/open3d_ml.html
