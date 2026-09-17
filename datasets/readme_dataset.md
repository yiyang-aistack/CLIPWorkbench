# CLIPWorkbench Dataset Generation

Both the ImageFolder tree and the CSV annotation are produced in one run. The
CSV uses **relative paths** and embeds an 80/20 train/val split per class.

### Run it

```bash
uv run python datasets/datasetGen.py
```

Customise `CLASSES`, `SAMPLES_PER_CLASS`, `IMG_SIZE` or `VAL_FRACTION` at the
top of the script to change the output – it is intentionally simple and
readable so you can copy-modify it to build **your own real dataset
pipelines** (scraping, augmentation, resizing, CSV annotation exports, etc.).

---

## Preparing Your Own Real Dataset

A few practical tips before you start:

1. **Label quality > data quantity.** A small, clean dataset beats a large
   noisy one, especially for CLIP which already has strong zero-shot priors.

2. **Class balance.** If classes are imbalanced, the training code applies
   inverse-frequency class weights (`compute_class_weights`) – but extreme
   skew still hurts. Consider up-sampling minority classes or collecting more
   data for them.

3. **Prompt engineering.** The default prompt template is
   `"a photo of a {class_name}"`. If your domain is specific (medical images,
   satellite, product shots), switch in `FineTuneStrategy` or override the
   prompt at inference time with something more descriptive, e.g.
   `"a photo of a {class_name}, a type of medical lesion"`.

4. **Image preprocessing.** CLIP expects 224×224 inputs (the `CLIPProcessor`
   handles resizing and normalization for you). Feed it full-resolution
   originals – downscaling is lossless enough at this stage.

5. **Hold-out set.** Always keep a separate test set that never touches the
   fine-tuning loop. The framework provides `val_split` for an internal
   validation split; add your own hold-out folder/rows for final evaluation.

---

## Folder layout reference

