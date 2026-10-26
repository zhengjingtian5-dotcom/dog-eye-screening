# Model card: dog eye screening baseline (v1)

## Summary

| | |
| --- | --- |
| Task | Binary screen of a close-up dog eye photo: `normal` vs `see_vet` |
| Model | Frozen [DINOv2-small](https://huggingface.co/facebook/dinov2-small) image embedding, then StandardScaler + logistic regression (C = 0.01, balanced class weights) |
| File | [`models/baseline.joblib`](models/baseline.joblib), scikit-learn 1.6.1 |
| Decision threshold | 0.527, the highest that reaches 95% see_vet recall on validation |
| Date | 25 October 2026 |
| Code | MIT licence |

## Intended use

A portfolio demonstration of careful evaluation for a health-adjacent image classifier. It may be used to explore how a screening tool could prompt owners to seek veterinary care.

**Not for**: diagnosing disease, deciding treatment, or replacing a vet. It is not a medical device and has not been validated clinically.

## Training data

Roboflow Universe ["Dog eyes"](https://universe.roboflow.com/dog-eyes) workspace, CC BY 4.0, cropped to the annotated eye with a 15% margin.

| Condition | Label |
| --- | --- |
| Healthy | normal |
| Conjunctivitis, blepharitis, entropion, eyelid tumour, pigmented keratitis, "mastopathy" | see_vet |
| Nuclear sclerosis | excluded (needs a vet's labelling rule) |

Provenance of the Roboflow data is undocumented. Its classes resemble Korea's AI Hub companion-animal eye dataset; this is unconfirmed. Labels have not been audited by a vet.

Splits are 70/15/15, stratified by label and grouped so that augmented copies and near-duplicates of a photo (same upload name, perceptual hash within 6 bits, or DINOv2 cosine ≥ 0.97) never cross splits.

## Evaluation

Held-out test split: 1,625 crops from 609 independent photo groups (75 healthy, 534 see_vet). 95% intervals are bootstrapped over photo groups.

| Metric | Value |
| --- | --- |
| Recall (see_vet) | 95.9% (94.1–97.5%) |
| Specificity | 99.1% (97.8–100%) |
| AUROC | 0.986 |
| AUPRC | 0.998 |
| Expected calibration error | 0.045 |

Per condition, the share flagged ranges from 100% (eyelid tumour) to **80.7% (entropion)**. Healthy eyes were falsely flagged 0.9% of the time.

## Known limitations and risks

- **Possible source shortcut.** Each condition was uploaded as a separate project, so the model may partly recognise the source (camera, lighting, framing) rather than the disease. The test split shares those sources, so the scores above likely overstate real-world performance. An external, vet-verified test set (DogEyeSeg4, requested) and owners' own phone photos are needed.
- **Entropion is missed about 1 time in 5.** It is a change in eyelid shape rather than colour, and tight crops hide it.
- **Under-confident probabilities.** Observed see_vet rates exceed predicted scores; the score shown in the demo is not a calibrated probability.
- **Few healthy examples** (75 independent groups in test), so specificity is imprecise.
- **Dogs only, one eye per photo, no eye detector.** Cats, whole-face photos and other species are out of scope.
- **Missed problems are the main harm.** The demo always says it is not a diagnosis, returns "not sure" for low-quality photos or scores just under the threshold, and advises seeing a vet when worried.

## How to reproduce

Run [`notebooks/baseline_colab.ipynb`](notebooks/baseline_colab.ipynb) on a free Colab GPU with a free Roboflow API key.
