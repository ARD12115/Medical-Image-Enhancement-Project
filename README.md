# MedhaDrishti — MRI Enhancement & ROI Segmentation

Yugma TechFest 2.0 · 24-Hour National Hackathon · JNNCE, Shivamogga · 24–25 July 2026

## What this is

A pipeline that enhances Brain (T1/T2/FLAIR) and Spine (T1/T2/STIR) MRI scans and segments
clinically relevant ROIs (tumors, edema, lesions, disc degeneration, herniation). Built to a
24-hour scope: dataset analysis → preprocessing → enhancement (Autoencoder / ResNet18 transfer
learning) → segmentation (basic U-Net, if time permits) → evaluation → report.

See `PRD_MedhaDrishti.docx` for scope/requirements and `TRD_MedhaDrishti.docx` for architecture
and module specs.

## Project structure

```
.
├── data/                   # NIfTI volumes (not committed — see Data below)
├── src/
│   ├── dataset_stats.py    # contrast / sharpness / noise analysis
│   ├── preprocessing.py    # denoise, rescale, histogram equalization
│   ├── enhancement_model.py
│   ├── segmentation_model.py
│   └── evaluate.py         # PSNR/SSIM, Dice/Jaccard, writes results.json
├── models/                 # trained weights (.pt)
├── results/
│   ├── results.json
│   └── plots/
├── docs/
│   ├── PRD_MedhaDrishti.docx
│   ├── TRD_MedhaDrishti.docx
│   ├── Final_Report_MedhaDrishti.docx
│   └── README.md
├── slides/
│   └── MedhaDrishti_Pitch.pptx
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # confirm GPU is visible
```

## Data

Place the hackathon-provided NIfTI files under `data/brain/` and `data/spine/`.
Brain MRI may additionally use the BRATS 2020 dataset. Spine MRI must use only the
hackathon-provided dataset.

## Running the pipeline

```bash
python src/dataset_stats.py      --data data/brain --out results/brain_stats.csv
python src/preprocessing.py      --data data/brain --out data/brain_preprocessed
python src/enhancement_model.py  --data data/brain_preprocessed --epochs 20 --out models/enhancer.pt
python src/segmentation_model.py --data data/brain_preprocessed --epochs 15 --out models/unet.pt   # optional, if on schedule
python src/evaluate.py           --models models/ --data data/brain_preprocessed --out results/results.json
```

Each stage writes its output before the next starts, so a partial run (e.g. enhancement done,
segmentation skipped) still produces usable results and a defensible demo.

## Metrics reported

- **Enhancement:** PSNR, SSIM (minimum). MSE/RMSE added only if ahead of schedule.
- **Segmentation:** Dice, Jaccard (minimum). Sensitivity/Specificity/F1 added only if ahead of schedule.

Full metric definitions are in `TRD_MedhaDrishti.docx`, Section 6.

## Team

- **Technical:** dataset analysis, preprocessing, model training, evaluation.
- **Documentation:** report, slides, demo video, final narrative.

## One-line pitch

"We implemented a transfer-learning approach using ResNet18 pretrained on ImageNet. MRI
preprocessing included denoising and histogram equalization. Enhancement was evaluated using
PSNR and SSIM, while segmentation was benchmarked with Dice and Jaccard metrics."
