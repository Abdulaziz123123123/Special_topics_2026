# Special Topics 2026 - Assignment 3

This repository contains the implementation for Assignment 3: a complete pipeline that converts a spoken video into a target language while preserving the speaker voice characteristics and synchronizing lip movements.

## Tasks Covered

1. Create the pipeline
2. Extract embeddings from original and generated audio and visual data
3. Run the baseline and compare the results
4. Analyze the amount of synthetic data required for each identity

## Pipeline

The implemented pipeline follows these steps:

1. Extract audio from the input video using FFmpeg
2. Detect language and transcribe speech using Whisper
3. Translate the recognized text from English to German
4. Generate German speech using XTTS voice cloning
5. Match the generated audio duration to the original video
6. Apply Wav2Lip for lip synchronization
7. Extract ECAPA-TDNN audio embeddings and FaceNet visual embeddings
8. Run the FOP baseline model for comparison
9. Analyze required synthetic data per identity

## Main Results

### Embedding Similarity

| Modality | Encoder | Shape | Cosine Similarity |
|---|---|---:|---:|
| Audio | ECAPA-TDNN | 192 | 0.6962 |
| Visual | FaceNet | 512 | 0.9805 |

### Baseline Comparison

| Sample | True Label | Predicted Label | Confidence | Correct |
|---|---:|---:|---:|---|
| Original English | 0 | 0 | 0.9999 | Yes |
| Generated German | 0 | 0 | 0.9541 | Yes |

### Synthetic Data Analysis

A target of 50 samples per identity was selected.  
The analysis showed that 708 synthetic samples are needed in total to balance the training set to this target.

## Important Output Files

- Final report: `A3_complete_pipeline_report.pdf`
- Main notebook: `a3_pipeline.ipynb`
- Final lip-synced video: `outputs/video_lipsync/sample_lipsync_full_de.mp4`
- Task 2 summary: `outputs/task2_embedding_summary/task2_embedding_summary.csv`
- Task 3 results: `outputs/task3_baseline_results/a3_baseline_original_vs_generated_results_CORRECTED.csv`
- Task 4 analysis: `outputs/task4_synthetic_data_analysis/synthetic_needed_per_identity.csv`
- Synthetic data plot: `outputs/plots/task4_synthetic_needed_per_identity.png`

## Notes

Large external models and repositories are not included in this repo.

Wav2Lip must be downloaded separately from:
https://github.com/Rudrabha/Wav2Lip

The Wav2Lip checkpoints are also downloaded separately:
- `wav2lip_gan.pth`
- `s3fd.pth`

The SpeechBrain ECAPA-TDNN model is downloaded automatically when running the notebook.