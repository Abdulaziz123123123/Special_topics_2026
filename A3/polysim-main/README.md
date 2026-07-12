## Running the PolySim Project

The main entry point for the FOP PolySim experiments is:

```bash
python main.py
```

The script trains the multimodal speaker-identification model and evaluates it under four conditions:

* P3: German test data with face and voice
* P4: German test data with the face modality removed
* P5: English test data with face and voice
* P6: English test data with the face modality removed

### Installation

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Enter the FOP directory and start the experiment:

```bash
cd FOP
python main.py
```

The configuration is defined in `FOP/config.py`. The training and test CSV files are stored in `FOP/feature_tracker/`. Trained checkpoints are saved in `FOP/checkpoints/`, while detailed evaluation results are saved in `FOP/results/`.

The project uses pre-extracted ECAPA-TDNN audio embeddings and FaceNet visual embeddings. Therefore, the corresponding `ecappafeats` and `facenetfeats` directories must be available before running the experiment.
