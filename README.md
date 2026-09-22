Multimodal Brain Age Prediction (MRI + EEG)

Predict chronological age from structural MRI and resting-state EEG using a simple late-fusion CNN + MLP model.

This started as a portfolio project. I wanted one clean pipeline that actually combines the two data types I work with most (MRI and EEG), instead of two completely separate one-off scripts.

Why brain age?

Brain-age prediction (the difference between predicted age and real chronological age — the “brain age gap”) is a well-established idea in neuroimaging. Most papers stick to a single modality, almost always MRI (see Cole et al.).

I wanted to try a straightforward fusion of structural MRI and EEG band power, partly because I’ve worked with both modalities and haven’t seen many projects that put them together for this particular task.

This is not a novel method. It’s just a clean, end-to-end late-fusion baseline. The goal was to build something correct and complete, not to beat the literature.

 Data

I’m using the [LEMON dataset](https://openneuro.org/datasets/ds000221) (MPI-Leipzig Mind-Brain-Body) from OpenNeuro. It has T1-weighted structural MRI and resting-state EEG (eyes open / eyes closed) for roughly 220 subjects aged 20–80, which makes it a natural fit for age prediction.

I’m only using a subset of subjects (controlled by `--n-subjects`) mainly to keep disk usage and iteration time reasonable while developing.

`download_data.py` only pulls the T1 MRI and eyes-closed EEG files — not the full BIDS derivatives.

Important note on MRI preprocessing:
The raw LEMON T1s are not skull-stripped. `extract_mri_features.py` assumes the volumes have already been brain-extracted (e.g. with FSL BET). If you skip that step the model will still run, but the results will be noisy.

Pipeline

Run the scripts roughly in this order:

1.download_data.py
   Downloads T1 MRI + eyes-closed EEG for the selected subjects.

2. extract_mri_features.py
   Resamples the (already skull-stripped) T1 volumes to 64³ and normalizes them.

3. extract_eeg_features.py
   Extracts relative band-power features (delta, theta, alpha, beta, gamma) from the resting-state EEG.

4. match_subjects.py 
   Finds subjects that have both modalities and writes `data/processed/subjects.csv`.

5. train.py
   Trains the late-fusion model. Supports three modes: `fusion`, `mri_only`, `eeg_only`.

6. evaluate.py  
   Runs the trained model on the held-out validation set and saves a scatter plot.

Model

Late fusion:

- MRI branch → small 3D CNN → feature vector
- EEG branch → simple MLP on the band-power features → feature vector
- The two vectors are concatenated and passed through a final regression head that predicts age

Nothing fancy — just a clean, working multimodal baseline.

Results

On the validation split (20 % held out, fixed seed):

 Mode        MAE (years)        R²   
------------------------------------
Fusion             ~6.8            ~0.72 
 MRI only       ~7.4            ~0.65 
 EEG only       ~9.1            ~0.48 

Numbers are approximate and will shift a bit depending on the exact subject subset.  
Scatter plots are saved to `results/scatter_{mode}.png`.

Notes / Limitations

- This is intentionally a small, fast-to-train model. It is not meant to compete with large foundation models or heavy 3D architectures.
- The EEG features are deliberately simple (just relative band power). Better EEG representations would almost certainly improve both the EEG-only and fusion results.
- Subject selection and the train/val split use a fixed random seed so the numbers are reproducible.

Setup

bash
pip install -r requirements.txt