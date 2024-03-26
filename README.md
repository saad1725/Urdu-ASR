

---

# Urdu ASR (Automatic Speech Recognition)

## Overview

Urdu ASR is a model designed for Automatic Speech Recognition in the Urdu language. It utilizes Facebook's wav2vec model for initial feature extraction and further fine-tunes the model using a custom dataset.

The feature extraction leverages Facebook's XLR model, which supports 55 languages, including Urdu.

Additionally, a custom Urdu language model is employed using an N-gram approach with the KenLM library.

## Installation

To install the necessary requirements, use the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

For inference use

```bash
python3 live_asr.py
```

For training code open the 
```bash
urdu_asr.ipynb
```


