# Roman Pashto to Native Pashto Transliteration Using BiLSTM and CTC

A deep learning-based system for converting **Roman Pashto (Latin-script Pashto)** into **Native Pashto script** using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network with **Connectionist Temporal Classification (CTC)**.

This repository contains the main model training notebook, the final parallel Roman Pashto–Native Pashto dataset used for training, a separately developed manually annotated word-level resource containing multiple Roman Pashto variants, and the final trained model files generated after model training.

---

## Overview

Pashto is traditionally written in its Native Pashto script, which is based on an Arabic-derived writing system. However, many Pashto speakers also use Roman/Latin characters when communicating through social media, messaging applications, and other digital platforms.

Roman Pashto does not follow a single standardized spelling system. The same Native Pashto word can therefore be written in several different ways using Roman characters.

This project addresses the transliteration problem:

```text
Roman Pashto
      ↓
Native Pashto
```

The proposed system uses a **character-level BiLSTM model combined with CTC** to learn the relationship between Roman Pashto input sequences and their corresponding Native Pashto output sequences.

The model learns these mappings from a parallel corpus rather than relying entirely on manually defined transliteration rules.

---

## Research Objective

The primary objective of this project is to develop a neural-network-based system capable of automatically converting Roman Pashto text into Native Pashto script.

The project aims to:

* Convert Roman Pashto into Native Pashto.
* Handle variations in Roman Pashto spelling.
* Learn character-level relationships between Roman and Native Pashto.
* Preserve the sentence and word structure of the input.
* Reduce dependence on manually designed transliteration rules.
* Provide resources for future Pashto Natural Language Processing (NLP) research.
* Investigate the use of recurrent neural networks for low-resource language transliteration.

---

# Model Architecture

The proposed system is based on a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network with **Connectionist Temporal Classification (CTC)**.

The overall processing pipeline can be represented as:

```text
Roman Pashto Input
        │
        ▼
Text Preprocessing
        │
        ▼
Character Encoding
        │
        ▼
Embedding
        │
        ▼
BiLSTM
        │
        ▼
TimeDistributed Dense
        │
        ▼
Layer Normalization
        │
        ▼
CTC
        │
        ▼
Native Pashto Output
```

---

## Character-Level Processing

The model operates at the **character level**.

Instead of representing a complete word as a single token, the input sentence is represented as a sequence of individual characters.

For example:

```text
Roman Pashto:

da wale?

Character sequence:

d → a →   → w → a → l → e → ?
```

This approach is particularly useful for Roman Pashto because spelling variations can occur at the character level.

For example, the same Pashto sound or word may be represented using different Roman spellings such as:

```text
d
da
dha
```

or:

```text
pa
pha
p
```

The model learns such variations from the training data.

---

# BiLSTM

A **Bidirectional Long Short-Term Memory (BiLSTM)** network processes the input sequence in both directions.

```text
Forward:
→ → → → →

Backward:
← ← ← ← ←
```

The forward LSTM captures information from preceding characters, while the backward LSTM captures information from following characters.

This allows the model to use contextual information from both sides of a character when learning the transliteration.

---

# Embedding

Before being processed by the BiLSTM, the input characters are converted into numerical vector representations through an embedding layer.

Conceptually:

```text
Roman character
      ↓
Numerical ID
      ↓
Embedding vector
      ↓
BiLSTM
```

The embedding layer allows the model to learn useful representations of the input characters during training.

---

# TimeDistributed Dense Layer

The TimeDistributed Dense layer applies a Dense transformation independently at each time step of the sequence.

In simple terms, it produces output character probabilities for each position in the sequence.

```text
BiLSTM output
      ↓
TimeDistributed Dense
      ↓
Character probabilities
```

These probabilities are subsequently used by the CTC-based training and decoding process.

---

# Layer Normalization

Layer Normalization is used within the neural network to normalize activations.

It helps stabilize the training process and can improve the consistency of model learning.

---

# Connectionist Temporal Classification (CTC)

The model uses **Connectionist Temporal Classification (CTC)** for its training objective.

CTC is useful for sequence-to-sequence tasks where the exact alignment between input and output characters is not explicitly provided.

Instead of requiring manually prepared character-level alignment, CTC allows the model to learn suitable alignments during training.

The process can be summarized as:

```text
Roman Pashto sequence
          │
          ▼
       BiLSTM
          │
          ▼
Character probabilities
          │
          ▼
         CTC
          │
          ▼
Native Pashto sequence
```

---

# Why BiLSTM + CTC?

A sequence-to-sequence architecture with attention could also be used for transliteration. However, this project uses BiLSTM with CTC because the task can be formulated as a character-level sequence conversion problem where explicit character-level alignment is not available.

BiLSTM provides contextual information from both directions, while CTC handles the alignment between the input and output sequences during training.

This combination is useful because:

* Roman and Native Pashto sequences can have different lengths.
* Explicit character-level alignment is not required.
* Roman Pashto contains spelling variations.
* The model can learn sequence relationships directly from parallel data.
* Character-level modeling is suitable for transliteration.

---

# Dataset

The repository contains two major data resources.

---

## 1. `Main_Dataset_for_Training_71K.txt`

`Main_Dataset_for_Training_71K.txt` is the **final parallel dataset used by the main BiLSTM training code**.

The file contains aligned Native Pashto and Roman Pashto sentences.

### Dataset Format

The dataset follows a **two-line parallel format**.

For every sentence pair:

* The first line contains the **Native Pashto sentence**.
* The second line contains the corresponding **Roman Pashto sentence**.

The same pattern continues throughout the file.

Conceptually:

```text
Native Pashto sentence 1
Roman Pashto equivalent 1

Native Pashto sentence 2
Roman Pashto equivalent 2

Native Pashto sentence 3
Roman Pashto equivalent 3

...
```

For example:

```text
دا ولې؟
da wale?

زه سبا ځم
za saba zam

په کور کې
pa kor ke
```

Therefore, every two consecutive lines represent one aligned:

```text
Native Pashto ↔ Roman Pashto
```

sentence pair.

### Important: Training Direction

Although the dataset stores the Native Pashto sentence first and the Roman Pashto sentence immediately underneath it, the **model is trained in the following direction**:

```text
Roman Pashto
      ↓
Native Pashto
```

Therefore:

```text
Dataset:

Native Pashto
Roman Pashto

       ↓

Training:

Roman Pashto → Input
Native Pashto → Target
```

The main training notebook reads the paired lines and uses the Roman Pashto sentence as the model input and the Native Pashto sentence as the expected output.

---

# Dataset Splitting

The final parallel dataset is divided into three subsets:

```text
70% → Training
15% → Validation
15% → Testing
```

### Training Set

The training set is used by the model to learn the mapping between Roman Pashto and Native Pashto.

### Validation Set

The validation set is used during training to monitor model performance and guide training decisions.

### Test Set

The test set is used to evaluate the trained model on data that was not used during model training.

The test set therefore provides an independent evaluation of the transliteration system.

---

# 2. `Top_12858_Txt.txt`

`Top_12858_Txt.txt` is a **separate manually annotated word-level resource** developed during the research.

It contains **12,858 Pashto words** with multiple Roman Pashto representations.

For each Native Pashto word, different Pashto speakers manually provided Roman Pashto variants.

Each word contains approximately **3–12 Roman variants**, depending on the word and available annotations.

Conceptually:

```text
Native Pashto Word
        │
        ├── Roman Variant 1
        ├── Roman Variant 2
        ├── Roman Variant 3
        ├── ...
        └── Roman Variant N
```

This resource was developed to capture the variation that exists in Roman Pashto writing.

### Important

`Top_12858_Txt.txt` is **not used directly in the main BiLSTM training code**.

The main model is trained using:

```text
Main_Dataset_for_Training_71K.txt
```

The manually annotated `Top_12858_Txt.txt` resource is provided separately and can support future research involving Roman Pashto spelling variation and transliteration.

---

# Final Trained Model Files

After completing the training process using the main BiLSTM notebook, the final trained model and its vocabulary/configuration files were generated.

The repository contains the following final output files:

### `Seq2Seq6_fast_Complt1.keras`

This is the **final trained Keras model file**.

It contains the learned parameters/weights and model architecture required to load the trained transliteration model.

The model can be loaded in a compatible TensorFlow/Keras environment and used for Roman Pashto → Native Pashto transliteration.

```python
from tensorflow import keras

model = keras.models.load_model("Seq2Seq6_fast_Complt1.keras")
```

### `vocab_fast_Complt1.json`

This JSON file contains the **vocabulary and character mapping information** used by the model.

The vocabulary is required to convert input characters into the numerical representations expected by the trained model and to interpret the model's output characters.

The model file and vocabulary file should therefore be kept together when using the trained system for inference.

### Final Model Files

```text
Seq2Seq6_fast_Complt1.keras
vocab_fast_Complt1.json
```

Together, these files represent the final trained model and the vocabulary information required for using the trained transliteration system.

---

# Repository Structure

```text
Roman-Pashto_Parallel_DataSet/
│
├── BiLSTM_150_with_dataSplits70_15_15.ipynb
│
├── Main_Dataset_for_Training_71K.txt
│
├── Top_12858_Txt.txt
│
├── Seq2Seq6_fast_Complt1.keras
│
├── vocab_fast_Complt1.json
│
└── README.md
```

---

# File Description

| File                                       | Description                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `BiLSTM_150_with_dataSplits70_15_15.ipynb` | Main notebook containing the BiLSTM model training and evaluation workflow                        |
| `Main_Dataset_for_Training_71K.txt`                  | Final parallel Native Pashto–Roman Pashto sentence dataset used for training                      |
| `Top_12858_Txt.txt`                        | Manually annotated word-level resource containing multiple Roman variants for 12,858 Pashto words |
| `Seq2Seq6_fast_Complt1.keras`              | Final trained BiLSTM/CTC Keras model                                                              |
| `vocab_fast_Complt1.json`                  | Vocabulary and character-mapping information required by the trained model                        |
| `README.md`                                | Documentation for the repository                                                                  |

---

# Main Model Training File

The main model training implementation is available in:

[`BiLSTM_150_with_dataSplits70_15_15.ipynb`](https://github.com/YunasShah/Roman-Pashto_Parallel_DataSet/blob/main/BiLSTM_150_with_dataSplits70_15_15.ipynb)

The notebook contains the main workflow for:

1. Loading the parallel dataset.
2. Reading Native Pashto and Roman Pashto sentence pairs.
3. Data preprocessing.
4. Character vocabulary creation.
5. Character encoding.
6. Sequence preparation.
7. Padding/truncation.
8. Training/validation/test splitting.
9. BiLSTM model construction.
10. CTC-based model training.
11. Validation.
12. Testing.
13. Transliteration/prediction.
14. Model evaluation.

---

# Training Configuration

The main experiment uses the following configuration:

| Parameter                 | Value                        |
| ------------------------- | ---------------------------- |
| Model                     | BiLSTM                       |
| Training objective        | CTC                          |
| Input representation      | Character-level              |
| Output representation     | Character-level              |
| Maximum sequence length   | 100                          |
| Learning rate             | 0.0001                       |
| Maximum epochs            | 150                          |
| Batch size                | 64                           |
| Training split            | 70%                          |
| Validation split          | 15%                          |
| Test split                | 15%                          |
| Transliteration direction | Roman Pashto → Native Pashto |

The notebook should be considered the authoritative source for the exact implementation and experimental configuration.

---

# Training Process

The complete training workflow can be summarized as:

```text
                 Parallel Dataset
                        │
                        ▼
              Roman / Native Pairing
                        │
                        ▼
                  Preprocessing
                        │
                        ▼
              Character Vocabulary
                        │
                        ▼
                Sequence Encoding
                        │
                        ▼
                 Dataset Splitting
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Training   Validation   Testing
             │
             ▼
       BiLSTM + CTC Model
             │
             ▼
          Training
             │
             ▼
       Final Trained Model
             │
       ┌─────┴─────┐
       ▼           ▼
   .keras        .json
    Model      Vocabulary
       │           │
       └─────┬─────┘
             ▼
      Roman Pashto →
      Native Pashto
```

---

# Training Control

The model is trained for up to **150 epochs**.

A learning-rate scheduling strategy is used to reduce the learning rate when validation performance stops improving.

This helps the model continue learning when progress becomes slower during training.

The training process can therefore be summarized as:

```text
Train Model
     ↓
Monitor Validation Performance
     ↓
Performance Improving?
     │
     ├── Yes → Continue Training
     │
     └── No → Reduce Learning Rate
```

---

# Using the Final Trained Model

The final trained model is stored in:

```text
Seq2Seq6_fast_Complt1.keras
```

The corresponding vocabulary is stored in:

```text
vocab_fast_Complt1.json
```

Both files are required for the complete inference pipeline.

The basic model-loading operation is:

```python
from tensorflow import keras

model = keras.models.load_model("Seq2Seq6_fast_Complt1.keras")
```

The vocabulary file should then be loaded by the inference/preprocessing code to ensure that Roman Pashto characters are encoded using the same mappings that were used during training.

For the complete preprocessing, encoding, CTC decoding, and transliteration workflow, refer to:

`BiLSTM_150_with_dataSplits70_15_15.ipynb`

---

# Evaluation Metrics

The transliteration system can be evaluated using character-level and word-level metrics.

## Character Error Rate (CER)

Character Error Rate measures the difference between the predicted and reference Native Pashto sequences at the character level.

It is particularly useful for transliteration because transliteration errors often occur at individual characters.

A lower CER indicates better character-level performance.

---

## Word Error Rate (WER)

Word Error Rate measures errors at the word level.

It provides a higher-level evaluation of whether complete words have been correctly transliterated.

A lower WER indicates better word-level performance.

---

## BLEU

BLEU can also be used to compare the generated Native Pashto output with the reference Native Pashto text.

It measures the similarity between generated and reference sequences using n-gram overlap.

A higher BLEU score generally indicates greater similarity to the reference text.

---

# Example

A simplified example of the intended transliteration process is:

```text
Input:
da wale?

Output:
دا ولې؟
```

Another example:

```text
Input:
pa

Output:
په
```

The actual predictions depend on the learned model and the input sequence.

---

# Roman Pashto Spelling Variation

One of the major challenges in Roman Pashto transliteration is the absence of a universally standardized Roman spelling system.

The same Native Pashto character, sound, or word may be represented in different ways.

For example:

```text
d
da
dha
```

may be used as alternative Roman representations in different contexts.

Similarly:

```text
pa
pha
p
```

may occur as different representations.

Such variation makes direct rule-based transliteration difficult.

The manually annotated `Top_12858_Txt.txt` resource was developed to document this variation by collecting multiple Roman representations of Pashto words from different Pashto speakers.

---

# Reproducibility

The repository provides the main model-training notebook, the training dataset, the manually annotated resource, and the final trained model files.

For reproducible experiments, it is recommended to use:

* The same dataset.
* The same preprocessing procedure.
* The same train/validation/test split.
* The same model architecture.
* The same hyperparameters.
* The same random seed, where specified in the notebook.
* Compatible versions of the required software libraries.

Because neural-network training can be affected by random initialization and data ordering, small differences may occur when reproducing the experiment in a different environment.

---

# Requirements

The project is implemented using Python and deep learning libraries.

The main environment may include:

```text
Python
TensorFlow / Keras
NumPy
Pandas
scikit-learn
```

The notebook can be executed in environments such as:

* Google Colab
* Kaggle Notebooks
* Jupyter Notebook

The exact package versions may depend on the environment used for training.

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/YunasShah/Roman-Pashto_Parallel_DataSet.git
```

Then:

```bash
cd Roman-Pashto_Parallel_DataSet
```

## 2. Open the Main Notebook

Open:

```text
BiLSTM_150_with_dataSplits70_15_15.ipynb
```

using Google Colab, Kaggle, or Jupyter Notebook.

## 3. Make the Dataset Available

Make sure:

```text
Main_Dataset_for_Training_71K.txt
```

is available at the path expected by the notebook.

## 4. Run the Notebook

Execute the notebook cells in sequence.

The notebook performs the complete model development workflow, including preprocessing, dataset preparation, model training, and evaluation.

---

# Research Contributions

This project contributes to Roman Pashto and Pashto NLP research through:

### 1. Parallel Dataset

A parallel Roman Pashto–Native Pashto sentence dataset used to train the neural transliteration model.

### 2. Manually Annotated Roman Pashto Resource

A word-level resource containing 12,858 Pashto words with multiple Roman Pashto variants contributed by different Pashto speakers.

### 3. Neural Transliteration Model

A character-level BiLSTM architecture designed for Roman Pashto → Native Pashto transliteration.

### 4. CTC-Based Training

The use of CTC allows the model to learn sequence alignment without requiring explicit character-level alignment annotations.

### 5. Final Trained Model

The repository provides the final trained Keras model and corresponding vocabulary file, allowing the trained system to be reused without retraining from scratch.

### 6. Low-Resource NLP

The project contributes computational resources and a neural approach for a relatively low-resource language and its Romanized digital form.

---

# Potential Applications

The developed resources and model can potentially support:

* Roman Pashto transliteration.
* Pashto text normalization.
* Pashto NLP systems.
* Roman Pashto search systems.
* Social media text processing.
* Pashto keyboard/input systems.
* Language-learning applications.
* Digital Pashto language resources.
* Computational linguistics research.
* Future neural machine transliteration systems.

---

# Limitations

Roman Pashto has substantial spelling variation because users may write the same Pashto word differently in Roman characters.

Consequently, model performance can be affected by:

* Unseen Roman spellings.
* Regional spelling differences.
* Speaker-specific writing conventions.
* Vocabulary coverage.
* Ambiguous Roman representations.
* Sentence length.
* Data distribution.

The manually annotated `Top_12858_Txt.txt` resource helps document Roman Pashto variation, but it is **not directly used as the training dataset for the main BiLSTM model**.

---

# Future Work

Possible future improvements include:

* Expanding the Roman Pashto parallel corpus.
* Increasing the number of manually annotated Roman variants.
* Comparing BiLSTM with Transformer-based architectures.
* Comparing CTC with attention-based sequence-to-sequence models.
* Developing larger-scale Pashto transliteration benchmarks.
* Investigating regional and speaker-specific Roman Pashto variation.
* Developing a deployable Roman Pashto transliteration application.
* Exploring hybrid neural and rule-based transliteration approaches.
* Evaluating the model on additional unseen Roman Pashto datasets.

---

# Citation

If you use the dataset, code, model, or manually annotated resource in academic research, please cite the associated research work.

```bibtex
@mastersthesis{shah2026romanpashto,
  author  = {Yunas Shah},
  title   = {Conversion of Latin-Pashto Script into Native-Pashto Using Recurrent Neural Networks},
  school  = {Shaheed Benazir Bhutto University Sheringal},
  year    = {2026}
}
```

---

# Acknowledgements

This research was conducted as part of an MS Computer Science research project.

Special acknowledgement is given to the Pashto speakers who contributed to the manual annotation of Roman Pashto variants in the `Top_12858_Txt.txt` resource.

---

# Author

**Yunas Shah**

MS Computer Science
Shaheed Benazir Bhutto University Sheringal
Khyber Pakhtunkhwa, Pakistan

Research interests include:

* Pashto Natural Language Processing
* Neural Machine Transliteration
* Deep Learning
* Recurrent Neural Networks
* Low-Resource NLP
* Computational Linguistics

---

# Repository

GitHub repository:

**Roman-Pashto_Parallel_DataSet**

Main model training notebook:

`BiLSTM_150_with_dataSplits70_15_15.ipynb`

Main training dataset:

`Main_Dataset_for_Training_71K5.txt`

Manually annotated resource:

`Top_12858_Txt.txt`

Final trained model:

`Seq2Seq6_fast_Complt1.keras`

Vocabulary:

`vocab_fast_Complt1.json`
