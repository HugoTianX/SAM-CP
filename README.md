# SAM-CP

This repository contains the implementation for the paper *Semantically Improved Adversarial Attack Based on Masked Language Model via Context Preservation*.

---

## Environment

1. **Create a Virtual Environment**:
   
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```
   
2. **Install PyTorch and CUDA**:
   
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   
3. **Install Other Dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

---

## Download

1. **Install Dependencies**  
   Ensure the `transformers` library is installed. If not, run:

   ```bash
   pip install transformers
   ```

2. **Download the Models**:
   
   ```bash
   python load_model_with_bert-base-uncased.py
   python load_model_with_debert-base.py
   ```

---

## Datasets

The following datasets will be used for fine-tuning after data preparation and will serve as baseline datasets for adversarial attacks. Please download and place them in the `./data` folder. For more details, see `./data/README.md`.

1. **Dataset Name: IMDB**  
   - **Download Link**: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 
   - **Storage Path**: `./data/IMDB/`

2. **Dataset Name: Yelp**  
   - **Download Link**: https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset
   - **Storage Path**: `./data/Yelp/`

3. **Dataset Name: SST-2**  
   - **Download Link**: https://github.com/clairett/pytorch-sentiment-classification/tree/master/data/SST2  
   - **Storage Path**: `./data/SST-2/`

---

## Model Training

This repository provides a JSON-formatted dataset template designed for fine-tuning tasks. The template includes original text and a set of candidate replacement words for specific positions in the text.

To train the model, run:

```bash
python train_xxx.py
```

---

## Attack

For the reproduction and usage of baseline attack methods, please refer to the following repositories:

- **BERT-Attack**: https://github.com/LinyangLee/BERT-Attack 
- **CLARE**: https://github.com/cookielee77/CLARE 
- **FBA**: https://github.com/MingzeLucasNi/FBA

These repositories provide implementations and guidelines for running baseline adversarial attacks on text models.

