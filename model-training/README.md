# SAM-CP

This repository contains the implementation for paper *Semantically Improved Adversarial Attack Based on Masked Language Model via Context Preservation*

### Environment

1. **Create a virtual environment**：
   
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```
   
2. **Install PyTorch and CUDA**
   
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   
3. **Install other dependencies**:：
   
   ```bash
   pip install -r requirements.txt
   ```

### Download 

1. **Install Dependencies**  
   Ensure the `transformers` library is installed, if not:

   ```bash
   pip install transformers
   ```

1. **Download the Model**
   `python load_model_with_bert-base-uncased.py`

### Model Training

​		Training MLM in train_xxx.py

