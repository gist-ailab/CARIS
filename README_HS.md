


## environment setting
```
conda create -n CARIS python=3.8
conda activate CARIS
# change CUDA version to 11.3
# install pytorch 1.11.0
pip install -r requirements.txt

```


## dataset & checkpoint setting
- download refcoco-family from https://github.com/dvlab-research/LISA?tab=readme-ov-file#training-data-preparation

```
pip install transformers requests
```
- download bert model from https://huggingface.co/google-bert/bert-base-uncased/tree/main