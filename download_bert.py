import os
import requests
from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"
save_dir = "./ckpt"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 토크나이저와 모델 다운로드 및 저장
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=save_dir)
model = BertModel.from_pretrained(model_name, cache_dir=save_dir)

# 파일 리스트
files = [
    "config.json", "pytorch_model.bin", "tf_model.h5", "tokenizer.json",
    "tokenizer_config.json", "vocab.txt", "flax_model.msgpack",
    "model.onnx", "model.safetensors", "rust_model.ot"
]

base_url = f"https://huggingface.co/google-bert/{model_name}/resolve/main/"

# 파일 다운로드
for file in files:
    url = base_url + file
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(save_dir, file), 'wb') as f:
            f.write(response.content)
            print(f"{file} 다운로드 완료")
    else:
        print(f"{file} 다운로드 실패")

print("모든 파일 다운로드 완료")
