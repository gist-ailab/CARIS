import pickle
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# 파일 경로 설정
file_path = '/SSDe/heeseon/src/CARIS/output/visualize/473.pickle'

# pickle 파일을 읽어오기
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 읽어온 데이터 출력 (원하는 방식으로 처리 가능)
print(data)
image = data['image']
targets = data['targets']
sentences = data['sentences']
attentions = data['attentions']

###  visualize image  ###
image = image.cpu().numpy()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

batch_size, channels, height, width = image.shape

for i in range(batch_size):
    img = image[i]  # i번째 이미지 선택

    # 채널별로 표준화 해제
    for c in range(channels):
        img[c] = img[c] * std[c] + mean[c]

    # 채널 순서를 (H, W, C)로 변환
    img = np.transpose(img, (1, 2, 0))

    # [0, 1] 범위로 클리핑
    img = np.clip(img, 0, 1)

    # 이미지 시각화
    plt.figure()
    plt.imshow(img)
    plt.axis('off')  # 축을 표시하지 않음
    plt.savefig(f'/SSDe/heeseon/src/CARIS/output/visualize/image_{i}.png', bbox_inches='tight', pad_inches=0)


###  print sentences  ###
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoded_sentences = []
for sentence in sentences:
    # sentence를 리스트로 변환 후, padding 토큰을 무시하고 decode
    decoded_sentence = tokenizer.decode(sentence[sentence > 0].tolist(), skip_special_tokens=True)
    decoded_sentences.append(decoded_sentence)
    print(decoded_sentence)

###  visualize mask  ###

# mask 텐서를 시각화
for i in range(targets['mask'].shape[0]):
    plt.figure(figsize=(5, 5))
    plt.imshow(targets['mask'][i].cpu().numpy(), cmap='gray')
    plt.title(f"Mask for class {targets['cls'][i].item()}")
    plt.suptitle(decoded_sentences[i], fontsize=14)
    plt.axis('off')
    plt.savefig(f'/SSDe/heeseon/src/CARIS/output/visualize/mask_{i}.png', bbox_inches='tight')
    plt.close()

###  visualize attentions  ###
sentences = sentences.cpu()
attentions = attentions.cpu()

# 시각화
for i in range(sentences.shape[0]):
    # 문장 디코딩
    decoded_sentence = tokenizer.decode(sentences[i], skip_special_tokens=True)
    
    # 토큰화된 단어 리스트
    tokens = tokenizer.convert_ids_to_tokens(sentences[i])

    # 시각화 내용 생성
    visualized_sentence = ""
    for j, token in enumerate(tokens):
        if attentions[i, j] == 1:
            visualized_sentence += f"**{token}** "  # 어텐션이 있는 토큰 강조
        else:
            visualized_sentence += f"{token} "

    # 시각화 출력
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, visualized_sentence, fontsize=14, ha='center')
    plt.axis('off')
    plt.title(f"Sentence {i + 1} Visualization")
    plt.savefig(f'/SSDe/heeseon/src/CARIS/output/visualize/sentence_attention_{i}.png', bbox_inches='tight')
    plt.close()