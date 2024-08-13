import matplotlib.pyplot as plt
import numpy as np
import cv2

# 임의의 이미지를 생성 (예: 400x400 크기의 흰색 이미지)
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 마스크 생성 (이미지와 동일한 크기의 검은색 배경)
mask = np.zeros_like(image)

# 'The Mighty Thor' 책의 좌표를 추정하여 마스크 적용 (좌표는 대략적인 값으로 조정 필요)
# 예시 좌표: x1, y1, x2, y2 = 220, 120, 320, 220
x1, y1, x2, y2 = 220, 120, 320, 220
mask[y1:y2, x1:x2] = [255, 0, 0]  # 빨간색으로 마스크 칠하기

# 원본 이미지와 마스크를 합성
masked_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

# 이미지를 출력
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title("Masked Image")
plt.axis('off')

plt.show()
plt.savefig('1.png')
