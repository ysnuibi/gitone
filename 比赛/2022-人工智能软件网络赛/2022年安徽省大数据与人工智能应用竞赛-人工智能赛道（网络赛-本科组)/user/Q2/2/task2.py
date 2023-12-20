import cv2
import numpy as np

# 读取四张图像
image1 = cv2.imread('./img1.jpg')
image2 = cv2.imread('./img2.jpg')
image3 = cv2.imread('./img3.jpg')
image4 = cv2.imread('./img4.jpg')

# 将四张图像放入一个列表中
images = [image1, image2, image3, image4]

# 创建一个空白画布，用于显示处理后的图像
canvas = np.zeros((800, 800, 3), dtype=np.uint8)

# 对每张图像进行Harris角点检测并画出角点
for i, image in enumerate(images):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Harris角点检测
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 对结果进行膨胀以标记角点
    dst = cv2.dilate(dst, None)

    # 设置阈值，只有大于阈值的角点才会被标记
    threshold = 0.01 * dst.max()
    for j in range(dst.shape[0]):
        for k in range(dst.shape[1]):
            if dst[j, k] > threshold:
                cv2.circle(image, (k, j), 5, (0, 0, 255), -1)

    # 将处理后的图像放入画布中
    canvas[i * image.shape[0]:(i + 1) * image.shape[0], :image.shape[1]] = image

# 显示处理后的图像
cv2.imshow('Harris Corner Detection', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
