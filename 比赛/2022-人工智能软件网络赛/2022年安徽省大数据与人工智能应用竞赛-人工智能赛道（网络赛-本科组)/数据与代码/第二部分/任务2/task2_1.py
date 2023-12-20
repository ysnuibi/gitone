import cv2
import numpy as np


# def read_and_resize_images(img_files, new_width, new_height):
#     images = {}
#     for i, file in enumerate(img_files, start=1):
#         img = cv2.imread(file,cv2.COLOR_BGR2GRAY)
#         resized_image = cv2.resize(img, (new_width, new_height))
#         height, width, channels = resized_image.shape
#         sub_width = width // 2
#         sub_height = height // 2
#         x = width - sub_width
#         y = height - sub_height
#         sub_image = resized_image[y:, x:]
#         images[f'img{i}'] = sub_image
#     return images
from matplotlib import pyplot as plt


def read_and_resize_images(img_files):
    images = {}
    for i, file in enumerate(img_files, start=1):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 使用Harris角点检测
        gray = np.float32(img)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # 对结果进行膨胀以标记角点
        dst = cv2.dilate(dst, None)
        threshold = 0.01 * dst.max()
        for j in range(dst.shape[0]):
            for k in range(dst.shape[1]):
                if dst[j, k] > threshold:
                    cv2.circle(img, (k, j), 5, (0, 0, 255), -1)
        images[f'img{i}'] = dst
    return images


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_file = ['../../../user/Q2/2/img1.jpg', '../../../user/Q2/2/img2.jpg', '../../../user/Q2/2/img3.jpg',
            '../../../user/Q2/2/img4.jpg']

resized_images = read_and_resize_images(img_file)

# for name, img in resized_images.items():
#     cv_show(name,img)
merged_image = np.hstack((resized_images['img1'], resized_images['img2']))
merged_image = np.vstack((merged_image, np.hstack((resized_images['img3'], resized_images['img4']))))
cv_show('Merged Image', merged_image)

# # 对每张图像进行Harris角点检测并画出角点
# for i, image in enumerate(images):
#     # 转换为灰度图像
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 使用Harris角点检测
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#
#     # 对结果进行膨胀以标记角点
#     dst = cv2.dilate(dst, None)
#
#     # 设置阈值，只有大于阈值的角点才会被标记
#     threshold = 0.01 * dst.max()
#     for j in range(dst.shape[0]):
#         for k in range(dst.shape[1]):
#             if dst[j, k] > threshold:
#                 cv2.circle(image, (k, j), 5, (0, 0, 255), -1)
#
#     # 将处理后的图像放入画布中
#     canvas[i * image.shape[0]:(i + 1) * image.shape[0], :image.shape[1]] = image
#
# # 显示处理后的图像
# cv2.imshow('Harris Corner Detection', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()