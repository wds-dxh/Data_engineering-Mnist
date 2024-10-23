import cv2
import numpy as np
import os
import uuid

"""
    从给定的图像中检测并分割出方格，并将这些方格保存到指定的文件夹中。

    :param image_path: 输入图像的路径
    :param output_folder: 输出方格图像的文件夹路径，默认为 'output'
    :param square_size: 方格的宽度和高度，默认为 200 像素
    :param area_threshold: 判断方格的面积阈值，默认为 500
    """
def split_and_save_squares(image_path, output_folder='output', square_size=200, area_threshold=500):
    
    # 读取图片
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 边缘检测
    edges = cv2.Canny(binary, 100, 150, apertureSize=3)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 定义一个函数来判断是否为方格
    def is_square(cnt, area_threshold):
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        return len(approx) == 4 and cv2.contourArea(approx) > area_threshold and cv2.isContourConvex(approx)

    # 定义方格的宽度和高度
    width = square_size
    height = square_size
    count = 0

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 检测方格并保存
    for i, cnt in enumerate(contours):
        if is_square(cnt, area_threshold):
            # 计算方格的最小面积矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # 将box中的点按照从左上角开始，顺时针的顺序排列
            # 计算中心点
            center = tuple(np.mean(box, axis=0).astype(int))
            # 计算每个点与中心点的角度
            angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0]) * 180 / np.pi
            # 根据角度排序，得到顺时针的点顺序
            box = box[np.argsort(angles)]

            # 透视变换
            src_pts = box.astype("float32")
            # 设置目标点，确保不旋转图片
            # 这里假设width和height是目标图片的宽和高
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # 从原图中切割出方格
            warped = cv2.warpPerspective(image, M, (width, height))
            # 保存分割后的图片
            filename = os.path.join(output_folder, f'{str(uuid.uuid4())}.jpg')
            success = cv2.imwrite(filename, warped)
            count += 1

    # 显示检测结果
    # 设置显示图片的尺寸比例，例如缩小到原来的 0.5 倍
    scale_factor = 0.2

    # 获取原始图片的尺寸
    height, width = image.shape[:2]

    # 计算新的尺寸
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # 缩放图片
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 显示缩放后的图片
    cv2.imshow('Squares Detection', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return count

# 示例调用
if __name__ == "__main__":
    image_path = './133368a505ae090e001135d4ffa3e94c.jpeg'
    output_folder = './tools/output'
    square_size = 100           # 方格的宽度和高度
    area_threshold = 500        # 面积阈值

    count = split_and_save_squares(image_path, output_folder, square_size, area_threshold)
    print(f"检测到并保存了 {count} 个方格")