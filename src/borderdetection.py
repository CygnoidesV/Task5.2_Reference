import cv2
import os

STICKER_CONTOUR_COLOR = (36, 255, 12)
VIDEO_PATH = os.path.join(os.path.abspath(
    os.path.dirname(os.path.dirname(__file__))), "sample/video.mp4")

class BorderDetection:

    def ImagePreprocessing(self, frame):
        """@brief 图像预处理函数，接受一个帧作为输入，返回值一帧相同大小的图片。\n@param frame 一帧图片，大小为640x480。"""

        # 将帧转换为灰度图像。
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行模糊处理。
        frame = cv2.blur(frame, (3, 3))

        # 使用Canny边缘检测算法检测边缘。
        frame = cv2.Canny(frame, 30, 60, 3)

        # 创建一个9x9的矩形结构元素用于图像膨胀。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        # 对Canny边缘检测结果进行膨胀。
        frame = cv2.dilate(frame, kernel)

        return frame

    def find_contours(self, frame):
        """@brief 寻找3x3x3立方体的轮廓。\n返回一个大小为9的列表，一个元素代表一个轮廓，其元素为形如(x,y,w,h)的元组，x，y为轮廓的左上角坐标，w，分别为其宽和高。"""
        contours, hierarchy = cv2.findContours(
            frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []

        # 步骤 1/4: 筛选出近似为正方形的轮廓。
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                (x, y, w, h) = cv2.boundingRect(approx)

                # 计算边界矩形的宽高比。
                ratio = w / float(h)

                # 检查轮廓是否接近正方形。
                if ratio >= 0.8 and ratio <= 1.2 and w >= 30 and w <= 60 and area / (w * h) > 0.4:
                    final_contours.append((x, y, w, h))

        # 如果找到的轮廓少于9个，提前返回空列表。
        if len(final_contours) < 9:
            return []

        # 步骤 2/4: 查找具有9个邻居（包括自身）的轮廓，然后返回所有这些邻居。
        found = False
        contour_neighbors = {}
        for index, contour in enumerate(final_contours):
            (x, y, w, h) = contour
            contour_neighbors[index] = []
            center_x = x + w / 2
            center_y = y + h / 2
            radius = 1.5

            # 创建9个位置，代表当前轮廓的邻居。
            # 我们将使用这个来检查每个轮廓有多少邻居。
            # 当前轮廓只有当它是立方体的中心时，所有这些位置才能匹配。
            # 如果找到中心，我们也知道了所有的邻居轮廓，
            # 从而知道了所有的轮廓，因此知道了这个形状可以被认为是3x3x3立方体。
            neighbor_positions = [
                # 左上
                [(center_x - w * radius), (center_y - h * radius)],

                # 上中
                [center_x, (center_y - h * radius)],

                # 右上
                [(center_x + w * radius), (center_y - h * radius)],

                # 中左
                [(center_x - w * radius), center_y],

                # 中心
                [center_x, center_y],

                # 中右
                [(center_x + w * radius), center_y],

                # 左下
                [(center_x - w * radius), (center_y + h * radius)],

                # 下中
                [center_x, (center_y + h * radius)],

                # 右下
                [(center_x + w * radius), (center_y + h * radius)],
            ]

            for neighbor in final_contours:
                (x2, y2, w2, h2) = neighbor
                for (x3, y3) in neighbor_positions:
                    # 邻居位置位于每个轮廓的中心，而不是左上角。
                    # 逻辑: (左上 < 中心位置) 并且 (右下 > 中心位置)
                    if (x2 < x3 and y2 < y3) and (x2 + w2 > x3 and y2 + h2 > y3):
                        contour_neighbors[index].append(neighbor)

        # 步骤 3/4: 现在我们知道了每个轮廓有多少邻居，
        # 我们将循环遍历它们，找到具有9个邻居（包括自身）的轮廓，
        # 这是立方体的中心部分。如果找到它，'neighbors'实际上就是我们要找的所有轮廓。
        for (contour, neighbors) in contour_neighbors.items():
            if len(neighbors) == 9:
                found = True
                final_contours = neighbors
                break

        if not found:
            return []

        # 步骤 4/4: 当代码执行到这一部分时，我们找到了类似立方体的轮廓。
        # 下面的代码将根据它们的X和Y值从左上到右下对所有轮廓进行排序。

        # 首先根据y值对轮廓进行排序。
        y_sorted = sorted(final_contours, key=lambda item: item[1])

        # 拆分成3行，并对每行按x值排序。
        top_row = sorted(y_sorted[0:3], key=lambda item: item[0])
        middle_row = sorted(y_sorted[3:6], key=lambda item: item[0])
        bottom_row = sorted(y_sorted[6:9], key=lambda item: item[0])

        sorted_contours = top_row + middle_row + bottom_row
        return sorted_contours


border_detector = BorderDetection()

if __name__ == "__main__":

    cam = cv2.VideoCapture(VIDEO_PATH)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cam.read()
        if (frame is None):
            break

        key = cv2.waitKey(33) & 0xff
        
        # 退出
        if key == 27:
            break

        frame0 = border_detector.ImagePreprocessing(frame)
        contours = border_detector.find_contours(frame0)

        if len(contours) == 9:
            for index, (x, y, w, h) in enumerate(contours):
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                            STICKER_CONTOUR_COLOR, 2)
                
        cv2.imshow("Preview", frame0)


    cv2.destroyAllWindows()
    