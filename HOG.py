# 学习:https://github.com/PENGZhaoqing/Hog-feature

import cv2
import numpy as np
import math


class Hog_descriptor():

    def __init__(self, img, cell_size=8, bin_size=9):
        '''
        一个block由2x2个cell组成，步长为1个cell大小
        :param img: 输入图像(更准确的说是检测窗口)，要求灰度图像,对于行人检测图像大小一般为128x64 即是输入图像上的一小块裁切区域
        :param cell_size: 细胞单元的大小 如8，表示8x8个像素
        :param bin_size: 直方图的bin个数
        '''
        '''
        采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目的是调节图像的对比度，降低图像局部
        的阴影和光照变化所造成的影响，同时可以抑制噪音。采用的gamma值为0.5。 f(I)=I^γ
        '''
        self.img = img
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))  # gamma=0.5
        self.img = self.img * 255  # 反归一化
        self.cell_size = cell_size  # 单元边长
        self.bin_size = bin_size  # 直方图列数
        self.angle_unit = 180 / self.bin_size  # 直方图每个列的宽度
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 180 % self.bin_size == 0, "bin_size should be divisible by 180"

    # 主功能，计算图像的HOG描述符，顺便求 HOG - image特征图
    def extract(self):
        height, width = self.img.shape
        ''' 
        1、计算图像每一个像素点的梯度幅值和方向
        '''
        gradient_value, gradient_angle = self.global_gradient()
        gradient_value = abs(gradient_value)  # 避免负数,下面直方图统计时，正负抵消没了
        ''' 
        2、计算输入图像的每个cell单元的梯度直方图，形成每个cell的直方图descriptor
            若图像128x64 可以得到16x8个cell，每个cell由9个bin组成
        '''
        cell_gradient_value = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_value.shape[0]):
            for j in range(cell_gradient_value.shape[1]):
                # 当前cell
                cs = self.cell_size
                cell_gradient = gradient_value[i * cs:i * cs + cs, j * cs:j * cs + cs]
                cell_angle = gradient_angle[i * cs:i * cs + cs, j * cs:j * cs + cs]
                cell_gradient_value[i][j] = self.cell_gradient(cell_gradient, cell_angle)  # 计算出该cell的直方图
        '''
        3、将2x2个cell组成一个block，一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
           将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，
           这就是最终分类的特征向量
        '''
        hog_vector = []  # 默认步长为一个cell大小，一个block由2x2个cell组成，遍历每一个block
        for i in range(cell_gradient_value.shape[0] - 1):
            for j in range(cell_gradient_value.shape[1] - 1):
                block_vector = []  # 第[i][j]个block
                block_vector.extend(cell_gradient_value[i][j])
                block_vector.extend(cell_gradient_value[i][j + 1])
                block_vector.extend(cell_gradient_value[i + 1][j])
                block_vector.extend(cell_gradient_value[i + 1][j + 1])
                '''块内归一化梯度直方图，去除光照、阴影等变化，增加鲁棒性，健壮性'''
                # 使用l2范数归一化
                divider = math.sqrt(sum(i ** 2 for i in block_vector) + 1e-5)
                block_vector = [i / divider for i in block_vector]
                hog_vector.extend(block_vector)

        # *可选  将得到的每个cell的梯度方向直方图绘出，得到特征图，便于观察
        hog_image = self.gradient_image(cell_gradient_value)
        return hog_vector, hog_image

    # def extract()的辅助函数，使用sobel算子计算每个像素沿x、y的梯度和方向
    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
        # 180°~360°，减去180度
        gradient_angle[gradient_angle >= 180] -= 180
        gradient_angle[gradient_angle == 180] = 0  # 不知道为什么，上一句筛不掉恰好180的???
        return gradient_magnitude, gradient_angle

    # def extract()的辅助函数，为一个cell单元构建梯度方向直方图
    def cell_gradient(self, cell_gradient, cell_angle):
        bin = np.zeros(self.bin_size)  # 线性插值，[0°,20°,40°.....]
        # 遍历cell中的像素点
        for i in range(cell_gradient.shape[0]):
            for j in range(cell_gradient.shape[1]):
                value = cell_gradient[i][j]  # 当前像素的梯度幅值
                angle = cell_angle[i][j]  # 当前像素的梯度方向
                left_i = int(angle / self.angle_unit) % self.bin_size  # 左邻居index
                right_w = (angle - left_i * self.angle_unit) / self.angle_unit  # 右邻居权重比例,联想杠杆原理

                bin[left_i] += value * (1 - right_w)
                bin[(left_i + 1) % self.bin_size] += value * right_w
        return bin

    # 辅助函数，将梯度直方图转化为特征图像,便于观察
    def gradient_image(self, cell_gradient):
        '''
        将得到的每个cell的梯度方向直方图绘出，得到特征图
        :param cell_gradient: 输入图像的每个cell单元的梯度直方图,形状为[h/cell_size,w/cell_size,bin_size]
        :return: 特征图
        '''
        image = np.zeros(self.img.shape)  # 一张画布,和输入图像一样大
        cell_width = self.cell_size / 2
        max_mag = cell_gradient.max()
        # 遍历每一个cell
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y] / max_mag  # 获取第[i][j]个cell的梯度直方图 and 归一化
                # 遍历每一个bin区间
                for i in range(self.bin_size):
                    value = cell_grad[i]
                    angle_radian = math.radians(i * 20)
                    # 计算起始坐标和终点坐标，长度为幅值(归一化),幅值越大、绘制的线条越长、越亮
                    x1 = int(x * self.cell_size + cell_width + value * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + value * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - value * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - value * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(value)))
        return image


# 测试
if __name__ == '__main__':
    ori_img = cv2.imread('images/test/girl.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
    hog_vec, hog_img = Hog_descriptor(ori_img).extract()
    '''
        绘制两幅图，做对比
    '''
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 2.0 * 3.2))
    plt.subplot(1, 2, 1)
    plt.imshow(ori_img, cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap=plt.cm.gray)  # 输出灰度图
    plt.show()
