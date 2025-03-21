import torch
import torch.nn as nn
import torch.nn.functional as F

class IOU(nn.Module):
    def __init__(
            self,
            epsilon: float = 1e-7
    ):
        super().__init__()
        self.epsilon = epsilon
    def forward(
            self,
            real_result: torch.Tensor,
            generated_result: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            real_result (torch.Tensor): 真实结果（独热编码）。[batch, 1, width, height]
            generated_result (torch.Tensor): 生成值结果（独热编码）。[batch, 1, width, height]

        Returns:
            (torch.Tensor): 尺寸为 [classes,]，对应不同类型的交并比。
        """
        generated_result = torch.softmax(generated_result, dim=1)

        # 计算逐类别交集和并集，沿空间维度 (W, H) 求和
        intersection = torch.sum(real_result * generated_result, dim=(2, 3)).float()  # [B, C]
        union = torch.sum(real_result, dim=(2, 3)).float() + torch.sum(generated_result, dim=(2, 3)).float() - intersection  # [B, C]

        # 计算每个类别的 IOU，添加 epsilon 保证数值稳定性
        iou_per_class = (intersection) / (union + self.epsilon)  # [B, C]
        return iou_per_class


class IOULoss(nn.Module):
    __doc__ = """
    交并比损失函数，这个函数用于给语义分割问题进行比较。
    映射规则：
    IoULoss = exp(1 - IoU) - 1
    """
    def __init__(
            self,
            epsilon: float = 1e-7
    ):
        """
        初始化方法。
        Args:
            epsilon (float): 一个为防止除以零的情况而被引入的很小的数字。
        """
        super().__init__()
        self.iou = IOU()
        self.epsilon = epsilon

    def forward(
            self,
            real_result: torch.Tensor,
            generated_result: torch.Tensor
    ):
        """
        Args:
            real_result (torch.Tensor): 真实标签（独热编码），已被预处理为[B, C, H, W]。
            generated_result (torch.Tensor): 模型输出的概率分布 [B, C, H, W]。
        """
        # 计算逐类IoU
        iou_per_class = self.iou(real_result, generated_result)

        # 计算批内平均mIoU
        iou_mean = torch.mean(iou_per_class)  # 标量

        # 计算损失
        loss = 1 - iou_mean
        return loss

if __name__ == '__main__':
    iou_instance = IOU()

