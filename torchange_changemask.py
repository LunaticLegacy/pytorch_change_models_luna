"""
Copyright (c) Zhuo Zheng and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

This .py file is written via:
     GitHub: 月と猫 - LunaNeko

I just rearranged the code for typing annotations and added the whole training workflow into a single file.
For anybody who not familiar with the workflow produced by library Ever, I removed the dataflow via that library.
Some functions' implementation were rewritten.

For the Chinese annotations, I'm gonna keep it for I was too lazy to translate my workflow into English as soon as
I'm free to do this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from einops import rearrange
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

import ever.module.loss as L

from PIL import Image  # 防止torchvision因环境问题报错的
from tqdm import tqdm
from datetime import datetime as dt

from dataclasses import dataclass   # 内置，提供自动初始化字典的支持。TypedDict不支持字典初始值。
from typing import List, Optional, Tuple, Union, Dict, Type, Callable, Any, Literal, LiteralString, TypedDict, overload
from _my_utils import (DataLoader, DualTempoDataLoader,
                       Evaluator, IOU)

# 这些东西是文本，需按常量对待。
CHANGE: str = 'change_prediction'
T1SEM: str = 't1_semantic_prediction'
T2SEM: str = 't2_semantic_prediction'


# 这里规定一个比对结果的字典内容……呃，我好像是在定义一个容器。
class ComparisonResultDict(TypedDict):
    confusion_matrix: torch.Tensor
    accuracy: Union[torch.Tensor, float]
    kappa: Union[torch.Tensor, float]
    macro_recall: Union[torch.Tensor, float]
    macro_precision: Union[torch.Tensor, float]
    avr_f1_score: Union[torch.Tensor, float]
    iou: Union[torch.Tensor, float]


class TotalComparisonDict(TypedDict):
    change_prediction: ComparisonResultDict
    t1_semantic_prediction: ComparisonResultDict
    t2_semantic_prediction: ComparisonResultDict


# 再规定一个参数注册字典。
class EvaluatorParamRegisterDict(TypedDict):
    param_name: str
    param_init_value: Any
    updating_rule: Literal["Ascend", "Descend"]
    update_func: Optional[Callable]


# 参数更新字典。
class EvaluatorParamUpdateDict(TypedDict):
    param_name: str
    param_value: Any


# 原始的损失函数，直接定义为函数。
def loss(
        s1_logit: torch.Tensor,
        s2_logit: torch.Tensor,
        c_logit: torch.Tensor,
        gt_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        s1_logit (torch.Tensor): 在时相T1时的语义分割结果。
        s2_logit (torch.Tensor): 在时相T2时的语义分割结果。
        c_logit (torch.Tensor): 变化情况分析结果。
        gt_masks (torch.Tensor): 2个时相的掩膜数据。
    Returns:
        (Dict[str, torch.Tensor]): 一个保存了每一类损失的dict。字段调度（键: 数据类型）如下：
        {
            's1_ce_loss': torch.Tensor,
            's1_dice_loss': torch.Tensor,
            's2_ce_loss': torch.Tensor,
            's2_dice_loss': torch.Tensor,
            'c_dice_loss': torch.Tensor,
            'c_bce_loss': torch.Tensor,
            'sim_loss': torch.Tensor
        }
    """
    s1_gt = gt_masks[:, 0, :, :].to(torch.int64)
    s2_gt = gt_masks[:, 1, :, :].to(torch.int64)

    # 2个语义分割损失，这里确定是正常的。
    s1_ce = F.cross_entropy(s1_logit, s1_gt, ignore_index=255)
    s1_dice = L.dice_loss_with_logits(s1_logit, s1_gt)

    s2_ce = F.cross_entropy(s2_logit, s2_gt, ignore_index=255)
    s2_dice = L.dice_loss_with_logits(s1_logit, s1_gt)

    # 语义分割损失设计完毕，现在开始变化检测。
    c_gt = (gt_masks[:, 0, :, :] == gt_masks[:, 1, :, :]).to(torch.float32)
    c_dice = L.dice_loss_with_logits(c_logit, c_gt)
    c_bce = L.binary_cross_entropy_with_logits(c_logit, c_gt)

    sim_loss = mse_loss(s1_logit, s2_logit, gt_masks)

    return {
        's1_ce_loss': s1_ce,
        's1_dice_loss': s1_dice,
        's2_ce_loss': s2_ce,
        's2_dice_loss': s2_dice,
        'c_dice_loss': c_dice,
        'c_bce_loss': c_bce,
        'sim_loss': sim_loss
    }


def mse_loss(s1_logit: torch.Tensor, s2_logit: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """ 均方根损失函数。"""
    c_gt = (gt_masks[:, 0, :, :] == gt_masks[:, 1, :, :]).to(torch.float32)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


class BiTemporalForward(nn.Module):
    def __init__(self):
        """
        类初始化时不需要任何参数。
        """
        super().__init__()

    def forward(
            self,
            x: torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        执行原始代码的bitemporal_forward方法。
        Args:
            x: 输入数据，其类型可能有：
                torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor]

        Returns:
            如果仅输入1个tensor，则返回Tuple[torch.Tensor, torch.Tensor]。
            否则，返回Tuple[List[torch.Tensor], List[torch.Tensor]]，且每一个list的元素数量等同于x的元素数量。

        """
        # 看起来我要怎么做？
        if isinstance(x, list) or isinstance(x, tuple):
            t1_feats: List[torch.Tensor] = []
            t2_feats: List[torch.Tensor] = []

            for feat in x:
                t1_feat, t2_feat = self._forward_workflow(feat)
                t1_feats.append(t1_feat)
                t2_feats.append(t2_feat)

        else:
            t1_feats, t2_feats = self._forward_workflow(x)

        return (t1_feats, t2_feats)

    def _forward_workflow(self, x: torch.Tensor) -> torch.Tensor:
        """ 前向传播工作流，这里单独封装。看起来应该是直接展开。 """
        return rearrange(x, 'b (t c) h w -> t b c h w', t=2)


class Squeeze(nn.Module):
    """ 一个用于快速压缩维度的nn.Module。 """

    def __init__(self, dim: int = 2):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)


class SpatioTemporalInteraction(nn.Sequential):
    """
    时空分析——对于时空关系而言是需要三维卷积的。
    注意：这是一个Sequential的，所以在实例化后只能使用__call__方法调度内部内容。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            type: Literal['conv3d', 'conv1plus2d'] = 'conv3d'
    ):
        """
        初始化方法。
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核尺寸。
            dilation (int): 卷积核空洞率。
            type (Literal['conv3d', 'conv1plus2d']): 层级类型，可选值：['conv3d', 'conv1plus2d']
        """
        if type == 'conv3d':
            padding = dilation * (kernel_size - 1) // 2
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, kernel_size=(2, kernel_size, kernel_size), stride=1,
                          dilation=(1, dilation, dilation),
                          padding=(0, padding, padding),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, kenrel_size=(2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(inplace=True) if kernel_size > 1 else nn.Identity(),
            )


class TemporalSymmetricTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int | List[int] | Tuple[int, ...],
            out_channels: int | List[int] | Tuple[int, ...],
            kernel_size: int,
            dilation: int = 1,
            interaction_type: Literal['conv3d', 'conv1plus2d'] = 'conv3d',
            symmetric_fusion: Optional[Literal['add', 'mul']] = 'add'
    ):
        """
        执行时空融合用的内容。
        Args:
            in_channels (int | List[int] | Tuple[int]): 输入通道数，可输入的类型有：int, list[int], tuple[int, ...]。
            out_channels (int | List[int] | Tuple[int]): 输出通道数。
            kernel_size (int): 卷积核尺寸。
            dilation (int): 卷积核空洞率。
            interaction_type (Literal['conv3d', 'conv1plus2d']): 层级类型，可选值：['conv3d', 'conv1plus2d']
            symmetric_fusion (Optional[Literal['add', 'mul']]): 对称融合方法，可选值：['add', 'mul']。
        """
        super(TemporalSymmetricTransformer, self).__init__()

        # 这个东西，二选一，但暂时不初始化。
        self.t: nn.ModuleList[nn.Sequential] | SpatioTemporalInteraction = None
        self.output_mode: str = ""  # 用于记录输出格式，可用值：multiple | single

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            # 如果输入的in_channels是列表或元组，则self.t设置为多层——详见上类。
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, dilation=dilation, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
            self.output_mode = "multiple"
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, dilation=dilation,
                                               type=interaction_type)
            self.output_mode = "single"

        self.symmetric_fusion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
        # 确定对称融合的表达式，记录匿名函数。
        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(
            self,
            features1: torch.Tensor | List[torch.Tensor],
            features2: torch.Tensor | List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        执行操作。
        Args:
            features1 (torch.Tensor | List[torch.Tensor]): 特征1，或者特征组1。
            features2 (torch.Tensor | List[torch.Tensor]): 特征2，或者特征组2。
            上述特征组1和特征组2的顺序建议别反序，默认认为特征1为时相1的数据，特征2为时相2的数据。

        Notes:
            特征1和特征2的数据类型必须相同，否则无法处理。
            且在实例化本类时，如果定义为处理多类，则不得输入单个torch.Tensor，必须是List[torch.Tensor]。

        Returns:
            按照输入的类型进行返回。如果输入类型均为torch.Tensor，则也返回1个torch.Tensor。
            如果输入List[torch.Tensor]，则返回List[torch.Tensor]。
        """

        # 如果features1是list类——看起来这个函数需要确保两个东西的类型相同。
        if self.output_mode == "multiple":
            # 例行公事：类型检查。
            if not isinstance(features1, list) or not isinstance(features2, list):
                raise ValueError(
                    f"2 features should be both List[torch.Tensor], but found {type(features1)} and {type(features2)}!")

            # 先按照列表推导式，将每个op应用到对应的f1和f2上。
            d12_features: List[torch.Tensor] = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                                                zip(self.t, features1, features2)]

            # 执行对称融合。
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features: List[torch.Tensor] = [self.symmetric_fusion(d12, d21) for d12, d21 in
                                                       zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        # 否则……看起来这里应该是一个torch.Tensor类。
        else:
            if not isinstance(features2, torch.Tensor) or not isinstance(features2, torch.Tensor):
                raise ValueError(
                    f"2 features should be both torch.Tensor, but found {type(features1)} and {type(features2)}!")

            # 执行对称融合，如果有。
            if self.symmetric_fusion:
                # 这里直接一个公式解决。
                change_features: torch.Tensor = self.symmetric_fusion(
                    self.t(torch.stack([features1, features2], dim=2)),
                    self.t(torch.stack([features2, features1], dim=2))
                )

            else:
                # 否则。
                change_features: torch.Tensor = self.t(torch.stack([features1, features2], dim=2))
            change_features: torch.Tensor = change_features.squeeze(dim=2)

        # 返回值可能有两种类型：
        return change_features


class ChangeMask(nn.Module):
    """
    好了，最头疼的部分——这里就是模型的主体了。
    这个东西需要被完全重组，它原本是在ever.register下被注册的，这里取消注册。
    """

    def __init__(
            self,
            semantic_classes: int,
            use_amp: bool = False,
            using_device: torch.device = torch.device("cpu"),
            batch_size: int = 1,
    ):
        """
        Args:
            semantic_classes (int): 最终分割为几类。
            use_amp (bool): 是否使用混合精度计算。
                如果启动了混合精度计算，则前向传播和反向传播的计算将在with torch.amp.autoscalar上下文中进行。
            using_device (torch.device): 使用的设备。
        """
        super().__init__()
        # 先将所有变量载入。
        self.semantic_classes = semantic_classes
        self.use_amp = use_amp
        self.device = using_device
        self.batch_size = batch_size

        # --------------------------------------开始定义模型----------------------------------------
        # 双时前向传播头。
        self.bitempo = BiTemporalForward().to(device=using_device)

        # 编码器，基于一个现有的编码器网络执行。
        self.encoder: nn.Module = smp.encoders.get_encoder('efficientnet-b0', weights='imagenet',
                                                           in_channels=6)
        self.encoder = self.encoder.to(device=using_device)  # 保证设备一致

        out_channels: Tuple[int, ...] = self.encoder.out_channels  # 基于基础模型的输出，看基础模型会输出多少个东西。
        decoder_channels: Tuple[int, ...] = tuple([p // 2 for p in out_channels])

        # 分割头
        self.semantic_decoder = UnetDecoder(
            encoder_channels=decoder_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        ).to(device=using_device)

        # 变化检测头
        self.change_decoder = UnetDecoder(
            encoder_channels=decoder_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        ).to(device=using_device)

        # 时空分析检测头
        self.temporal_transformer = TemporalSymmetricTransformer(
            decoder_channels, decoder_channels,
            3, interaction_type='conv3d', symmetric_fusion='add',
        ).to(device=using_device)

        # 两个卷积头——我不知道这两个东西都是干啥用的。
        # AI告诉我，第一个是语义分割，第二个是变化检测——我估计是吧。
        self.s = nn.Conv2d(16, self.semantic_classes, 1).to(device=using_device)
        self.c = nn.Conv2d(16, 1, 1).to(device=using_device)

        self.to(device=self.device)
        # --------------------------------------结束定义模型----------------------------------------
        # 辅助工具：IOU处理器
        self.iou_processor = IOU()

        # 直接起个优化器和学习率调度器。这个参数自己调吧。
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)

        self.epoches: nn.Parameter = nn.Parameter(torch.tensor(0, dtype=torch.int64), requires_grad=False)

        self.device_str = str(self.device).split(":")[0]  # 提取代表当前设备情况的字符串。
        # torch.amp的机制有些傻逼，还暂时只能接受字符串。

        # 如果启用了混合精度计算
        self.grad_scaler = torch.amp.GradScaler(
            device=self.device_str,
            enabled=self.use_amp
        )
        self.autocaster = torch.amp.autocast(
            device_type=self.device_str,
            dtype=torch.float16 if self.device_str == 'cuda' else torch.bfloat16,
            enabled=self.use_amp
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播请使用这个，这个东西可用于：
        ·训练期间前向传播
        ·验证期间前向传播（先使用model.eval()）
        ·模型部署或推理（先使用model.eval()）
        Notes:
            语义分割结果中，每个通道均对应一个类型的值的结果。在每一个位于[width, height]的像元内选取最大值通道作为最终预测类型。
            而变化检测结果只有1个通道。值域为[0, 1]。

        Args:
            x (torch.Tensor): 输入的数据。数据尺寸要求：[batch_size, 2 * bands, width, height]
                为什么是2 * bands？因为需要输入2张图像（2个时相）。（推荐使用torch.cat合并2个时相的数据）
        Returns:
            (Dict[str, torch.Tensor]): 返回值。描述：(键: 对应数据描述)
            {
                T1SEM: 时相1的语义分割结果。
                T2SEM: 时相2的语义分割结果。
                CHANGE: 变化检测结果，值域位于[0, 1]。
            }
            上述3个Tensor的尺寸均为[batch, m, width, height]
            其中，m为1或semantic_classes。
        """
        if self.use_amp:
            with self.autocaster:
                result = self._forward_workflow(x)
        else:
            result = self._forward_workflow(x)

        return result

    def _forward_workflow(
            self,
            x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        一个被封装进来的工作流内容。
        Args:
            x (torch.Tensor): 输入的数据。数据尺寸要求：[batch_size, 2 * bands, width, height]
                为什么是2 * bands？因为需要输入2张图像（2个时相）。（推荐使用torch.cat合并2个时相的数据）
        Returns:
            (Dict[str, torch.Tensor]): 返回值。描述：(键: 对应数据描述)
            {
                T1SEM: 时相1的语义分割结果。
                T2SEM: 时相2的语义分割结果。
                CHANGE: 变化检测结果，值域位于[0, 1]。
            }
            上述3个Tensor的尺寸均为[batch, m, width, height]
            其中，m为1或semantic_classes。
        """
        xm = self.encoder(x)

        t1_feats, t2_feats = self.bitempo(xm)
        # 先得到2个东西。

        s1_logit = self.s(self.semantic_decoder(*t1_feats))
        s2_logit = self.s(self.semantic_decoder(*t2_feats))

        temporal_features = self.temporal_transformer(t1_feats, t2_feats)
        c_logit = self.c(self.change_decoder(*temporal_features))

        # 返回的变化检测内容是一个sigmoid函数，有点意思。
        return {
            T1SEM: s1_logit.softmax(dim=1),
            T2SEM: s2_logit.softmax(dim=1),
            CHANGE: c_logit.sigmoid(),
        }

    def backward(
            self,
            logits: Dict[str, torch.Tensor],
            target: torch.Tensor,
            retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        我的反向传播方法，这个方法会返回一个loss。
        本方法仅可用于数据，请在训练模型期间调度该方法，目前该方法将调度工作流以进行操作。

        Args:
            logits (Dict[str, torch.Tensor]): 来自模型forward方法输出的数据，包含：
                [T1SEM]: 时相1的语义分割结果
                [T2SEM]: 时相2的语义分割结果
                [T1SEM]: 时相2-时相1的结果。
            target (torch.Tensor): 真实标签，会经过预处理变为三者的标签。
                形状为：[batch, 2, width, height]
                其中，dim1处，[0]为时相1的语义分割结果，[1]为时相2的语义分割结果。
                损失函数将自动处理真实值结果，无需手动进行。
            retain_graph (bool): 在本次反向传播过程中是否保留计算图。
        Returns:
            (Dict[str, torch.Tensor]): 损失函数得到的结果。
        """
        if self.use_amp:
            with self.autocaster:
                result = self._backward_workflow(logits, target, retain_graph=retain_graph)
        else:
            result = self._backward_workflow(logits, target, retain_graph=retain_graph)

        return result

    def _backward_workflow(
            self,
            logits: Dict[str, torch.Tensor],
            target: torch.Tensor,
            retain_graph: bool = False,
    ) -> torch.Tensor:
        """
        被封装起来的反向传播方法。
        Args:
            logits (Dict[str, torch.Tensor]): 来自模型forward方法输出的数据，包含：
                [T1SEM]: 时相1的语义分割结果
                [T2SEM]: 时相2的语义分割结果
                [T1SEM]: 时相2-时相1的结果。
            target (torch.Tensor): 真实标签，会经过预处理变为三者的标签。
                形状为：[batch, 2, width, height]
                其中，dim1处，[0]为时相1的语义分割结果，[1]为时相2的语义分割结果。
                损失函数将自动处理它。
            retain_graph (bool): 在本次反向传播过程中是否保留计算图。
        Returns:
            (Dict[str, torch.Tensor]): 损失函数得到的结果。
        """
        s1_logit, s2_logit, c_logit = logits[T1SEM], logits[T2SEM], logits[CHANGE]
        total_loss: Dict[str, torch.Tensor] = loss(s1_logit, s2_logit, c_logit, target)

        # 然后对整个模型的total_loss里所有的键求值。
        loss_keys = total_loss.keys()
        loss_sum: torch.Tensor = None  # 不初始化，但会在下面初始化。

        # 累计所有损失。
        for key in loss_keys:
            if loss_sum is None:
                loss_sum = total_loss[key]
            else:
                loss_sum = loss_sum + total_loss[key]

        # 开始进行梯度更新
        if self.use_amp:
            # 使用混合精度计算时
            self.grad_scaler.scale(loss_sum).backward(retain_graph=retain_graph)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            # 不用混合精度计算时
            loss_sum.backward(retain_graph=retain_graph)
            self.optimizer.step()

        # 在最后让优化器清空梯度。
        self.optimizer.zero_grad()
        # 返回总损失
        return loss_sum

    def calc_loss(
            self,
            logits: Dict[str, torch.Tensor],
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        只输出当前loss，但不进行反向传播的梯度下降过程。
        Args:
            logits (Dict[str, torch.Tensor]): 来自模型forward方法输出的数据，包含：
                [T1SEM]: 时相1的语义分割结果
                [T2SEM]: 时相2的语义分割结果
                [CHANGE]: 变化检测结果
            target (torch.Tensor): 真实标签，会经过预处理变为三者的标签。
                形状为：[batch, 2, width, height]
                其中，dim1处，[0]为时相1的语义分割结果，[1]为时相2的语义分割结果。
                损失函数将自动处理它。
        Returns:
            (Dict[str, torch.Tensor]): 损失函数得到的结果。
        """

        s1_logit, s2_logit, c_logit = logits[T1SEM], logits[T2SEM], logits[CHANGE]
        total_loss: Dict[str, torch.Tensor] = loss(s1_logit, s2_logit, c_logit, target)

        # 然后对整个模型的total_loss里所有的键求值。
        loss_keys = total_loss.keys()
        loss_sum: torch.Tensor = None  # 不初始化，但会在下面初始化。

        # 累计所有损失。
        for key in loss_keys:
            if loss_sum is None:
                loss_sum = total_loss[key]
            else:
                loss_sum = loss_sum + total_loss[key]

        # 在最后让优化器清空梯度。
        self.optimizer.zero_grad()
        # 返回总损失
        return loss_sum

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    # 从这里开始就是我从我自己的代码里搬运过来的东西，也算代码复用了吧。
    # 数据预处理部分。
    def _preprocess_real(
            self,
            image: torch.Tensor
    ) -> torch.Tensor:
        """
        对真实值图像的处理。
        真实值图像中，像素位置和原图像位置映射，而像素值代表该位置的数值。

        Note:
            被该方法处理后的结果是一系列的独热编码图像，且每种类型被映射为对应的波段。
            在每个波段里，1为“是本类型”，0为“不是本类型”。
        Args:
            image (torch.Tensor): 被化为tensor的图像。
        Returns:
            (torch.Tensor): 原始图像经过独热编码后的结果。
        """
        # torch内部有没有众数下采样的东西？
        out = (F.one_hot(image.long(), num_classes=self.semantic_classes).to(dtype=torch.float32)
               .squeeze(1).permute(0, 3, 1, 2))
        return out

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    # 从这里需要实现本类的验证逻辑。
    def comparison(
            self,
            logits: Dict[str, torch.Tensor],
            target: torch.Tensor,
            ignore_index: Optional[int] = None,
    ) -> TotalComparisonDict:
        """
        Args:
            logits (Dict[str, torch.Tensor]): 来自模型forward方法输出的数据，包含：
                [T1SEM]: 时相1的语义分割结果
                [T2SEM]: 时相2的语义分割结果
                [CHANGE]: 变化检测结果
            target (torch.Tensor): 真实标签，会经过预处理变为三者的标签。
                形状为：[batch, 2, width, height]
                其中，dim1处，[0]为时相1的语义分割结果，[1]为时相2的语义分割结果。
            ignore_index (Optional[int]): 忽略第几类。

        Returns:
            一个Dict[Dict]，其中内部dict为ComparisonResultDict，定义在上方。
            而外部dict的有效键为：["s1_out", "s2_out", "change_out"]，每个键的值均对应一个结果。
            分别对应时相1的语义分割结果、时相2的语义分割结果和变化检测结果。
        """
        s1_logit, s2_logit, c_logit = logits[T1SEM], logits[T2SEM], logits[CHANGE]

        result_s1: torch.Tensor = self._preprocess_real(target[:, 0, :, :])
        result_s2: torch.Tensor = self._preprocess_real(target[:, 1, :, :])
        result_change: torch.Tensor = (target[:, 0, :, :] == target[:, 1, :, :]).to(dtype=torch.float32).unsqueeze(1)

        # 执行工作流，得到语义分割图像的评估结果——基于混淆矩阵。
        s1_out: Dict[str, Union[torch.Tensor, float]] = self._comparison_workflow(
            real_result=result_s1,
            generated_result=s1_logit,
            classes=self.semantic_classes,
            ignore_index=ignore_index
        )

        s2_out: Dict[str, Union[torch.Tensor, float]] = self._comparison_workflow(
            real_result=result_s2,
            generated_result=s2_logit,
            classes=self.semantic_classes,
            ignore_index=ignore_index
        )

        change_out: Dict[str, Union[torch.Tensor, float]] = self._comparison_workflow(
            real_result=result_change,
            generated_result=c_logit,
            classes=2,
            ignore_index=ignore_index
        )

        result: TotalComparisonDict = {
            "change_prediction": change_out,
            "t1_semantic_prediction": s1_out,
            "t2_semantic_prediction": s2_out,
        }

        return result

    def _comparison_workflow(
            self,
            real_result: torch.Tensor,
            generated_result: torch.Tensor,
            classes: int = 7,
            ignore_index: Optional[int] = None,
    ) -> ComparisonResultDict:
        """
        对实际结果进行验证。
        Args:
            real_result (torch.Tensor): 真实结果，需要被转化。[batch, 1, width, height]
            generated_result (torch.Tensor): 模型生成的结果。[batch, type_num, width, height]
            classes (int): 最终结果有几类（不包含被忽略的类）。
            ignore_index (Optional[int]): 忽略第几类。
        Return:
            (ComparisonResultDict): 一个字典，定义参考上方类型。
                该字典保存了混淆矩阵、准确率、Kappa系数、F1分数、召回率（宏平均召回率）和精确率（宏平均精确率）。
        """
        # 对生成的结果，取每个像元最大值所在的类别。这两个是专门用于生成混淆矩阵的。
        generated_for_conf_mat = (torch.argmax(generated_result, dim=1)
                                  .reshape(-1, 1, generated_result.shape[2], generated_result.shape[3]))
        real_for_conf_mat = (torch.argmax(real_result, dim=1)
                             .reshape(-1, 1, real_result.shape[2], real_result.shape[3]))

        # 然后立即先计算IOU。
        iou: torch.Tensor = self.iou_processor(real_result, generated_result)
        # 计算混淆矩阵
        confusion_matrix: torch.Tensor = self.compute_confusion_matrix(
            real_for_conf_mat,
            generated_for_conf_mat,
            classes,
            self.device,
            ignore_index=ignore_index
        )

        # 计算准确率
        total: torch.Tensor = confusion_matrix.sum().item()
        accuracy: torch.Tensor = (torch.diagonal(confusion_matrix).sum().item() / total if total != 0 else 0.0)

        # 计算Kappa系数
        kappa: float = 0.0
        try:
            total_samples = total
            row_sums = confusion_matrix.sum(dim=1).float()  # 每个类别的真实样本数
            col_sums = confusion_matrix.sum(dim=0).float()  # 每个类别的预测样本数
            expected_agreement = (row_sums * col_sums).sum().item() / (total_samples ** 2)
            observed_agreement = accuracy
            kappa = ((observed_agreement - expected_agreement) / (1 - expected_agreement)
                     if (1 - expected_agreement) != 0 else 0.0)
        except ZeroDivisionError:
            pass

        # 计算召回率（宏平均召回率）
        # 对于每个类别，召回率 = 真正例 / (真正例 + 假负例)
        # 这里行和即为每个类别真实样本数，混淆矩阵对角线为真正例
        row_sums = confusion_matrix.sum(dim=1).float()  # 每个类别真实样本数
        diag = torch.diagonal(confusion_matrix).float()
        recalls = torch.zeros_like(diag)

        ranging_num: int = classes if ignore_index is None else (classes - 1)

        for i in range(ranging_num):
            if row_sums[i] != 0:
                recalls[i] = diag[i] / row_sums[i]
            else:
                recalls[i] = 0.0
        macro_recall = recalls.mean().item()

        # 计算精确率（宏平均精确率）
        # 对于每个类别，精确率 = 真正例 / (真正例 + 假正例)
        # 这里列和即为每个类别预测样本数
        col_sums = confusion_matrix.sum(dim=0).float()  # 每个类别预测样本数
        precisions = torch.zeros_like(diag)
        for i in range(ranging_num):
            if col_sums[i] != 0:
                precisions[i] = diag[i] / col_sums[i]
            else:
                precisions[i] = 0.0
        macro_precision = precisions.mean().item()

        avr_f1_score = float(
            torch.sum(2 * precisions * recalls / (precisions + recalls + 1e-7)) / classes)
        # 眼下，只能加一个epsilon，防止除以零了。
        result: ComparisonResultDict = {
            "confusion_matrix": confusion_matrix.to(dtype=torch.int32),
            "accuracy": accuracy,
            "kappa": kappa,
            "macro_recall": macro_recall,
            "macro_precision": macro_precision,
            "avr_f1_score": avr_f1_score,
            "iou": torch.mean(iou).item()
        }
        return result

    def compute_confusion_matrix(
            self,
            real_result: torch.Tensor,
            generated_result: torch.Tensor,
            num_classes: int,
            device: torch.device,
            ignore_index: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        构建混淆矩阵，行表示真实类别，列表示预测类别。
        Args:
            real_result (torch.Tensor): [batch, type_num, width, height]，被转化后的真实结果。
            generated_result (torch.Tensor): [batch, type_num, width, height]，直接尝试输出的图像语义分割结果。
            num_classes (int): 类别总数。
            device (torch.device): 执行计算的设备。
        Return:
            (torch.Tensor): 一个大小为[num_classes, num_classes]的混淆矩阵。
        """
        # 对真实结果计算每个像素的类别索引
        real_result_indices = real_result.view(-1)

        # 对模型输出的得分张量，选择每个像素最高分对应的类别索引
        generated_result_indices = generated_result.view(-1)

        indices = real_result_indices * num_classes + generated_result_indices

        confusion_matrix = (torch.bincount(
            indices,
            minlength=(num_classes) ** 2  # 输出长度，此时加入了背景类。
        ))

        confusion_matrix = confusion_matrix.reshape(num_classes, num_classes).to(device)

        if ignore_index is not None:
            confusion_matrix = self.minor_mat(confusion_matrix, ignore_index)

        return confusion_matrix

    @staticmethod
    def minor_mat(
            matrix: torch.Tensor,
            location: int | Tuple[int, int] | List[int]
    ) -> torch.Tensor:
        """
        求一个矩阵在特定位置上的余子矩阵。
        Args:
            matrix: 需要被处理的矩阵。
            location: 需要被删除的位置。
        """
        if isinstance(location, int):
            location = (location, location)

        i, j = location[0], location[1]

        # 删除第 i 行
        minor_matrix = matrix[torch.arange(matrix.size(0)) != i]
        # 删除第 j 列
        minor_matrix = minor_matrix[:, torch.arange(matrix.size(1)) != j]

        return minor_matrix


def train_workflow(
        model: nn.Module | ChangeMask,
        train_data: DualTempoDataLoader,
        val_data: DualTempoDataLoader,
        using_device: torch.device = torch.device("cpu"),
        epoch: int = 10,
        batch_size: int = 1,
        save_model_batches_interval: int = 10,
        compile_model: bool = False,
        model_save_path: str = "",
):
    """
    训练工作流。
    Args:
        model (nn.Module): 模型本体。
        train_data (DualTempoDataLoader): 训练数据集，双时相。
        val_data (DualTempoDataLoader): 验证数据集，双时相。
        epoch (int): 要训练几个epoch。
        using_device (torch.device): 使用的设备。
        batch_size (int): Batch大小。这个大小会在训练集和测试集上同时生效。
        save_model_batches_interval (int): 每隔多少回保存一次模型。
        compile_model (bool): 是否编译模型。
        model_save_path (str): 在哪里保存模型。
    """
    evaluator: Evaluator = Evaluator()  # 初始化一个评估器。

    # 需要监控的参数列表。
    params_monitoring: List[EvaluatorParamRegisterDict] = [
        # 时相1的语义分割评估参数
        {"param_name": "Accuracy_period1", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Kappa_period1", "param_init_value": torch.Tensor([-1.0]), "updating_rule": "Ascend"},
        {"param_name": "mIoU_period1", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "F1_period1", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Recall_period1", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Precision_period1", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},

        # 时相2的语义分割评估参数
        {"param_name": "Accuracy_period2", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Kappa_period2", "param_init_value": torch.Tensor([-1.0]), "updating_rule": "Ascend"},
        {"param_name": "mIoU_period2", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "F1_period2", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Recall_period2", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Precision_period2", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},

        # 对变化情况的评估参数
        {"param_name": "Accuracy_change", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Kappa_change", "param_init_value": torch.Tensor([-1.0]), "updating_rule": "Ascend"},
        {"param_name": "mIoU_change", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "F1_change", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Recall_change", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
        {"param_name": "Precision_change", "param_init_value": torch.Tensor([0.0]), "updating_rule": "Ascend"},
    ]

    # 给评估器传入参数。
    evaluator.register_multiple_params(params_monitoring)

    if compile_model:
        model = torch.compile(model)
        print("Compile model enabled. It will be a slow start but will be VERY FASTER after compile completed.")

    for ep in range(epoch):
        # 训练
        model.train()
        with tqdm(total=len(train_data) // batch_size, ncols=200) as bar:
            for i in range(len(train_data) // batch_size):
                # 内存监视器。
                current_mem = torch.cuda.memory_allocated(using_device)
                reserved_mem = torch.cuda.memory_reserved(using_device)
                total_mem = torch.cuda.get_device_properties(using_device).total_memory

                # 对于训练数据集而言，如果batch_size大于1则启动这里。
                if batch_size == 1:
                    x, y = train_data[i]
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)
                else:
                    x, y = train_data[i * batch_size: (i + 1) * batch_size: 1]
                    x = torch.stack(x, dim=0)
                    y = torch.stack(y, dim=0)

                result = model.forward(x)
                loss_value = model.backward(result, y)

                bar.update(1)
                bar.set_description(
                    f"Epoch {model.epoches.item()} | Iteration {ep} | Loss: {loss_value.item():.8} | "
                    f"Cuda mem: {current_mem / 1024 ** 2:.2f}/{total_mem / 1024 ** 2:.2f} MB | "
                )

        # 验证
        model.eval()
        eval_result = eval_workflow(
            model=model,
            val_data=val_data,
            using_device=using_device,
            batch_size=batch_size,
            now_epoch=ep,
        )  # 这个东西会返回一个字典，作为解包结果。

        # 然后更新评估器。
        param_updating_list: List[Dict[str, Any]] = [
            # 时相1的语义分割评估参数
            {"param_name": "Accuracy_period1", "param_value": eval_result[T1SEM]["accuracy"]},
            {"param_name": "Kappa_period1", "param_value": eval_result[T1SEM]["accuracy"]},
            {"param_name": "mIoU_period1", "param_value": eval_result[T1SEM]["iou"]},
            {"param_name": "F1_period1", "param_value": eval_result[T1SEM]["avr_f1_score"]},
            {"param_name": "Recall_period1", "param_value": eval_result[T1SEM]["macro_recall"]},
            {"param_name": "Precision_period1", "param_value": eval_result[T1SEM]["macro_precision"]},
            # 时相2的语义分割评估参数
            {"param_name": "Accuracy_period2", "param_value": eval_result[T2SEM]["accuracy"]},
            {"param_name": "Kappa_period2", "param_value": eval_result[T2SEM]["accuracy"]},
            {"param_name": "mIoU_period2", "param_value": eval_result[T2SEM]["iou"]},
            {"param_name": "F1_period2", "param_value": eval_result[T2SEM]["avr_f1_score"]},
            {"param_name": "Recall_period2", "param_value": eval_result[T2SEM]["macro_recall"]},
            {"param_name": "Precision_period2", "param_value": eval_result[T2SEM]["macro_precision"]},
            # 对变化情况的评估参数
            {"param_name": "Accuracy_change", "param_value": eval_result[CHANGE]["accuracy"]},
            {"param_name": "Kappa_change", "param_value": eval_result[CHANGE]["accuracy"]},
            {"param_name": "mIoU_change", "param_value": eval_result[CHANGE]["iou"]},
            {"param_name": "F1_change", "param_value": eval_result[CHANGE]["avr_f1_score"]},
            {"param_name": "Recall_change", "param_value": eval_result[CHANGE]["macro_recall"]},
            {"param_name": "Precision_change", "param_value": eval_result[CHANGE]["macro_precision"]},
        ]   # 共计18个参数
        # 然后，批量更新参数。这18个参数按照上方预设都一样。
        evaluator_result: List[bool] = evaluator.update_multiple_params(param_updating_list)

        # TODO: 然后你就可以自己规定保存规则，基于更新参数的结果进行保存。
        # 定期保存模型
        if ep % save_model_batches_interval == 0:
            save_workflow(model, model_save_path, compile_model, ep)

        # 更新模型的epoch数
        model.epoches += 1

def save_workflow(
        model: nn.Module | ChangeMask,
        model_save_path: str,
        compile_model: bool = False,
        ep: int = 0,
):
    """
    保存模型的工作流。
    Args:
        model (nn.Module): 模型本体。
        model_save_path (str): 保存模型的文件夹位置。
        compile_model (bool): 是否已编译模型。
        ep (int): 当前经过了几个epoch。
    """
    tempo = dt.now()
    tempo_str: str = tempo.strftime("%Y_%m_%d_%H_%M_%S")
    model_save_path_pe: str = model_save_path + tempo_str + f"_{model.epoches.item()}.pth"
    if compile_model:
        torch.save(model._orig_mod.state_dict(), model_save_path_pe)
    else:
        torch.save(model.state_dict(), model_save_path_pe)
    print(f"\n | Model saved in iteration {ep} at {model_save_path_pe}.")

def eval_workflow(
        model: nn.Module | ChangeMask,
        val_data: DualTempoDataLoader,
        using_device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        now_epoch: int = 0,
) -> TotalComparisonDict:
    """
    对模型进行一轮验证，并输出它的验证结果。
    Args:
        model (nn.Module): 模型本体。
        val_data (DualTempoDataLoader): 验证数据集，双时相。
        using_device (torch.device): 使用的设备。
        batch_size (int): Batch大小。这个大小会在训练集和测试集上同时生效。
        now_epoch (int): 当前在第几个epoch内。
    Returns:
        (TotalComparisonDict): 对于每一个类的总评估结果。
    """
    total_dict: TotalComparisonDict = {
        'change_prediction': None,
        't1_semantic_prediction': None,
        't2_semantic_prediction': None
    }  # 创建一个只有键的总字典。

    with tqdm(total=len(val_data), ncols=200) as bar:
        for i in range(len(val_data) // batch_size):
            # 内存监视器。
            current_mem = torch.cuda.memory_allocated(using_device)
            reserved_mem = torch.cuda.memory_reserved(using_device)
            total_mem = torch.cuda.get_device_properties(using_device).total_memory

            if batch_size == 1:
                x, y = val_data[i]
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
            else:
                x, y = val_data[i * batch_size: (i + 1) * batch_size: 1]
                x = torch.stack(x, dim=0)
                y = torch.stack(y, dim=0)

            result = model.forward(x)

            loss_value = model.calc_loss(result, y)
            compare_result = model.comparison(result, y)

            # 插入元素的逻辑。
            for key in compare_result:
                # 如果没有元素，则赋予元素。
                if total_dict[key] is None:
                    total_dict[key] = compare_result[key]

                else:
                # 如果有元素，则对该元素内的每一个tensor或float执行相加逻辑。
                    for inner_key in compare_result[key]:
                        total_dict[key][inner_key] += compare_result[key][inner_key]


            bar.update(1)
            bar.set_description(
                f"Epoch {model.epoches.item()} | Iteration {now_epoch} | Loss: {loss_value.item():.8} | "
                f"Cuda mem: {current_mem / 1024 ** 2:.2f}/{total_mem / 1024 ** 2:.2f} MB | "
            )
    for key in total_dict:
        for inner_key in total_dict[key]:
            # 除混淆矩阵不平均之外，其他属性全都平均了
            if inner_key != "confusion_matrix":
                total_dict[key][inner_key] /= (len(val_data) // batch_size)

    return compare_result


def main():
    train_data_file: str = r"E:/My Python files/毕业论文/其他数据/Second_TRAIN_ZIP/"  # 在哪里加载训练数据？
    val_data_file: str = r"E:/My Python files/毕业论文/其他数据/Second_TRAIN_ZIP/"  # 在哪里加载验证数据？

    preload_data: bool = False  # 是否要将数据预加载到内存中（以占用内存为代价换取更快的训练速度，且预加载需要一段时间）
    using_device: torch.device = torch.device("cuda:0")  # 使用的设备
    epoches: int = 10  # 训练几个epoch
    use_amp: bool = True  # 是否使用混合精度以加速？
    compile_model: bool = False  # 是否编译模型以加速？（需后端支持）
    batch: int = 1  # batch size
    save_interval: int = 2  # 每多少个epoch保存一次。
    model_save_path: str = ""  # 模型保存在哪个文件夹里。注意：这个东西的末尾必须是/，我没写os支持。

    load_model: bool = False  # 是否加载已保存的模型？
    model_load_path: str = ""  # 在哪里加载模型？

    # 解析数据。注意：data_location为原始图像位置，result_location为掩膜位置。要求格式均为png。
    train_data_old = DataLoader(
        data_location=train_data_file + "im1",
        result_location=train_data_file + "label1_wo_cm",
        preload_to_tensor=preload_data,
        device=using_device
    )
    train_data_new = DataLoader(
        data_location=train_data_file + "im2",
        result_location=train_data_file + "label2_wo_cm",
        preload_to_tensor=preload_data,
        device=using_device
    )
    dt_train_data = DualTempoDataLoader(train_data_old, train_data_new)

    val_data_old = DataLoader(
        data_location=val_data_file + "im1",
        result_location=val_data_file + "label1_wo_cm",
        preload_to_tensor=preload_data,
        device=using_device
    )
    val_data_new = DataLoader(
        data_location=val_data_file + "im2",
        result_location=val_data_file + "label2_wo_cm",
        preload_to_tensor=preload_data,
        device=using_device
    )
    dt_val_data = DualTempoDataLoader(val_data_old, val_data_new)

    # 建立模型。
    model = ChangeMask(semantic_classes=7, using_device=using_device, use_amp=use_amp)
    if load_model:
        state_for_model = torch.load(model_load_path)
        model.load_state_dict(state_for_model)
        print(f"Model loaded via {model_load_path}, continue training.")
    else:
        print("Training for a new model.")

    # 训练数据集。
    train_workflow(
        model,
        dt_train_data,
        dt_val_data,
        using_device=using_device,
        epoch=epoches,
        batch_size=batch,
        compile_model=compile_model,
        save_model_batches_interval=save_interval,
        model_save_path=model_save_path,
    )


if __name__ == '__main__':
    main()
