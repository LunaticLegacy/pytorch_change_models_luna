"""
Attributes:
    Author (str): "GitHub: 月と猫 - LunaNeko"

Notes:
    This is not the class Dataloader provided via torch. It's my own implementation of another dataloader
    for the dataset for semantic segmentation.
"""

import torch
from PIL import Image
import torchvision.io as io
import numpy as np
from tqdm import tqdm

from typing import List, Tuple, Any, Dict, Union
import os


class DataLoader:
    # 数据加载器。每一个实例都会搜索在目录下名字相同的内容，并将其载入图像中。
    def __init__(self,
                 data_location: str,
                 result_location: str,
                 preload_to_tensor: bool = False,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            data_location (str): 数据位置，文件夹。
            result_location (str): 结果位置，文件夹。
            preload_to_tensor (bool): 是否将这些东西预加载为tensor，以便于快速使用。
            device (torch.device): 最终会将tensor在哪个设备上处理。
        """
        self.data_location = data_location
        self.result_location = result_location
        self.preloaded = preload_to_tensor
        self.device = device

        if preload_to_tensor:
            self.data_list: List[torch.Tensor] = []
            self.result_list: List[torch.Tensor] = []
            self.preload_data()
        else:
            self.data_index_list: List[str] = []
            self.result_index_list: List[str] = []
            self.preload_index()

    def count_data_number(
            self,
            extensions: str = ".png"
    ) -> int:
        """
        统计目录下有多少个数据。
        Args:
            extensions (str): 文件扩展名（后缀名）。注意：目录下所有文件的扩展名必须一致。

        Returns:
            (int): 目录下有多少数据。
        """
        return len([i for i in os.listdir(self.data_location) if i[-len(extensions):] == extensions])

    def preload_data(self) -> None:
        """
        预加载数据。
        遍历目录中的每一条数据，并将其转为tensor，随后存储到数据列表内。
        在预加载数据时不能将其上到显存里，而应当直接在内存中处理。
        Returns:
            不返回任何数据。
        """
        # 获取并排序文件名列表
        file_names = sorted([i for i in os.listdir(self.data_location) if i[-4:] == ".png"])

        for i in tqdm(file_names, desc="Preloading data..."):
            # 只加载png格式
            data_path = self.data_location + "/" + i
            result_path = self.result_location + "/" + i

            # if not self._is_file_complete(data_path) or not self._is_file_complete(result_path):
            #     print(f"Skipping incomplete file: {data_path} or {result_path}")
            #     continue

            try:

                data_pic: Image = io.read_image(data_path)
                result_pic: Image = io.read_image(result_path)

                data_array: np.ndarray = np.array(data_pic)
                result_array: np.ndarray = np.array(result_pic)

                data_tensor: torch.Tensor = (torch.tensor(data_array))
                result_tensor: torch.Tensor = (torch.tensor(result_array).unsqueeze(0))

                self.data_list.append(data_tensor)
                self.result_list.append(result_tensor)
            except Exception as e:
                print(f"Error loading file {data_path} or {result_path}: {e}")

        print(f"Loaded {len(self.data_list)} data.")

    def preload_index(self) -> None:
        """
        导入目录中有效数据的索引。在迭代器中，目标将稍候被转为tensor并处理。
        """
        # 获取并排序文件名列表
        file_names = sorted([i for i in os.listdir(self.data_location) if i[-4:] == ".png"])

        for i in tqdm(file_names, desc="Preloading index..."):
            data_path = self.data_location + "/" + i
            result_path = self.result_location + "/" + i

            # if not self._is_file_complete(data_path) or not self._is_file_complete(result_path):
            #     print(f"Skipping incomplete file: {data_path} or {result_path}")
            #     continue

            self.data_index_list.append(data_path)
            self.result_index_list.append(result_path)

        print(f"Loaded {len(self.data_index_list)} data's index.")

    def _is_file_complete(self, file_path: str) -> bool:
        """
        检查文件是否完整。
        Args:
            file_path (str): 文件路径。

        Returns:
            (bool): 文件是否完整。
        """
        try:
            with open(file_path, 'rb') as f:
                f.seek(0, 2)
                file_length = f.tell()
                f.seek(-2, 2)
                return f.read() == b'\xff\xd9' and file_length > 0
        except Exception as e:
            print(f"Error checking file {file_path}: {e}")
            return False

    def shuffle_dataset(self) -> None:
        from random import shuffle
        # 输出的结果是这样：
        # self.data_list[self.index - 1], self.result_list[self.index - 1]
        # 也就是说我需要将这两个东西合起来再shuffle一次。
        if self.preloaded:
            tempo_shuffle_bucket: list = list(zip(self.data_list, self.result_list))
            shuffle(tempo_shuffle_bucket)
            self.data_list, self.result_list = zip(*tempo_shuffle_bucket)
        else:
            tempo_shuffle_bucket: list = list(zip(self.data_index_list, self.result_index_list))
            shuffle(tempo_shuffle_bucket)
            self.data_index_list, self.result_index_list = zip(*tempo_shuffle_bucket)

    # 写一个迭代器，在每一步迭代中都需要输出1组：数据分布和实际类型分布。
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        下一个。
        Returns:
            (Tuple[torch.Tensor, torch.Tensor]) 一个由2张图组成的内容，其中一个是输入结果，另一个是需验证的结果。
            所有输出的结果的尺寸都是：[bands, width, height]
        """
        # 是否已预加载数据？
        if self.preloaded:
            # 使用预加载数据
            if self.index < len(self.data_list):
                self.index += 1
            else:
                raise StopIteration
            # 将其转入到对应的设备内，并转变数据格式。这里哪怕产生开销也无所谓了——我64G的内存应该够吃。
            return (self.data_list[self.index - 1].to(device=self.device, dtype=torch.float32),
                    self.result_list[self.index - 1].to(device=self.device, dtype=torch.int64))

        else:
            # 没有预加载数据，使用索引
            if self.index < len(self.data_index_list):
                self.index += 1
            else:
                raise StopIteration
            # 预加载索引，开始载入数据。
            data_pic: Image = io.read_image(self.data_index_list[self.index - 1])
            result_pic: Image = io.read_image(self.result_index_list[self.index - 1])

            data_array: np.ndarray = np.array(data_pic)
            result_array: np.ndarray = np.array(result_pic)

            # 在这里需要将数据的设备和格式进行转换，因为是直接基于数据索引从硬盘上扒下来的数据。
            data_tensor: torch.Tensor = (torch.tensor(data_array)
                                         .to(device=self.device, dtype=torch.float32))
            result_tensor: torch.Tensor = (torch.tensor(result_array)
                                           .to(device=self.device, dtype=torch.int64))

            return data_tensor, result_tensor

    def __getitem__(self, item: Union[int, slice]) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        获得值。
        Args:
            item (Union[int, slice]): 索引方式。
        Returns:
            所有输出的Tensor的尺寸都是：[bands, width, height]
        """
        if self.preloaded:
            # 处理预加载的数据
            if isinstance(item, int):
                return (self.data_list[item].to(device=self.device, dtype=torch.float32),
                        self.result_list[item].to(device=self.device, dtype=torch.int64))
            elif isinstance(item, slice):
                start, stop, step = item.start, item.stop, item.step
                return (
                [self.data_list[i].to(device=self.device, dtype=torch.float32) for i in range(start, stop, step)],
                [self.result_list[i].to(device=self.device, dtype=torch.int64) for i in range(start, stop, step)])
            else:
                raise TypeError(f"Unsupported index type: {type(item)}")
        else:
            # 处理非预加载的数据
            if isinstance(item, int):
                data_pic: Image = io.read_image(self.data_index_list[item])
                result_pic: Image = io.read_image(self.result_index_list[item])

                data_array: np.ndarray = np.array(data_pic)
                result_array: np.ndarray = np.array(result_pic)

                data_tensor: torch.Tensor = (torch.tensor(data_array)
                                             .to(device=self.device, dtype=torch.float32))
                result_tensor: torch.Tensor = (torch.tensor(result_array)
                                               .to(device=self.device, dtype=torch.int64))

                return data_tensor, result_tensor
            elif isinstance(item, slice):
                start, stop, step = item.start, item.stop, item.step
                data_tensors = []
                result_tensors = []

                for i in range(start, stop, step):
                    data_pic: Image = io.read_image(self.data_index_list[i])
                    result_pic: Image = io.read_image(self.result_index_list[i])

                    data_array: np.ndarray = np.array(data_pic)
                    result_array: np.ndarray = np.array(result_pic)

                    data_tensor: torch.Tensor = (torch.tensor(data_array)
                                                 .to(device=self.device, dtype=torch.float32))
                    result_tensor: torch.Tensor = (torch.tensor(result_array)
                                                   .to(device=self.device, dtype=torch.int64))

                    data_tensors.append(data_tensor)
                    result_tensors.append(result_tensor)

                return data_tensors, result_tensors
            else:
                raise TypeError(f"Unsupported index type: {type(item)}")

    def __len__(self):
        if self.preloaded:
            return len(self.data_list)
        else:
            return len(self.data_index_list)


class DualTempoDataLoader:
    """
    输入两个DataLoader，返回一个双时相融合的图像。
    目前这个类尚未成熟——暂时还无法返回对齐规则和刷新规则。
    （它现在仅仅是能用，还不能算是“用得很好”）
    """
    def __init__(
            self,
            dataloader1: DataLoader,
            dataloader2: DataLoader
    ):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2

    def __getitem__(self, item: int | slice) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:

        if isinstance(item, int):
            # 如果是int类型，则在1和2类中各自选择其中一类。
            item1: Tuple[torch.Tensor, torch.Tensor] = self.dataloader1[item]
            item2: Tuple[torch.Tensor, torch.Tensor] = self.dataloader2[item]

            value: torch.Tensor = torch.cat([item1[0], item2[0]], dim=0)
            mask: torch.Tensor = torch.cat([item1[1], item2[1]], dim=0)

            return value, mask

        elif isinstance(item, slice):
            # 获取两组数据，然后分层进行解析。
            item1: Tuple[List[torch.Tensor], List[torch.Tensor]] = self.dataloader1[item]
            item2: Tuple[List[torch.Tensor], List[torch.Tensor]] = self.dataloader2[item]

            data_tensors = [torch.cat([data1, data2], dim=0) for data1, data2 in zip(item1[0], item2[0])]
            result_tensors = [torch.cat([result1, result2], dim=0) for result1, result2 in zip(item1[1], item2[1])]

            return data_tensors, result_tensors

        else:
            raise TypeError(f"Unsupported index type: {type(item)}")

    def __len__(self):
        return min(len(self.dataloader1), len(self.dataloader2))