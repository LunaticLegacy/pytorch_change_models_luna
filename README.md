## 介绍
 对于原内容进行的一份属于自己的实现，和一份完整的工作流。<br>
 原作：https://github.com/Z-Zheng/pytorch-change-models

## 更新内容
 这个内容没有使用ever库的工作流，而是自己基于原版功能，重写了工作流。（对于不太会用ever库的人更友好了——比如我）<br>
 此外，在这个文件里修改了主模型的一些函数的实现。<br>
 并且加了大量的类型注解，以及支持torch.compile()。<br>
 按照个人风格，我习惯在一个模型中定义一个backward方法，用于进行反向传播。

## 注意事项
 你可以更改TODO部分，以完成自己的模型保存逻辑。默认模型保存逻辑是定期保存。<br>

## Copyright
Copyright (c) Zhuo Zheng and affiliates. <br>
All rights reserved. <br>

This source code is licensed under the license found in the <br>
LICENSE file in the root directory of this source tree. <br>
