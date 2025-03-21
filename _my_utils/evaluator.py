"""
Attributes:
    Author (str): "GitHub: 月と猫 - LunaNeko"

"""
import inspect
from typing import Dict, List, Any, Callable, Optional, Literal


class EvaluatorParamRegisterDict(Dict):
    param_name: str
    param_init_value: Any
    updating_rule: Literal["Ascend", "Descend"]
    update_func: Optional[Callable]

class EvaluatorParamUpdateDict(Dict):
    param_name: str
    param_value: Any


class Evaluator:
    """
    一个评估器，但需要手动进行参数更新。
    且这里不提供任何评估逻辑，只提供参数管理服务。
    """
    def __init__(self):
        """
        初始化一个对模型的评估器。
        """
        # 新定义一个dict。
        self.tracing_params: Dict[str, Dict[str, Any]] = dict()

    def register_param(
            self,
            param_name: str,
            param_init_value: Any,
            updating_rule: Literal["Ascend", "Descend"] = "Ascend",
            update_func: Optional[Callable] = None,
    ) -> bool:
        """
        注册一个参数，以及这个东西的类型。
        使用方法update_param以更新参数。
        Args:
            param_name (str): 注册到这里的参数名。
            param_init_value (Any): 参数初始值。
            updating_rule (Literal["Ascend", "Descend"]): 参数更新规则。可选值：["Ascend", "Descend"]
            update_func (Optional[Callable[Any, Any]]): 自定义递增/递减规则判断函数。

        Notes:
            如果规定了参数update_func，则updating_rule选项失效。此时请确保函数的类型签名为：
            (old: Any, new: Any) -> bool
            其中，old为输入旧值，new为输入新值。返回值如果是True则表示进行更新，如果为False则表示不进行更新。
            如果没有规定对应的值，默认使用python内置的"<"或">"（对应特殊方法为__lt__或__gt__）。
            在updating_rule字段中，如果为"Ascend"，则新参数必须大于旧参数才可更新。如果为"Descend"，则新参数必须小于旧参数才可更新。
            这个字段如果为其他值则会立即报错。
            （该死的pycharm格式，后缀还必须有一个换行符才可以正常换行）

        Returns:
            如果参数列表中有了同名参数，则返回False，否则返回True。
        """
        if updating_rule not in ["Ascend", "Descend"]:
            raise ValueError(
                f"Unknown updating type {updating_rule}. You can only use: 'Ascend' or 'Descend'."
            )

        if param_name in self.tracing_params.keys():
            return False

        # 如果有更新判断函数，则判断它的参数是否合规。
        if update_func is not None:
            # 判断：这个东西必须先callable
            if not callable(update_func):
                raise ValueError(f"update_func should be callable, but found {type(update_func)}.")

            # 判断它的函数签名
            sig = inspect.signature(update_func)
            params: List[inspect.Parameter] = list(sig.parameters.values())
            if len(sig.parameters) != 2:
                raise ValueError(
                    f"update_func should have exactly two parameters, but found {len(sig.parameters)}."
                )
            if params[0].name != "old" or params[1].name != "new":
                raise ValueError(f"Invalid param name or order in update_func. "
                                 f"Need (old, new) , but found ({params[0].name, params[1].name})!")

            # 检查返回类型，必须为bool。
            return_annotation = sig.return_annotation
            if return_annotation != bool and return_annotation != inspect.Signature.empty:
                raise ValueError(
                    f"update_func should return a bool, but found {return_annotation}."
                )


        self.tracing_params.update({
            param_name: {
                "value": param_init_value,
                "updating_rule": updating_rule,
                "update_func": update_func,
            }
        })
        return True

    def register_multiple_params(
            self,
            param_register_list: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        注册多个参数。
        Notes:
            本函数将批量调度方法Evaluator.register_param以注册大量参数。

        Args:
            params (List[Dict[str, Any]]): 待注册参数的列表。其中列表的每一个元素都必须有如下参数：
            {
                "param_name": str,
                "param_init_value": Any,
                "updating_rule": Literal["Ascend", "Descend"],
                "update_func": Optional[Callable],
            }
        Returns:
            (List[bool]): 对每一个参数是否注册成功的返回值。
        """

        return [
            self.register_param(**param_register)
            for param_register in param_register_list
        ]

    def get_param_value(
            self,
            param_name: str
    ) -> Any:
        """
        输入参数名，返回其当前值。

        Notes:
        在查询参数时请确保参数被注册，否则直接报错。

        Args:
            param_name (str): 参数名。

        Returns:
            对应的参数值。
        """
        if param_name not in self.tracing_params.keys():
            raise ValueError(f"Param {param_name} not registed!")

        return self.tracing_params[param_name]["value"]

    def update_param(
            self,
            param_name: str,
            param_value: Any
    ) -> bool:
        """
        更新注册的参数。

        Notes:
            请确保在更新参数时，两个参数的数据类型相同。
            在更新参数时请确保参数被注册，否则直接报错。

        Args:
            param_name (str): 需要更新的参数名。
            param_value (Any): 更新的数值。

        Returns:
            如果按照规则更新成功则返回True，否则返回False。
        """
        if param_name not in self.tracing_params.keys():
            raise ValueError(f"Param {param_name} not registed!")

        origin_val: Any = self.tracing_params[param_name]["value"]
        if self.tracing_params[param_name]["update_func"] is not None:
            # 如果有更新函数规则，直接调度函数判断。
            if self.tracing_params[param_name]["update_func"](old=origin_val, new=param_value):
                self.tracing_params[param_name]["value"] = param_value
                return True
            else:
                return False
        else:
            # 否则直接用"<"和">"。
            if self.tracing_params[param_name]["updating_rule"] == "Ascend":
                if param_value > origin_val:
                    self.tracing_params[param_name]["value"] = param_value
                    return True
                else:
                    return False

            elif self.tracing_params[param_name]["updating_rule"] == "Descend":
                if param_value < origin_val:
                    self.tracing_params[param_name]["value"] = param_value
                    return True
                else:
                    return False
            else:
                # 如果传入了一个错误的更新规则，直接报错。
                raise ValueError(
                    f"Unknown updating type {self.tracing_params[param_name]['updating_rule']} "
                    f"for param {param_name}. It seems like you didn't read the annotation of method "
                    "Evaluator.register_param(). READ IT ONCE MORE!"
                )
    def update_multiple_params(
            self,
            param_update_list: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        批量更新参数，原理和另一个函数类似。不解释。

        Args:
            param_update_list (List[Dict[str, Any]]): 参数列表，每一个元素仅可拥有元素；
            {
                "param_name": str,
                "param_value": Any,
            }
        """
        return [
            self.update_param(**param_update)   # 直接解包字典
            for param_update in param_update_list
        ]