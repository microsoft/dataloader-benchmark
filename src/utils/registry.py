"""Provides base registry class.
"""

import collections
from typing import Any, Callable, DefaultDict, Dict, Optional, Type

from pytorch_lightning import LightningDataModule, LightningModule
from torch.nn import Module


class Singleton(type):
    """Meta class for Types
    Args:
        type (_type_): _description_
    Returns:
        _type_: _description_
    """

    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Registry(metaclass=Singleton):
    """Base registry
    Args:
        metaclass (Singleton, optional): _description_. Defaults to Singleton.
    Returns:
        _type_: _description_
    """

    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(to_register, assert_type), f"{to_register} must be a subclass of {assert_type}"
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_torchmodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a pytorch module to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl("torchmodule", to_register, name, assert_type=LightningDataModule)

    @classmethod
    def register_datamodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a pytorch lightning datamodule to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl("datamodule", to_register, name, assert_type=LightningDataModule)

    @classmethod
    def register_lightningmodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a pytorch lightning module to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl("lightningmodule", to_register, name, assert_type=LightningModule)

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_torchmodule(cls, name: str) -> Type[Module]:
        return cls._get_impl("inputtransform", name)

    @classmethod
    def get_datamodule(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("datamodule", name)

    @classmethod
    def get_lightningmodule(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("lightningmodule", name)


registry = Registry()
