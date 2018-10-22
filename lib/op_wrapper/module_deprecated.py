import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class FlexModule(Module):
    """sub-class for flexible module implementation
    """
    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.
        The parameter can be accessed as an attribute using given name.
        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            parameter (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        self._parameters[name] = param

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None and not (isinstance(value, torch.Tensor) and value.is_leaf is False):
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter, non-leaf torch.Tensor or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

