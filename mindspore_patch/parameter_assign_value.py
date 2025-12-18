# Monkey patch Parameter.assign_value to avoid re-initializing parameters after assigning values.
# This patch could be removed once the issue is resolved in MindSpore.

from mindspore import Parameter, Tensor


def hacked_param_assign_value(self, value):
    if value.has_init:
        self.init_flag = True
        self.init = value.init
    return Tensor.assign_value(self, value)

Parameter.assign_value = hacked_param_assign_value
