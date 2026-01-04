
from .utils import is_ascend_310p

if is_ascend_310p():
    from mindspore import mint
    mint.nan_to_num = lambda x, *args, **kwargs: x
