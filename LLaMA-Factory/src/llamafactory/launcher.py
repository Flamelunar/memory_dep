# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llamafactory.train.tuner import run_exp  # use absolute import
# 在文件开头添加以下两行
from numpy.core.multiarray import _reconstruct
from torch.serialization import add_safe_globals; add_safe_globals([_reconstruct])
import torch

# 添加在文件最开头
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # 强制禁用安全加载
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
