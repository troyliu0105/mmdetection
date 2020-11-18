import abc
import os

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


class BaseBackbone(nn.Module):
    def __init__(self,
                 out_indices=(),
                 frozen_stages=-1,
                 norm_eval=True):
        super(BaseBackbone, self).__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

    def init_weights(self, pretrained=True):
        if pretrained:
            from pytorchcv.models.model_store import download_model
            download_model(
                net=self,
                model_name=self.name,
                local_model_store_dir_path=os.path.join("~", ".torch", "models"),
                ignore_extra=True)
        else:
            self._init_params()

    @abc.abstractmethod
    def _init_params(self):
        pass

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
