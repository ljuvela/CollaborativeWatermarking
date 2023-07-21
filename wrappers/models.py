import os
import sys
import torch

from torchaudio.transforms import Resample

import importlib.util
from types import SimpleNamespace

def import_module_from_file(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lfcc = import_module_from_file(
    name='lfcc',
    filepath=os.path.realpath(f"{__file__}/../../third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py"))


class LFCC_LCNN(lfcc.Model):

    def __init__(self, in_dim, out_dim, sample_rate):
        """
        Args: 
            in_dim: input dimension, default 1 for single channel wav
            out_dim: output dim, default 1 for single value classifier
        """

        prj_conf = SimpleNamespace()
        prj_conf.optional_argument = [""]
        args = None
        mean_std = None
        super().__init__(
            in_dim, out_dim,
            args=args, prj_conf=prj_conf,
            mean_std=mean_std)
        
        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            self.resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        else:
            self.resampler = None

    def forward(self, x):
        """
        Args:
            x: (batch, channels=1, length)

        Returns:
            scores: (batch, length=1)

        """

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        if self.resampler is not None:
            x = self.resampler(x)

        feature_vec = self._compute_embedding(x[:, 0, :], datalength=None)
        # return feature_vec
        scores = self._compute_score(feature_vec)
        scores = scores.reshape(-1, 1)
        return scores
    

if __name__ == "__main__":

    model = LFCC_LCNN(in_dim=1, out_dim=1)
    model = model.eval()

    batch = 2
    timesteps = 16000
    channels = 1
    x = 0.1 * torch.randn(batch, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)

    scores = model.forward(x)

    # check that gradients pass
    scores.sum().backward()

    print(f"{x.grad}")

