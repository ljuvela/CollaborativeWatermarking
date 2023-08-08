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

rawnet = import_module_from_file(
    name='rawnet',
    filepath=os.path.realpath(f"{__file__}/../../third_party/asvspoof-2021/LA/Baseline-RawNet2/model.py")
)

class LFCC_LCNN(lfcc.Model):

    def __init__(self, in_dim, out_dim, 
                 sample_rate,
                 sigmoid_output=True,
                 dropout_prob=0.7,
                 use_batch_norm=True):
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
            mean_std=mean_std,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm)
        
        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            self.resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        else:
            self.resampler = None

        self.sigmoid_out = sigmoid_output


    def eval(self, pass_gradients=True):
        """
        Set model to eval mode

        cuDNN RNNs are not passing gradients in eval mode 
        and these need to be exempted if gradient flow is needed
        """
        if not pass_gradients:
            self.eval()
        else:
            for name, module in self.named_children():
                print(name)
                if name != 'gru':
                    module.eval()
            self.training = False
            

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
        scores = self._compute_score(feature_vec, inference=(not self.sigmoid_out))
        scores = scores.reshape(-1, 1)
        return scores
    

class RawNet(rawnet.RawNet):

    def __init__(
            self,
            sample_rate=16000,
            first_conv=1024,
            in_channels=1,
            filts=[20, [20, 20], [20, 128], [128, 128]],
            nb_fc_node= 1024,
            gru_node=1024,
            nb_gru_layer=3,
            nb_classes=2,
            device=torch.device('cpu'),
            use_batch_norm=True,
            pad_input_to_len: int = None
            ):
        """
        Args:
            first_conv: no. of filter coefficients 
            in_channels: ?
            filts: no. of filters channel in residual blocks
            nb_fc_node: ?
            gru_node: ?
            nb_gru_layer: ?
            nb_classes: ?
            pad_input_to_len: pad input to specific length (default: None, uses input as is)
        
        """
        d_args = {
            'first_conv': first_conv,
            'in_channels': in_channels,
            'filts': filts,
            'nb_fc_node': nb_fc_node,
            'gru_node': gru_node,
            'nb_gru_layer': nb_gru_layer,
            'nb_classes': nb_classes
        }


        super().__init__(d_args=d_args, device=device, 
                         use_batch_norm=use_batch_norm)

        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            self.resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        else:
            self.resampler = None

        self.pad_input_to_len = pad_input_to_len


    def eval(self, pass_gradients=True):
        """
        Set model to eval mode

        cuDNN RNNs are not passing gradients in eval mode 
        and these need to be exempted if gradient flow is needed
        """
        if not pass_gradients:
            self.eval()
        else:
            for name, module in self.named_children():
                print(name)
                if name != 'gru':
                    module.eval()
            self.training = False
            


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

        if self.pad_input_to_len is not None:
            # left padding
            x = torch.nn.functional.pad(x, pad=(self.pad_input_to_len - x.size(-1), 0), mode='constant', value=0.0)

        log_out = super().forward(x[:, 0, :])

        # slice from (batch, num_classes) -> (batch, 1)
        log_out = log_out[:, 0:1]

        return torch.exp(log_out)


def test_lfcc_lcnn():
        
    model = LFCC_LCNN(in_dim=1, out_dim=1, sample_rate=16000)

    batch = 2
    timesteps = 16000
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)

    scores = model.forward(x)
    # check that gradients pass
    scores.sum().backward()

    assert x.grad is not None


def test_rawnet():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch = 3
    timesteps = 22050
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)
    x_dev = x.to(device)

    model = RawNet(sample_rate=16000)
    model = model.to(device)

    scores = model.forward(x_dev)
    scores.pow(2).sum().backward()

    assert x.grad is not None


def test_rawnet_no_batch_norm():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch = 3
    timesteps = 22050
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)
    x_dev = x.to(device)

    model = RawNet(sample_rate=16000, use_batch_norm=False)
    model = model.to(device)

    assert not any(['running_mean' in k for k in  model.state_dict().keys()])

    scores = model.forward(x_dev)
    scores.pow(2).sum().backward()

    # for name, param in model.named_parameters():
    #     print(name)

    assert x.grad is not None


def test_lfcc_lcnn_no_dropout_no_batchnorm():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = LFCC_LCNN(in_dim=1, out_dim=1, sample_rate=22050,
                      dropout_prob=0.0, use_batch_norm=False)
    model = model.to(device)

    assert not any(['running_mean' in k for k in  model.state_dict().keys()])

    batch = 3
    timesteps = 16000
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)

    scores = model.forward(x.to(device))
    # check that gradients pass
    scores.sum().backward()

    assert x.grad is not None


def test_rawnet_padding():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch = 3
    sample_rate = 22050
    timesteps = 22050
    channels = 1
    x = 0.1 * torch.randn(batch, 1, timesteps, requires_grad=True)
    x = torch.nn.Parameter(x)
    x_dev = x.to(device)

    model = RawNet(sample_rate=sample_rate, use_batch_norm=False, pad_input_to_len=64600)
    model = model.to(device)

    scores = model.forward(x_dev)
    scores.pow(2).sum().backward()

    assert x.grad is not None



if __name__ == "__main__":

    test_rawnet_padding()
    test_rawnet_no_batch_norm()
    test_lfcc_lcnn()
    test_rawnet()
    test_lfcc_lcnn_no_dropout_no_batchnorm()





