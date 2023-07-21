import torch

from models import MultiPeriodDiscriminator, MultiScaleDiscriminator

from .models import LFCC_LCNN

class WatermarkModelEnsemble(torch.nn.Module):

    def __init__(self, model_type:str, sample_rate:int):
        super().__init__()

        self.models = torch.nn.ModuleDict()

        if model_type == "hifi_gan":
            self.models['mpd'] = MultiPeriodDiscriminator()
            self.models['msd'] = MultiScaleDiscriminator()
        elif model_type == "lfcc_lcnn":
            self.models['lfcc_lcnn'] = LFCC_LCNN(
                in_dim=1, out_dim=1, 
                sample_rate=sample_rate)
        elif model_type is None:
            pass
        else:
            raise ValueError(f"Unsupported watermark model type {model_type}")


    def forward(self, x_real, x_fake):

        outputs_real = []
        outputs_fake = []
        for name, model in self.models.items():
            
            if name == 'mpd':
                y_real, y_fake, _, _ = model(x_real, x_fake)
            elif name == 'msd':
                y_real, y_fake, _, _ = model(x_real, x_fake)
            else:
                y_real = [model(x_real)]
                y_fake = [model(x_fake)]


            outputs_real.append(y_real)
            outputs_fake.append(y_fake)

        return outputs_real, outputs_fake
    
    def get_labels(self):
        return list(self.models.keys())
    
    def get_num_models(self):
        return len(self.models.keys())

