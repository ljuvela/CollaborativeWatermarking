import torch
import torch.nn.functional as F
import torchaudio
from audiodataset import AudioDataset, get_dataset_filelist

class RandomTimeStretch(torch.nn.Module):

    def __init__(self, min_scale_factor=0.9, max_scale_factor=1.1):

        super().__init__()

        if min_scale_factor > max_scale_factor:
            raise ValueError(f"Minimum scale factor must be smaller than maximum,"
                             f" got min: {min_scale_factor} {max_scale_factor}")

        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor



    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)

        Args:
            y : (batch, channels, time * rand_scale_factor)
        
        """

        scale_factor = (self.max_scale_factor - self.min_scale_factor) * torch.rand(1)[0] + self.min_scale_factor
        y = F.interpolate(input=x, size=int(scale_factor*x.size(-1)), mode='linear')
        return y


def test_rand_time_stretch():

    batch = 3
    timesteps = 16000
    channels = 1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    stretch = RandomTimeStretch(min_scale_factor=0.9, max_scale_factor=1.1)

    x = 0.1 * torch.randn(batch, channels, timesteps)
    x = torch.nn.Parameter(x)
    y = stretch(x.to(device))

    y.sum().backward()

    assert x.grad is not None


class NoiseAugment(torch.nn.Module):

    def __init__(self, dataset, batch_size, snr=10.0, num_workers=0):
        super().__init__()

        self.dataset = dataset
        self.snr = torch.Tensor([snr])

        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=num_workers, drop_last=True, 
            )

    def _noise_generator(self):

        while True:
            for x, _ in iter(self.dataloader):
                yield x


    def forward(self, waveform):
        """
        Args:
            waveform, shape = (batch, channels=1, time)

        Returns:
            noisy_waveform, shape (batch, channels=1, time) 
        """

        batch_size, channels, timesteps = waveform.size()

        noise = next(self._noise_generator())
        noisy_waveform = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, snr=self.snr * torch.ones(batch_size, channels))
        
        return noisy_waveform 

def test_noise_augment_grad():

    files, _ = get_dataset_filelist(
        input_training_file='experiments/filelists/musan/train_list.txt',
        input_validation_file='experiments/filelists/musan/valid_list.txt',
        input_wavs_dir='../../DATA/musan/noise/free-sound',
        ext='', # filelist already includes extensions
    )
    
    batch_size = 4
    segment_size = 16000
    dataset = AudioDataset(training_files=files,
                           segment_size=segment_size,
                           sampling_rate=22050,
                           resample=True)
    noise_augment = NoiseAugment(dataset=dataset, batch_size=batch_size, 
                 snr=20, num_workers=0)

    x = torch.ones(batch_size, 1, segment_size)
    x = torch.nn.Parameter(x)

    y = noise_augment(x)
    y.sum().backward()

    assert x.grad is not None


def test_noise_augment_iter():

    files, _ = get_dataset_filelist(
        input_training_file='experiments/filelists/musan/train_list.txt',
        input_validation_file='experiments/filelists/musan/valid_list.txt',
        input_wavs_dir='../../DATA/musan/noise/free-sound',
        ext='', # filelist already includes extensions
    )
    
    batch_size = 4
    segment_size = 16000
    dataset = AudioDataset(training_files=files,
                           segment_size=segment_size,
                           sampling_rate=22050,
                           resample=True)
    noise_augment = NoiseAugment(dataset=dataset, batch_size=batch_size, 
                 snr=20, num_workers=0)

    x = torch.ones(batch_size, 1, segment_size)
    x = torch.nn.Parameter(x)

    for i in range(100):
        with torch.no_grad():
            y = noise_augment(x)




if __name__ == "__main__":

    test_rand_time_stretch()
    test_noise_augment_grad()
    test_noise_augment_iter()

