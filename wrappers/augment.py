import torch
import torch.nn.functional as F
import torchaudio
from audiodataset import AudioDataset, get_dataset_filelist

def get_augmentations(h, a, device):

    aug_modules_train = []
    aug_modules_valid = []
    aug_types = h.get('augmentation_types', [])
    for aug_type in aug_types:
        if aug_type == 'stretch':
            aug_modules_train.append(RandomTimeStretch(min_scale_factor=0.9, max_scale_factor=1.1))
            aug_modules_valid.append(RandomTimeStretch(min_scale_factor=0.9, max_scale_factor=1.1))
        elif aug_type == 'noise':
            noise_filelist_train, noise_filelist_valid = get_dataset_filelist(
                input_training_file=a.noise_input_training_file,
                input_validation_file=a.noise_input_validation_file,
                input_wavs_dir=a.noise_input_wavs_dir,
                ext=''
                )
            noise_dataset_train = AudioDataset(
                split=True,
                training_files=noise_filelist_train,
                sampling_rate=h.sampling_rate, 
                segment_size=h.segment_size,
                resample=True, device=device)
            aug_modules_train.append(NoiseAugment(
                noise_dataset_train, batch_size=h.batch_size,
                snr=10, num_workers=0))
            
            noise_dataset_valid = AudioDataset(
                split=False,
                training_files=noise_filelist_valid,
                sampling_rate=h.sampling_rate, 
                resample=True, device=device)
            aug_modules_valid.append(NoiseAugment(
                noise_dataset_valid, batch_size=1,
                snr=10, num_workers=0))
            
        elif aug_type == 'reverb':
            reverb_filelist_train, reverb_filelist_valid = get_dataset_filelist(
                input_training_file=a.reverb_input_training_file,
                input_validation_file=a.reverb_input_validation_file,
                input_wavs_dir=a.reverb_input_wavs_dir,
                ext=''
                )
            reverb_dataset_train = AudioDataset(
                training_files=reverb_filelist_train, 
                sampling_rate=h.sampling_rate,
                split=False,
                resample=True,
                device=device)
            aug_modules_train.append(ReverbAugment(
                reverb_dataset_train,
                num_workers=0))

            reverb_dataset_valid = AudioDataset(
                training_files=reverb_filelist_train, 
                sampling_rate=h.sampling_rate,
                split=False,
                resample=True,
                device=device)
            aug_modules_valid.append(ReverbAugment(
                reverb_dataset_valid,
                num_workers=0))

        else:
            raise ValueError(f"Unsupported augmentation type '{aug_type}'")

    if len(aug_modules_train) > 0:
        augmentation_train =  torch.nn.Sequential(*aug_modules_train).to(device)
    else:
        augmentation_train = None

    if len(aug_modules_valid) > 0:
        augmentation_valid =  torch.nn.Sequential(*aug_modules_valid).to(device)
    else:
        augmentation_valid = None

    return augmentation_train, augmentation_valid

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
        # negative pad values truncate the signal
        y = torch.nn.functional.pad(y, pad=(x.size(-1)-y.size(-1), 0), mode='constant', value=0.0)
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
        self.register_buffer('snr', torch.Tensor([snr]))

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
        noise = noise.to(waveform.device)
        noise = torch.nn.functional.pad(
            noise, pad=(waveform.size(-1)-noise.size(-1), 0),
            mode='constant', value=0.0)

        noisy_waveform = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, 
            snr=self.snr*torch.ones(batch_size, channels, device=waveform.device))
        
        return noisy_waveform
    


class ReverbAugment(torch.nn.Module):

    def __init__(self, dataset, num_workers=0):
        super().__init__()

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            num_workers=num_workers, drop_last=True, 
            )

    def _rir_generator(self):
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

        device = waveform.device
        rir = next(self._rir_generator())

        # align to maximum peak
        offset = 20
        start = torch.maximum(torch.argmax(rir.abs()) - offset, torch.zeros(1))
        stop = torch.minimum(torch.Tensor([rir.size(-1)]), start+timesteps)
        rir = rir[..., int(start):int(stop)]

        rir = rir.to(device)
        # pad
        rir = torch.nn.functional.pad(rir, pad=(waveform.size(-1)-rir.size(-1), 0), mode='constant', value=0.0)
        # normalize
        rir = rir / (rir.norm(2) + 1e-6)
    
        y = torchaudio.functional.fftconvolve(waveform, rir)
        return y[..., :timesteps]

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


def test_reverb_augment_iter():


    files, _ = get_dataset_filelist(
        input_training_file='experiments/filelists/mit-rir/train_list.txt',
        input_validation_file='experiments/filelists/mit-rir/valid_list.txt',
        input_wavs_dir='../../DATA/MIT-RIR/Audio',
        ext='', # filelist already includes extensions
    )
    
    batch_size = 4
    sample_rate = 22050
    segment_size = sample_rate * 2
    dataset = AudioDataset(training_files=files,
                           segment_size=segment_size,
                           sampling_rate=sample_rate,
                           resample=True)
    reverb_augment = ReverbAugment(dataset=dataset, num_workers=0)

    x = torch.zeros(batch_size, 1, segment_size)
    x[..., 1000] = 1.0
    x = torch.nn.Parameter(x)

    for i in range(1):
        with torch.no_grad():
            y = reverb_augment(x)
    y_np = y[0].detach().squeeze().numpy()
    import soundfile as sf
    sf.write('reverb_test.wav', y_np, sample_rate)


if __name__ == "__main__":

    test_reverb_augment_iter()
    test_rand_time_stretch()
    test_noise_augment_grad()
    test_noise_augment_iter()

