import os
import random
import torch
import torchaudio
import soundfile as sf
import math


def get_dataset_filelist(
        input_training_file,
        input_validation_file,
        input_wavs_dir='',
        ext='.wav'):

    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(input_wavs_dir, x.split('|')[0] + ext)
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(input_wavs_dir, x.split('|')[0] + ext)
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

def load_wav(full_path):
    data, sampling_rate = sf.read(full_path)
    return data, sampling_rate

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, training_files, segment_size,
                sampling_rate, split=True, shuffle=True, 
                n_cache_reuse=1, resample=False,
                device=None,):
        
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.resample = resample
        self.resamplers = torch.nn.ModuleDict()
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            self.cached_wav = (audio, sampling_rate)

            self._cache_ref_count = self.n_cache_reuse
        else:
            audio, sampling_rate = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if sampling_rate != self.sampling_rate:
            if self.resample:
                key = str(sampling_rate)
                if key in self.resamplers.keys():
                    resampler = self.resamplers[key]
                else:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sampling_rate, new_freq=self.sampling_rate)
                    self.resamplers[key] = resampler
                audio = resampler(audio)
            else:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return (audio, filename)

    def __len__(self):
        return len(self.audio_files)




if __name__ == "__main__":

    training_files, validation_files = get_dataset_filelist(
        input_training_file='experiments/filelists/musan/train_list.txt',
        input_validation_file='experiments/filelists/musan/valid_list.txt',
        input_wavs_dir='../../DATA/musan/noise/free-sound',
        ext='', # filelist already includes extensions
    )

    dataset_train = AudioDataset(training_files=training_files,
                                 segment_size=16000,
                                 sampling_rate=22050,
                                 resample=True)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset_train, batch_size=4,
        num_workers=2, drop_last=True)

    for x, filename in dataloader:
        print(f"{filename}: shape is {x.shape}")