class AudioConfig:
    SAMPLE_RATE = 16_000
    MAX_TIME = 1
    N_FFT = 1024
    HOP_LEN = 512
    N_MELS = 64

class TrainConfig:
    TRAIN_MU = -13.7342
    TRAIN_STD = 16.1290

    TRAIN_PCT = .8



class Config:
    train_config = TrainConfig
    audio_config = AudioConfig
