
def preprocess_signal(
        signal, 
        sample_rate=44100, 
        offset=0, 
        length=None,
        mono=False, 
        **kwargs
    ):
    if mono and len(signal.shape) > 1:
        signal = signal[:, 0]
    if length is None:
        length = len(signal)
    return signal[offset:offset + length]
