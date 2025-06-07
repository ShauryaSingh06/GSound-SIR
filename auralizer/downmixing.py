import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import librosa

# Load clean mono audio
clean_audio, sr_audio = sf.read("/workspaces/GSound-SIR/1272-128104-0002.flac")  # 16kHz
assert clean_audio.ndim == 1, "Input audio must be mono."

# Load IR (64ch @ 48kHz)
ir_data, sr_ir = sf.read("/workspaces/GSound-SIR/auralizer/ambisonic_ir.wav")
assert ir_data.ndim == 2 and ir_data.shape[1] == 64, "Expected 64-channel IR."

# Resample IR to 16kHz
if sr_ir != sr_audio:
    print(f"Resampling IR from {sr_ir} Hz to {sr_audio} Hz...")
    ir_data = np.stack([
        librosa.resample(ir_data[:, ch], orig_sr=sr_ir, target_sr=sr_audio)
        for ch in range(64)
    ], axis=1)

# Downmix to 32 channels
ir_downmixed = ir_data[:, :32]

# Convolve each IR channel with mono audio
spatial_output = np.stack([
    fftconvolve(clean_audio, ir_downmixed[:, ch], mode='full')
    for ch in range(32)
], axis=-1)

# Normalize
spatial_output /= np.max(np.abs(spatial_output))

# Save output
sf.write("spatial_output_32ch.wav", spatial_output, sr_audio)

print("✅ Saved spatial_output_32ch.wav (32ch @ 16kHz)")
