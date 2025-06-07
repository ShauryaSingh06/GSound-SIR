import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

# Load clean mono audio
clean_audio, sr_audio = sf.read("/workspaces/GSound-SIR/1272-128104-0002.flac")  # Ensure it's mono

# Load IR (multichannel)
ir_data, sr_ir = sf.read("/workspaces/GSound-SIR/auralizer/ambisonic_ir.wav")  # shape: (samples, channels)

# Make sure sample rates match
assert sr_audio == sr_ir == 16000, "Sample rates must match."

# Convolve clean audio with each channel of IR
convolved_channels = []
for ch in range(ir_data.shape[1]):
    convolved = fftconvolve(clean_audio, ir_data[:, ch], mode='full')
    convolved_channels.append(convolved)

# Stack all channels
spatial_audio = np.stack(convolved_channels, axis=-1)

# Normalize
max_val = np.max(np.abs(spatial_audio))
if max_val > 1.0:
    spatial_audio = spatial_audio / max_val

# Write output
sf.write("spatial_audio_output.wav", spatial_audio, sr_audio)
