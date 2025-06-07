import json
import numpy as np
import spherical_harmonics as sh
from scipy.io.wavfile import write

# --- Load JSON file ---
with open("/workspaces/GSound-SIR/ray_generator/examples/path_data_obj_real.json", "r") as f:
    path_data = json.load(f)

# --- Extract listener directions ---
listener_directions = np.array(path_data['listener_directions'], dtype=np.float32)  # shape (N, 3)
listener_x = listener_directions[:, 0]
listener_y = listener_directions[:, 1]
listener_z = listener_directions[:, 2]

# --- Extract intensities ---
intensities = np.array(path_data['intensities'], dtype=np.float32)

# --- Extract other parameters ---
distances = np.array(path_data['distances'], dtype=np.float32)
speeds = np.array(path_data['speeds_of_sound'], dtype=np.float32)
path_types = np.array(path_data['path_types'], dtype=np.int32)
frequency_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32)

# --- Set parameters ---
order = 7
sample_rate = 48000

# --- Generate ambisonic IR ---
ir = sh.generate_ambisonic_ir(
    order=order,
    listener_directions=listener_directions,
    intensities=intensities,
    distances=distances,
    speeds=speeds,
    path_types=path_types,
    frequency_points=frequency_points,
    sample_rate=sample_rate,
    normalize=True
)

print(f"IR original shape: {ir.shape}")

# If ir.shape is (64, 145679), transpose it to (145679, 64)
ir = ir.T
print(f"IR transposed shape: {ir.shape}")

# Then convert and save as before
max_val = np.max(np.abs(ir))
if max_val > 0:
    ir_norm = ir / max_val
else:
    ir_norm = ir

ir_int16 = (ir_norm * 32767).astype(np.int16)

write("ambisonic_ir.wav", sample_rate, ir_int16)