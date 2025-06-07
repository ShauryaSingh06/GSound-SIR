import json
import numpy as np
from scipy.signal import convolve
from scipy.special import sph_harm
import soundfile as sf
import argparse

def ambisonic_coefficients(azimuth, elevation, max_order):
    """
    Compute Ambisonic coefficients up to max_order for given azimuth and elevation.
    """
    phi = np.pi / 2 - elevation  # Polar angle
    theta = azimuth  # Azimuthal angle
    coeffs = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            Y = sph_harm(abs(m), n, theta, phi)
            if m < 0:
                coeff = np.sqrt(2) * (-1)**m * Y.imag
            elif m > 0:
                coeff = np.sqrt(2) * Y.real
            else:
                coeff = Y.real
            coeffs.append(coeff)
    return np.array(coeffs)

def load_ray_data(json_path):
    """
    Load ray path data from JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    listener_directions = np.array(data['listener_directions'], dtype=np.float32)
    intensities = np.array(data['intensities'], dtype=np.float32)
    distances = np.array(data['distances'], dtype=np.float32)
    speeds = np.array(data['speeds_of_sound'], dtype=np.float32)
    path_types = np.array(data['path_types'], dtype=np.int32)
    return listener_directions, intensities, distances, speeds, path_types

def generate_ambisonic_ir(listener_directions, intensities, distances, speeds, path_types, sample_rate, ambisonic_order):
    """
    Generate Ambisonic impulse response from ray data.
    """
    num_channels = (ambisonic_order + 1) ** 2
    delays = distances / speeds
    max_delay = np.max(delays)
    ir_length = int(max_delay * sample_rate) + 1
    ir = np.zeros((ir_length, num_channels), dtype=np.float32)
    
    for i in range(len(delays)):
        delay_samples = int(delays[i] * sample_rate)
        if delay_samples >= ir_length:
            continue
        direction = listener_directions[i] / np.linalg.norm(listener_directions[i])
        azimuth = np.arctan2(direction[1], direction[0])
        elevation = np.arcsin(direction[2])
        coeffs = ambisonic_coefficients(azimuth, elevation, ambisonic_order)
        amplitude = intensities[i] if intensities.ndim == 1 else np.sum(intensities[i])
        # Adjust amplitude based on path type (example)
        if path_types[i] == 0:  # Direct
            amplitude *= 1.0
        elif path_types[i] == 1:  # Reflected
            amplitude *= 0.8
        elif path_types[i] == 2:  # Diffracted
            amplitude *= 0.5
        ir[delay_samples, :] += amplitude * coeffs
    return ir

def spatialize_audio(clean_audio, ir):
    """
    Convolve clean audio with Ambisonic IR to produce spatialized audio.
    """
    num_channels = ir.shape[1]
    spatialized_audio = np.zeros((len(clean_audio) + len(ir) - 1, num_channels), dtype=np.float32)
    for ch in range(num_channels):
        spatialized_audio[:, ch] = convolve(clean_audio, ir[:, ch], mode='full')
    return spatialized_audio

def generate_spatial_audio(json_path, audio_path, ambisonic_order=3):
    """
    Generate spatialized audio from ray data and clean audio.
    
    Args:
        json_path (str): Path to the JSON file with ray data.
        audio_path (str): Path to the clean audio file in FLAC format.
        ambisonic_order (int): Ambisonic order (default is 3 for third-order).
    
    Returns:
        tuple: (spatialized_audio, sample_rate) where spatialized_audio is a NumPy array
               of shape (num_samples, num_channels) and sample_rate is an integer.
    """
    # Load ray data
    listener_directions, intensities, distances, speeds, path_types = load_ray_data(json_path)
    
    # Load clean audio (FLAC format)
    clean_audio, sample_rate = sf.read(audio_path)
    if clean_audio.ndim > 1:
        clean_audio = clean_audio[:, 0]  # Take first channel if stereo
    
    # Generate Ambisonic impulse response
    ir = generate_ambisonic_ir(listener_directions, intensities, distances, speeds, path_types, sample_rate, ambisonic_order)
    
    # Spatialize the audio
    spatialized_audio = spatialize_audio(clean_audio, ir)
    
    return spatialized_audio, sample_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatialize audio using ray data")
    parser.add_argument("--json", required=True, help="Path to ray data JSON file")
    parser.add_argument("--input", required=True, help="Path to clean audio file (FLAC)")
    parser.add_argument("--output", required=True, help="Path to output spatialized audio file")
    parser.add_argument("--order", type=int, default=3, help="Ambisonic order")
    args = parser.parse_args()
    
    spatialized_audio, sample_rate = generate_spatial_audio(args.json, args.input, args.order)
    sf.write(args.output, spatialized_audio, sample_rate)
    print(f"Spatialized audio saved to {args.output}")