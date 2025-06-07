import soundfile as sf

# Replace this with the actual path to your FLAC file
filename = "/workspaces/GSound-SIR/1272-128104-0002.flac"

data, samplerate = sf.read(filename)

print(f"Sample Rate: {samplerate}")
print(f"Shape: {data.shape}")

if len(data.shape) == 1:
    print("Mono audio")
elif len(data.shape) == 2:
    print(f"Stereo or multichannel audio with {data.shape[1]} channels")
else:
    print("Unknown format")