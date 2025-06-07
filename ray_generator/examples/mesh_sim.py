import numpy as np
import pygsound as ps
import json

def dict_to_json_real_data(d):
    # convert all numpy arrays to lists (real data, not shape)
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()   # <-- convert to list here
        elif isinstance(v, dict):
            d[k] = dict_to_json_real_data(v)
    return d


def main():
    # Simulation using .obj file (and an optional .mtl file)
    ctx = ps.Context()
    ctx.diffuse_count = 20000
    ctx.specular_count = 2000
    ctx.channel_type = ps.ChannelLayoutType.stereo
    
    mesh1 = ps.loadobj("cube.obj")
    scene = ps.Scene()
    scene.setMesh(mesh1)

    src_coord = [10, 3, 0.5]
    lis_coord = [5, 3, 0.5]

    src = ps.Source(src_coord)
    src.radius = 0.01
    src.power = 100.0

    lis = ps.Listener(lis_coord)
    lis.radius = 0.01

    # res = scene.computeIR([src], [lis], ctx)    # you may pass lists of sources and listeners to get N_src x N_lis IRs
    # audio_data = np.array(res['samples'][0][0])     # the IRs are indexed by [i_src, i_lis, i_channel]
    # with WaveWriter('test1.wav', channels=audio_data.shape[0], samplerate=int(res['rate'])) as w1:
    #     w1.write(audio_data)
    #     print("IR using .obj input written to test1.wav.")
    
    res = scene.getPathData([src], [lis], ctx)["path_data"]
    # get the first listener's path data
    path_data = res[0]
    path_data_real = dict_to_json_real_data(path_data)

    with open("path_data_obj_real.json", "w") as f:
        json.dump(path_data_real, f, indent=4)

    # Simulation using a shoebox definition
    mesh2 = ps.createbox(100, 100, 3, 0.5, 0.1)
    scene = ps.Scene()
    scene.setMesh(mesh2)

    # res = scene.computeIR([src_coord], [lis_coord], ctx)    # use default source and listener settings if you only pass coordinates
    # audio_data = np.array(res['samples'][0][0])
    # with WaveWriter('test2.wav', channels=audio_data.shape[0], samplerate=int(res['rate'])) as w2:
    #     w2.write(audio_data)
    #     print("IR using shoebox input written to test2.wav.")
    
    res = scene.getPathData([src_coord], [lis_coord], ctx)["path_data"]
    path_data = res[0]
    path_data_real = dict_to_json_real_data(path_data)

    with open("path_data_shoebox_real.json", "w") as f:
        json.dump(path_data_real, f, indent=4)

if __name__ == '__main__':
    main()
