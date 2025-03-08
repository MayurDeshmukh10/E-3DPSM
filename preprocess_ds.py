import h5py
import os
import numpy as np
import cv2
# import numba
import argparse
import functools
from EventEgoPoseEstimation.dataset.representation import LNES, RawEvent
from configs.settings import config as cfg
from EventEgoPoseEstimation.model import EventTensor as EgoHPE
import torch
from EventEgoPoseEstimation.utils.vis import visualize_temporal_bins
from EventEgoPoseEstimation.dataset.dataset_utils import event_augmentation, save_augmented_data
import json


TIME_WINDOW = 20 # in milliseconds
NUM_EVENTS = 8000 # roughly equivalent to 1 msec
DS_HEIGHT = 480
DS_WIDTH = 640
CHECKPOINT_PATH = "/CT/EventEgo3Dv2/work/EventEgo3Dv2/results/wandb_data/pretraining_new_dataloader_oom_fixed_5_2_2/EventEgo3Dv2/lmm6tavt/checkpoints/last.ckpt"
h5py_File = functools.partial(h5py.File, libver='latest', swmr=True)


# file = "/CT/datasets07/nobackup/EE3D-S/pose_111_18/events.h5"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)

    model_cfg = {'num_joints': 16, 'eros': True, 'input_channel': 20, 'posenet_input_channel': 20, 'image_size': [256, 192], 'batch_size': 6}

    args = argparser.parse_args()

    input_file = args.file
    output = args.output

    folder_name = input_file.split("/")[-2]

    output_path = f"{output}/{folder_name}" 

    os.makedirs(output_path, exist_ok=True)

    # with h5py.File(args.file, 'r') as f:
        # dset = f['event']
    events = h5py_File(args.file, 'r')['event']

    lnes = LNES(cfg, DS_HEIGHT, DS_WIDTH)
    raw_events = RawEvent(cfg, DS_HEIGHT, DS_WIDTH, 10, False)


    event_tensor_model = EgoHPE(cfg, **model_cfg).cuda()

    checkpoint = torch.load(CHECKPOINT_PATH)
    if 'best_state_dict' in checkpoint:
        checkpoint = checkpoint['best_state_dict']

    event_tensor_model.load_state_dict(checkpoint, strict=False)
    event_tensor_model.eval()

    lnes_writer = h5py.File(f"{output_path}/lnes.hdf5", 'w')
    event_tensor_writer = h5py.File(f"{output_path}/event_tensor.hdf5", 'w')


    idx = 0
    total_frames = 0
    while True:
        data_batches = []
        frame_time = 0
        while frame_time < TIME_WINDOW:
            data = events[idx: idx + NUM_EVENTS]
            ts = data[:, 2]

            if not len(ts):
                break

            ts = (ts[-1] - ts[0]) * 1e-3 # microseconds to milliseconds

            data_batches.append(data)

            frame_time += ts
            idx += NUM_EVENTS
        
        if len(data_batches) == 0:
            break

        data_batches_np = np.concatenate(data_batches, axis=0)

        
        lnes_data = lnes(data_batches_np)
        raw_events_data = raw_events(data_batches_np)

        with torch.no_grad():
            tensor_event_repr = event_tensor_model([raw_events_data['input'].cuda()])
        
        raw_events_data['input'] = tensor_event_repr.squeeze(0)

        # frame_path = f"{output_path}/{total_frames}"

        # save_augmented_data(raw_events_data['input'], f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/voxel_preprocess/{total_frames}_tensor.png')

        # lnes_vis = lnes.visualize(lnes_data['input'])
        # cv2.imwrite(f'/CT/EventEgo3Dv2/work/EventEgo3Dv2/visualizations/voxel_preprocess/{total_frames}_lnes.png', lnes_vis)
        
        lnes_grp = lnes_writer.create_group(str(total_frames))
        lnes_grp.create_dataset('input', data=lnes_data['input'], compression="gzip")
        lnes_grp.create_dataset('frame_index', data=lnes_data['frame_index'])

        event_tensor_grp = event_tensor_writer.create_group(str(total_frames))
        event_tensor_grp.create_dataset('input', data=raw_events_data['input'].cpu().numpy(), compression="gzip")
        event_tensor_grp.create_dataset('frame_index', data=raw_events_data['frame_index'])

        total_frames += 1

    with open(f"{output_path}/meta.json", "w") as outfile: 
        json.dump({'total_frames': total_frames}, outfile)

    lnes_writer.close()
    event_tensor_writer.close()

    print("Total frames : ", total_frames)
