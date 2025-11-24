import h5py
import numpy as np
import numba
import argparse

@numba.njit
def count_frames(frame_ids, max_frame):
    counts = np.zeros(max_frame + 1, dtype=np.int64)
    for f in frame_ids:
        counts[f] += 1
    return counts

# file = "/CT/datasets07/nobackup/EE3D-S/pose_111_18/events.h5"

# with h5py.File(file, 'r') as f:
#     dset = f['event']
#     frame_ids = dset[:, 4].astype(np.int64)  # this still loads the column into memory

# max_frame = int(frame_ids.max())
# min_frame = int(frame_ids.min())
# print("Max frame:", max_frame)
# print("Min frame:", min_frame)
# frame_counts = count_frames(frame_ids, max_frame)
# np.save("frame_counts.npy", frame_counts)
# print(frame_counts)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)

    args = argparser.parse_args()

    input_file = args.file
    output = args.output

    folder_name = input_file.split("/")[-2]

    with h5py.File(args.file, 'r') as f:
        dset = f['event']
        frame_ids = dset[:, 4].astype(np.int64)

    max_frame = int(frame_ids.max())
    min_frame = int(frame_ids.min())
    print("Max frame:", max_frame)
    print("Min frame:", min_frame)
    frame_counts = count_frames(frame_ids, max_frame)
    np.save(f"{output}/{folder_name}.npy", frame_counts)
