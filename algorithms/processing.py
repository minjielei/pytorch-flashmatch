import numpy as np

def extend_track(detector, threshold, segment_size, track):
    min_idx = np.argmin(np.array(track)[:, 0])
    max_idx = np.argmax(np.array(track)[:, 0])

    if ((track[min_idx][0] - detector['ActiveVolumeMin'][0] < threshold) or
        (track[min_idx][1] - detector['ActiveVolumeMin'][1] < threshold) or
        (track[min_idx][2] - detector['ActiveVolumeMin'][2] < threshold) or 
        (detector['ActiveVolumeMax'][0] - track[max_idx][0] < threshold) or
        (detector['ActiveVolumeMax'][1] - track[max_idx][1] < threshold) or
        (detector['ActiveVolumeMax'][2] - track[max_idx][2] < threshold)):

        A = np.array([track[max_idx][0], track[max_idx][1], track[max_idx][2]])
        B = np.array([track[min_idx][0], track[min_idx][1], track[min_idx][2]])
        AB = B - A
        x_C = detector['PhotonLibraryVolumeMin'][0]
        lengthAB = np.linalg.norm(AB)
        lengthBC = (x_C - B[0]) / (B[0] - A[0]) * lengthAB
        C = B + AB / lengthAB * lengthBC
        unit = (C - B) / np.linalg.norm(C - B)

        # add to track the part between boundary and C
        num_pts = int(lengthBC / segment_size)
        current = np.copy(B)
        for i in range(num_pts+1):
            current_segment_size = segment_size if i < num_pts else (lengthBC - segment_size*num_pts)
            current = current + unit * current_segment_size / 2.0
            q = current_segment_size * detector['LightYield'] * detector['MIPdEdx']
            if (track[0][0] < track[-1][0]):
                track.insert(0, [current[0], current[1], current[2], q])
            else:
                track.append([current[0], current[1], current[2], q])
            current = current + unit * current_segment_size / 2.0