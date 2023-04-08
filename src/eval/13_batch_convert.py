# perform the conversino steps in batches!

# parse the relevant bits out of the file
import cv2
from collections import defaultdict
import json
import os
import json
import numpy as np
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import sys
sys.path.append('..')

from mappings import pos_to_coco_lines
from anim_util import *
from glob import glob
from hough_util import *
from projection_util import *
from unreal_util import *

# === run settings ===
comb_path = '/disk2/data/UE5sim_3DHPE'
render_dirname = 'MovieRenders'
unreal_img_extension = 'jpeg'

# === initialize metrabs
do_metrabs = True
if do_metrabs:
    import tensorflow as tf
    import tensorflow_hub as hub
    model_file = '/home/tobi/apps/pose_estimation/models/metrabs_xl'
    model_metrabs = hub.load(model_file)

# == inititalize lookup
print('load lookup')
ranges = {'azim': list(np.arange(-89, -1, 2)),
  'elev': list(np.arange(-25, -1, 1)),
  'dist': [50],
  'fov': [40],#list(np.arange(1, 10, 2)),
  'F': [1]#list(np.arange(1, 10, 2))
 }
no_roll = True
use_gt = False
smooth_vanish = True 
df_lookup = create_look_up_grid(ranges, no_roll)

# === grab all folders ===
#files = glob(os.path.join(comb_path, '*.txt'))
files = os.listdir(comb_path)
for f in files:
    print('\n\n')
    print('==============')
    print(f)
    print('==============')

    # grab the experimental name and prep its folder.
    fname = f.split('/')[-1].split('.')[0]
    unreal_path = os.path.join(comb_path, fname)

    azim_file = '%s/azim_pairs_simgan'%unreal_path
    if no_roll:
        azim_file += '_noroll'
    if use_gt:
        azim_file += '_gt'
    if smooth_vanish:
        azim_file += '_smoothVP'
    azim_file += '.pkl'
    if os.path.exists(azim_file):
        continue

    # collect all the images from this experiment and copy them to MovieRenders
    img_paths = glob(os.path.join(comb_path, render_dirname, '%s*.png'%fname))
    unreal_prefix = fname
    img_prefix = fname
    print(unreal_prefix, unreal_path)

    # === shared file stuff
    log_file = os.path.join(unreal_path, '%s.txt'%unreal_prefix)
    json_file = os.path.join(unreal_path, '%s.json'%unreal_prefix)
    if not os.path.exists(unreal_path):
        os.makedirs(unreal_path)
    img_fold = os.path.join(unreal_path, 'MovieRenders')
    img_pattern = os.path.join(img_fold, img_prefix + '.%04d.' + unreal_img_extension)
    hough_file = os.path.join(unreal_path, 'hough_results.pkl')
    metrabs_file = os.path.join(unreal_path, 'metrabs.pkl')
    groundtruth_file = os.path.join(unreal_path, 'groundtruth.pkl')

    # === read txt ===
    print('load json data')
    sim_data = parse_data_from_file(log_file)
    N = len(sim_data['camrot']) - 1
    print('save as ...', json_file)
    json.dump(sim_data, open(json_file, 'w'))

    # === parse into groundtruth ===
    if os.path.exists(groundtruth_file) and False:
        groundtruth = pickle.load(open(os.path.join(unreal_path, 'groundtruth.pkl'), 'rb'))
    else:
        groundtruth = convert_json_to_groundtruth(sim_data, img_pattern, unreal_path,
                render=False)

    # === extract metrabs results for all frames! ===
    if not os.path.exists(metrabs_file) and do_metrabs:
        plot_it = False
        frame_to_pred = {}
        vid_array = []
        for frame in tqdm(range(N)):
            if plot_it:
                frame = 132

            img_path = img_pattern%frame
            if not os.path.exists(img_path):
                print('MISSING', img_path)
                continue
            img = np.array(Image.open(img_path).convert('RGB'))
            vid_array.append(img)
            dn = 'coco_19'
            pred = model_metrabs.detect_poses(img, skeleton=dn)
            # TODO: make numpy array instead of tensor
            frame_to_pred[frame] = pred

            if plot_it:
                plt.figure(figsize=(15,10))
                plt.imshow(img)

                pose2d = pred['poses2d'][0].numpy()[coco_to_h36m]
                gt2d = groundtruth[frame]['2d']

                lines_gt = pos_to_coco_lines(gt2d)
                for l in lines_gt:
                    plt.plot(*l.T, color='green')

                lines_pred = pos_to_coco_lines(pose2d)

                for i, p in enumerate(pose2d):
                    plt.text(*p, '%d'%i, color='red')

                for l in lines_pred:
                    plt.plot(*l.T, color='blue')
                plt.xlim(400, 600)
                plt.ylim(500, 300)
                plt.show()

                break
        pickle.dump(frame_to_pred, open(metrabs_file, 'wb'))
    else:
        print('load metrabs results from disk:', metrabs_file)
        frame_to_pred = pickle.load(open(metrabs_file, 'rb'))

        # collect all imgs
        print('load images...')
        vid_array = []
        for frame in tqdm(range(N)):
            img_path = img_pattern%frame
            if not os.path.exists(img_path):
                print('MISSING', img_path)
                continue
            img = np.array(Image.open(img_path).convert('RGB'))
            vid_array.append(img)

    # === extract geometry! ===
    # 1. determine vanishing points

    if not os.path.exists(hough_file):
        imgs = []
        frame_to_vanish_all = {}
        proto_segs = {}
        frame_to_hough_dict = {}

        hough_arg_dict = {}
        plot_it = False
        for i in tqdm(range(N)):
            img = vid_array[i].copy()
            plines, nplines = get_lines_for_img(img, arg_dict=hough_arg_dict)
            frame_to_hough_dict[i] = plines

            proto_segments = get_main_segments_from_lines(plines[:, 0, :])
            for k, p in enumerate(proto_segments):
                proto_segments[k] = seg_to_full_width(img, p)

            proto_seg = proto_segments.astype(int)[:, np.newaxis, :]
            v0, v1 = determine_vanish_points(proto_seg, nplines)
            frame_to_vanish_all[i] = [v0, v1]

            img = draw_lines_on_img(img, proto_seg, nplines)
            midline = proto_seg[len(proto_seg)//2, 0, :]

            proto_segs[i] = proto_seg

            if plot_it:
                print(v0)
                plt.scatter(*v0)
                plt.imshow(img)
                plt.xlim(-3000, 1300)
                plt.ylim(800, -800)
                break

            imgs.append(img)
        store_dict = {'vanish': frame_to_vanish_all, 'proto': proto_segs, 'plines': frame_to_hough_dict}
        pickle.dump(store_dict, open(hough_file, 'wb'))
    else:
        store_dict = pickle.load(open(hough_file, 'rb'))
        frame_to_vanish_all = store_dict['vanish']
        proto_segs = store_dict['proto']
        frame_to_hough_dict = store_dict['plines']

    frame_to_proto_seg_to_id, frame_lane_id_to_proto_id = track_lanes(proto_segs)

    # == get gt VPs ==
    # get the groundtruth vanishing points!
    gt_v0s = {}
    gt_protosegs = {}
    gt_frame_lane_id_to_proto_id = {}
    for frame in range(N):
        if frame not in groundtruth: 
            continue
        # 1. get params and project
        params = groundtruth[frame]['projection']
        lanes2d = project_lines(params, lanes_new[1])

        # 2. crop to protosegs
        ps = []
        for l in lanes2d[::2]:
            l2 = seg_to_full_width(img, l)
            ps.append(l2.reshape(2,2))
        ps = np.array(ps)

        # 3. determine the v0 and store
        v0 = get_median_vanish(ps[:, np.newaxis])
        gt_protosegs[frame] = ps.reshape(-1, 4)[:, np.newaxis]
        gt_v0s[frame] = [v0, None]
        gt_frame_lane_id_to_proto_id[frame] = {i:i for i in range(len(ps))}

    # == smooth VPs ==
    # smooth the measured vanishing points!
    v0s = np.vstack([v[0] for v in frame_to_vanish_all.values()])
    window = 20
    smooth_together = smooth(v0s, window)
    v0x = v0s[:, 0]
    v0x_smooth = np.hstack((v0x[:window//2], smooth(v0x, window)))
    v0y = v0s[:, 1]
    v0y_smooth = np.hstack((v0y[:window//2], smooth(v0y, window)))
    gt_v0_arr = np.array([v[0] for k, v in gt_v0s.items()])
    frame_to_vanish_all_smooth = {k: [v, None] for k, v in enumerate(zip(v0x_smooth, v0y_smooth))}


    # == run geometry extraction! ==
    print('extract geometry!')
    import os
    from multiprocess import Pool
    from importlib import reload
    import _pool_func_azim
    reload(_pool_func_azim)


    max_pool = 12
    N = len(vid_array)

    if os.path.exists(azim_file):# and False:
        azim_pairs = pickle.load(open(azim_file, 'rb'))
        print('loaded', azim_file)
    else:
        if smooth_vanish:
            vanish_points = frame_to_vanish_all_smooth
        else:
            vanish_points = frame_to_vanish_all
        
        if use_gt:
            proto_segs_pick = gt_protosegs
            vanish_points = gt_v0s
            frame_lane_id_to_proto_id_pick = gt_frame_lane_id_to_proto_id
        else:
            proto_segs_pick = proto_segs
            vanish_points = frame_to_vanish_all
            frame_lane_id_to_proto_id_pick = frame_lane_id_to_proto_id
                
        with Pool(max_pool) as p:
            pool_outputs = list(
                tqdm(
                    p.imap(lambda k: _pool_func_azim.process_k_(k, df_lookup, proto_segs_pick, 
                                    frame_lane_id_to_proto_id_pick, vanish_points,
                                    {}),
                           range(N)),
                    total=N
                )
            )
        
        azim_pairs = dict(pool_outputs)
        pickle.dump(azim_pairs, open(azim_file, 'wb'))
