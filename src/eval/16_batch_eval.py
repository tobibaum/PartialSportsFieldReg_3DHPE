# Draw the difference between athletes at differnt distances!
import os
from PIL import Image
import cv2
from glob import glob
from pprint import pprint

import sys
sys.path.append('..')

from anim_util import *
from unreal_util import *
from projection_util import *
from mappings import *
from hough_util import *
from lift3d_util import *
from mappings import *

# === run settings ===
comb_path = '/disk2/data/UE5sim_3DHPE'
render_dirname = 'MovieRenders'
unreal_img_extension = 'jpeg'

def get_angle_from_chain(ch):
    a, b, c = ch
    u = a - b
    v = c - b
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return np.arccos((u).dot(v)) / np.pi*180

skip_idc = [7, 9, 10]
good_idc = list(np.arange(17))
for si in skip_idc:
    good_idc.remove(si)


#unreal settings
# unreal_path = '/disk2/data/unreal/0228_1407_horizontalfov/'
files = os.listdir(comb_path)
for f in files:
    print('\n\n\n\n')
    print('==============')
    print(f)
    print('==============')
    
    fname = f.split('/')[-1].split('.')[0]
    unreal_path = os.path.join(comb_path, fname)
    
    unreal_prefix = fname
    img_prefix = fname

    img_fold = os.path.join(unreal_path, 'MovieRenders')
    img_pattern = os.path.join(img_fold, img_prefix + '.%04d.' + unreal_img_extension)
    hough_file = os.path.join(unreal_path, 'hough_results.pkl')
    metrabs_file = os.path.join(unreal_path, 'metrabs.pkl')

    groundtruth = pickle.load(open(os.path.join(unreal_path, 'groundtruth.pkl'), 'rb'))
    frame_to_pred = pickle.load(open(metrabs_file, 'rb'))
    N = len(frame_to_pred)
    
    no_roll = True
    use_gt = False 
    smooth_vanish = True 
    azim_file = '%s/azim_pairs_simgan'%unreal_path
    if no_roll:
        azim_file += '_noroll'   
    if use_gt:
        azim_file += '_gt'
    if smooth_vanish:
        azim_file += '_smoothVP'
    azim_file += '.pkl'

    azim_pairs = pickle.load(open(azim_file, 'rb'))
    print('loaded', azim_file)
    
    # determine the limb lengths for groudntruth and metrabs data
    gt3d_arr = np.array([g['3d'] for k, g in groundtruth.items() if k!=0])
    gt_norm_length = np.linalg.norm(gt3d_arr[:, NECK, :] - gt3d_arr[:, PELVIS, :], axis=1)

    gt_target_limb_lengths = {}
    for l_i, (limb_name, ld) in enumerate(limb_defs.items()):
        lens = []
        for i, lr in enumerate(['left', 'right']):
            a, b = ld[i]
            lengths = np.linalg.norm(gt3d_arr[:, a, :] - gt3d_arr[:, b, :], axis=1) / gt_norm_length
            plt.plot(lengths, label='%s_%s'%(limb_name, lr), color=tab10(l_i), linestyle='--' if i == 0 else '-')
            lens.append(np.mean(lengths))
        gt_target_limb_lengths[limb_name] = np.mean(lens)


    try:
        met3d_arr = np.array([transform_metrabs_to_stadium(f['poses3d'][0].numpy()) for k, f in frame_to_pred.items()])
        met_norm_length = np.linalg.norm(met3d_arr[:, NECK, :] - met3d_arr[:, PELVIS, :], axis=1)
    except Exception as e:
        print(e)
        continue

    met_target_limb_lengths = {}
    for l_i, (limb_name, ld) in enumerate(limb_defs.items()):
        lens = []
        for i, lr in enumerate(['left', 'right']):
            a, b = ld[i]
            lengths = np.linalg.norm(met3d_arr[:, a, :] - met3d_arr[:, b, :], axis=1) / met_norm_length
            plt.plot(lengths, label='%s_%s'%(limb_name, lr), color=tab10(l_i), linestyle='--' if i == 0 else '-')
            lens.append(np.mean(lengths))
        met_target_limb_lengths[limb_name] = np.mean(lens)
        
        
    # render positions from first to last frame! --> target positions=equally spaced.
    # draw view-target positions for all frames!
    from matplotlib import cm

    #use_gt = True
    for use_gt in [True, False]:
        if use_gt:
            res_file = os.path.join(unreal_path, 'results_gt.pkl')
        else:
            res_file = os.path.join(unreal_path, 'results.pkl')

        if os.path.exists(res_file):
            continue

        calc_only = True
        imgs_comp = []
        compare_data = defaultdict(list)
        proj_error = defaultdict(list)
        for frame_i in tqdm(range(1, N)):
        # for frame_i in tqdm(range(1, 50)):
            if not calc_only:
                fig = plt.figure(figsize=(45, 25))

            b = groundtruth_to_b(groundtruth[frame_i])

            camrot = groundtruth[frame_i]['camrot']
            gt2d = groundtruth[frame_i]['2d']
            gt3d = groundtruth[frame_i]['3d']

            skel3d = transform_metrabs_to_stadium(frame_to_pred[frame_i]['poses3d'][0].numpy(), np.array(camrot)/2)
            skel3d_norot = transform_metrabs_to_stadium(frame_to_pred[frame_i]['poses3d'][0].numpy())

            metrabs3d = skel3d - skel3d[0]
            metrabs2d = frame_to_pred[frame_i]['poses2d'][0].numpy()[coco_to_h36m]
            metrabs3d += gt3d[0]
            metrabs3d_norot = skel3d_norot - skel3d_norot[0] + gt3d[0]

            # pick the ablation input!!
            if use_gt:
                y_off = gt3d[0][1]
                skel2d_pick = gt2d
                skel3d_pick = gt3d
                limb_length_pick = gt_target_limb_lengths
            else:
                # kinda fair to chose the gt offset, since we can get there using steps!
                y_off = gt3d[0][1]
                skel2d_pick = metrabs2d
                skel3d_pick = metrabs3d
                limb_length_pick = met_target_limb_lengths

            skel2d_pick = skel2d_pick.copy()
            skel2d_pick[:, 1] = 720 - skel2d_pick[:, 1]

            skel3d_const, sk3d, skel_options, torso_target, = skel_placement_for_frame(b, skel2d_pick, skel3d_pick,
                                                                                       y_off, limb_length_pick)

            new_skel = np.zeros((17, 3))
            for name, opt in skel_options.items():
                idx = limb_chains[name]

                # what skeleton to compare to?!
                sk_comp = sk3d[idx,:] - sk3d[0]
                opt2 = np.array(opt) - skel3d_const[0]
                diff_sk = np.array(opt2) - sk_comp[np.newaxis, ...]
                diff = np.mean(np.linalg.norm(diff_sk, axis=2), 1)

                if 'leg' in name and False:
                    min_idx = np.argmin(np.std(np.array(opt)[:, 1:, 1], axis=1))
                else:
                    min_idx = np.argmin(diff)

                new_skel[idx, :] = opt[min_idx]

            skel3d_const = new_skel
            
            # == compute the reprojection error!!
            metrabs_proj = project_lines(b, metrabs3d)
            metrabs_proj[:, 1] = 720 - metrabs_proj[:, 1]
            
            metrabs_norot_proj = project_lines(b, metrabs3d_norot)
            metrabs_norot_proj[:, 1] = 720 - metrabs_norot_proj[:, 1]

            skel3d_const_proj = project_lines(b, skel3d_const)
            skel3d_const_proj[:, 1] = 720 - skel3d_const_proj[:, 1]

            metrabs_err = np.mean(np.linalg.norm(gt2d - metrabs_proj, axis=1)[good_idc])
            metrabs_norot_err = np.mean(np.linalg.norm(gt2d - metrabs_norot_proj, axis=1)[good_idc])
            skel3d_err = np.mean(np.linalg.norm(gt2d - skel3d_const_proj, axis=1)[good_idc])
            # == compute the reprojection error!!

            pelv_const = skel3d_const[0].copy()
            if not use_gt:
                skel3d_const -= pelv_const
                skel3d_const += gt3d[0]

            compare_data['gt'].append(gt3d)
            compare_data['metrabs'].append(metrabs3d)
            compare_data['metrabs_norot'].append(metrabs3d_norot)
            compare_data['lift3d'].append(skel3d_const)
            proj_error['metrabs'].append(metrabs_err)
            proj_error['metrabs_norot'].append(metrabs_norot_err)
            proj_error['lift3d'].append(skel3d_err)

        pickle.dump({'skels': compare_data, 'projection_errors': proj_error}, open(res_file, 'wb'))
        
        # calculate score for metrabs vs. lift!
        leg_idc = set()
        for name, idc in limb_chains.items():
            if 'leg' in name:
                leg_idc = leg_idc.union(set(idc))

        leg_idc = list(leg_idc)

        method_to_err = defaultdict(list)
        for key in compare_data.keys():
            if key == 'gt':
                continue

            for frame in range(len(compare_data[key])):
                dists = np.linalg.norm(compare_data[key][frame] - compare_data['gt'][frame], axis=1)[good_idc]
                dists_legs = np.linalg.norm(compare_data[key][frame] - compare_data['gt'][frame], axis=1)[leg_idc]
                method_to_err[key].append([np.mean(dists), np.mean(dists_legs)]) 
                
        for i, name in enumerate(['total', 'legs']):
            print('-- %s --'%name)
            for k, v in method_to_err.items():
                print(k, np.mean(v[i]), np.median(v[i]), np.std(v[i]))
