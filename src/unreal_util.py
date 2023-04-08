'''
author: Tobias Baumgartner
date: 04/06/2023
contact: tobibaum@gmail.com
'''

import os
from PIL import Image
from projection_util import *
from stadium_def import *
from mappings import *
from anim_util import *

r_unreal_stadium = R.from_euler('z', -90, degrees=True).as_matrix()

def transform_unreal_to_stadium(_pts):
    # placed a cube in the unreal stadium and measured its position
    #origin_coords = [-3428, -4070, 265] # before scale up
    origin_coords = [-3608, -4250, 280] # with scale up
    # in unreal, 1 unit is 1cm, we have 1 unit = 1m
    scale = 100
    
    orig = np.array(origin_coords.copy())
    orig[1]*=-1
#     transform unreal to stadium!!

    pts = _pts.copy()
    pts -= orig
    pts = pts.dot(r_unreal_stadium) / scale
    return pts

def draw_stadium(ax, _lanes=lanes_new, **kwargs):
    for sub_lanes in _lanes:
        for i, l in enumerate(sub_lanes):
            ax.plot3D(*l.T, linewidth=2, c='blue', **kwargs)

def draw_stadium_top_view(ax, _lanes=lanes_new):
    for sub_lanes in _lanes:
        for i, l in enumerate(sub_lanes):
            ax.plot(*l[:2].T, linewidth=2, c='blue')

def transform_metrabs_to_stadium(skel3d, camrot=[0, 0, 0]):
    # coco to h36m skeleton
    pred3d = skel3d.copy()[coco_to_h36m]
    
    # metrabs --> unreal. some rotates and a good ole scale: cm vs mm.
    r = R.from_euler('x', 90, degrees=True).as_matrix()
    pred3d = pred3d.dot(r) / 10
    pred3d = pred3d.dot(R.from_euler('z', -camrot[1], degrees=True).as_matrix())
    
    # unreal --> stadium
    pred3d = transform_unreal_to_stadium(pred3d)
    return pred3d

def get_matrix_from_string(_proj):
    mat = []
    for p in _proj.split(']'):
        clean = p.strip().replace('[', '').split(' ')
        if len(clean) != 4:
            continue
        row = [float(c) for c in clean]
        mat.append(row)
    return mat

def parse_data_from_file(log_file, pref_name='STADIUM'):
    # == collect data for each of the stored data.
    
    # read the file
    with open(log_file) as fh:
        lines = [l.strip() for l in fh.readlines()]

    collect = defaultdict(list)
    for l in lines:
        p = l.find(pref_name)
        if p != -1:            
            parts = l[p:].split(' ')
            rec_class = parts[0][len(pref_name)+1:]

            frame = int(parts[2])

            data = parts[3:]

            if rec_class in ['Projection', 'ViewMatrix', 'ViewProjection']:
                data = ' '.join(parts[3:])
            collect[rec_class].append([frame, data])
    
    # joints
    frame_to_all_joints = defaultdict(list)
    for f, d in collect['Joints']:
        frame_to_all_joints[f].append(d)
    single = ['pelvis', 'head', 'neck_01', 'spine_04']
    both = ['thigh', 'calf', 'foot', 'upperarm', 'lowerarm', 'hand']
    joints = single
    for b in both:
        joints.append(b+'_l')
        joints.append(b+'_r')

    frame_to_joints = {}
    for f, rows in frame_to_all_joints.items():
        picks = {}
        for joint_row in rows:
            name = joint_row[0]
            if name in joints:
                dat = []
                for jr in joint_row[1:]:
                    d = float(jr.split('=')[1])
                    dat.append(d)
                picks[name] = dat
        frame_to_joints[f] = picks
    print('joints', len(frame_to_joints))

    # rot
    frame_to_camrot = {}
    for f, rot in collect['CamRot']:
        d = []
        for r in rot:
            d.append(float(r.split('=')[1]))
        frame_to_camrot[f] = d
    print('rot', len(frame_to_camrot))

    # pos
    frame_to_campos = {}
    for f, rot in collect['CamPos']:
        d = []
        for r in rot:
            d.append(float(r.split('=')[1]))
        frame_to_campos[f] = d
    print('pos', len(frame_to_campos))

    # view
    frame_to_viewdir = {}
    for f, rot in collect['ViewDir']:
        d = []
        for r in rot:
            d.append(float(r.split('=')[1]))
        frame_to_viewdir[f] = d
    print('view', len(frame_to_viewdir))

    # bodypos
    frame_to_bodpos= {}
    for f, rot in collect['BodPos']:
        d = []
        for r in rot:
            d.append(float(r.split('=')[1]))
        frame_to_bodpos[f] = d
    print('bodypos', len(frame_to_bodpos))

    # fov
    frame_to_fov = {}
    for f, rot in collect['CamFoc']:
        frame_to_fov[f] = float(rot[0])
    print('fov', len(frame_to_fov))

    # projmat
    frame_to_proj = {}
    for f, proj in collect['Projection']:
        frame_to_proj[f] = get_matrix_from_string(proj)
    print('projmat', len(frame_to_proj))

    # viewmat
    frame_to_view = {}
    for f, proj in collect['ViewMatrix']:
        frame_to_view[f] = get_matrix_from_string(proj)
    print('viewmat', len(frame_to_view))

    # viewProjmat
    frame_to_viewproj = {}
    for f, proj in collect['ViewProjection']:
        frame_to_viewproj[f] = get_matrix_from_string(proj)
    print('viewProjmat', len(frame_to_viewproj))

    # store to dict
    save_dict = {'fov': frame_to_fov,
                 'camrot': frame_to_camrot,
                 'joints': frame_to_joints,
                 'campos': frame_to_campos,
                 'projection': frame_to_proj,
                 'viewMatrix': frame_to_view,
                 'viewProjection': frame_to_viewproj,
                 'viewdir': frame_to_viewdir,
                 'bodpos': frame_to_bodpos}
    return save_dict

def convert_json_to_groundtruth(sim_data, img_pattern, unreal_path, 
                                single_frame=False, render=True):
    N = len(sim_data['camrot'])
    
    # convert ue5 joints to h36m
    joint_order = ['pelvis','thigh_l','calf_l','foot_l','thigh_r','calf_r', 
                   'foot_r', 'spine_04', 'neck_01', None, 'head',
                   'upperarm_r', 'lowerarm_r','hand_r','upperarm_l', 
                   'lowerarm_l', 'hand_l'
                  ]

    t = type(list(sim_data['joints'].keys())[0])
        
    skel = np.zeros((N+1, 17, 3))
    for f in range(1, N):
        for j, joint in enumerate(joint_order):
            if joint is None:
                continue
            key = t(f)
            if not key in sim_data['joints']:
                continue
            skel[f, j] = sim_data['joints'][key][joint]
    
    frames = [121, 70, 89, 90, 91, 123, 250, 290]
    if single_frame:
        plt.figure(figsize=(20, 5*len(frames)/2))

    groundtruth = {}
    imgs = []
    for i, frame in tqdm(enumerate(range(N)), total=N):
        if single_frame:
            frame = frames[i]
            ax = plt.subplot(len(frames)//2, 2, i+1)

        img_path = img_pattern%(frame)
        frame_gt = frame

        t = type(list(sim_data['fov'].keys())[0])

        if not os.path.exists(img_path):
            print('MISSING', img_path)
            continue    
        if not t(frame_gt) in sim_data['fov']:
            continue

        img = np.array(Image.open(img_path).convert('RGB'))

        # read data from storage
        fov = sim_data['fov'][t(frame_gt)]
        camrot = sim_data['camrot'][t(frame_gt)]
        cam_pos = sim_data['campos'][t(frame_gt)].copy()
        elev, azim, roll = camrot

        # construct the rotation and transform from unreal to our coordinates
        r_raw = rotation_from_azelro(azim+90, -elev, roll)
        r = r_unreal_stadium.dot(r_raw)

        # adjust the camera to our coordinates (unreal right handed, plt left handed o_O)
        cam_pos[1] *= -1
        cam_pos = transform_unreal_to_stadium(cam_pos)

        # get the 3d skeelton adn transform it to our default coordinates
        pos = skel[frame_gt].copy()
        pos[:, 1] *= -1
        pos = transform_unreal_to_stadium(pos)

        params = {'r': r, 'cam_pos': cam_pos, 'fov': fov}
        lines = pos_to_coco_lines(pos)
        lines_proj = project_lines(params, lines)
        pos_proj = project_lines(params, pos)

        # the groundtruth and images are shifted by 
        groundtruth[frame] = {'2d': pos_proj, '3d': pos, 'projection': params, 
                              'camrot': camrot, 'campos': cam_pos, 'fov': fov}

        if single_frame:
            for j, l in enumerate(lines_proj):
                ax.plot(*l.T, linewidth=2, color=tab20(j), alpha=.5)

            ax.imshow(img)

            # = test plot projection
            gt3d = groundtruth[frame]['3d']
            params = groundtruth[frame]['projection']        
            proj2d_target = project_lines(params, gt3d)
            ax.scatter(*proj2d_target.T, color='red', s=5)
            # = test plot projection

            # draw stadium lines.
            for lanes in lanes_new:
                proj_lines = project_lines(params, np.array(lanes))
                for line in proj_lines:
                    plt.plot(*line.T, color='blue')

            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)
            if i>=len(frames)-1:
                plt.show()
                break
        else:
            col = (255, 50, 50)
            for j, l in enumerate(lines_proj):
                start = (int(l[0][0]), int(l[0][1]))
                end = (int(l[1][0]), int(l[1][1]))
                cv2.line(img, start, end, col, 2, lineType=cv2.LINE_AA)

            cv2.putText(img, '%s'%frame, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 100, 100))
            imgs.append(img)
            
    if not single_frame:
        if render:
            from anim_util import create_anim_from_array
            anim = create_anim_from_array(imgs, figsize=(25,20))
            from IPython.display import clear_output
            def callback_fun(i, n):
                print(f'\rSaving frame {i} of {n}',)
                clear_output(wait=True)
            writervideo = animation.FFMpegWriter(fps=30)
            anim_path = os.path.join(unreal_path, 'render_our_projection.mp4')
            anim.save(anim_path, writer=writervideo, progress_callback=callback_fun)

        # save the groundtruth
        pickle.dump(groundtruth, open(os.path.join(unreal_path, 'groundtruth.pkl'), 'wb'))
    return groundtruth
