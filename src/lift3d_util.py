from projection_util import *

NECK = 8
PELVIS = 0

limb_defs = {
    'hip': [[0, 1], [0, 4]],
    'thigh': [[1, 2], [4, 5]],
    'shin': [[2, 3], [5, 6]],
    'shoulder': [[8, 14], [8, 11]],
    'upper_arm': [[14, 15], [11, 12]],
    'lower_arm': [[15, 16], [12, 13]]
}

limb_chains = { \
    'left_leg': [0, 1, 2, 3],
    'right_leg': [0, 4, 5, 6],
    'left_arm': [8, 14, 15, 16],
    'right_arm': [8, 11, 12, 13]
              }

reconst_joints = []
for limb_name, limb in limb_defs.items():
    for l in limb:
        reconst_joints += l
joint_order = list(set(reconst_joints))

def plot_reconstructed_skel(sk, ax=None, col='blue'):
    segs = []
    for limb_name, limb in limb_defs.items():
        for limb_a, limb_b in limb:
            lseg = np.vstack((sk[limb_a], sk[limb_b]))
            segs.append(lseg)
            if ax is not None:
                ax.plot3D(*lseg.T, color=col)
    return segs
            
def skel_placement_for_frame(b, _skel2d, _skel3d, y_off, target_limb_lengths, adjust3d=False):
    
    skel2d_cast = _skel2d.copy()
    neck3d, pelv3d = cast_neck_and_pelv(b, skel2d_cast, y_off)
    torso_target = np.linalg.norm(pelv3d - neck3d)

    # load the metrabs 3d prediction and fit it to the visible 2D HPE
    if adjust3d:
        r_z = R.from_euler('z', 90, degrees=True).as_matrix()
        r_x = R.from_euler('x', -90, degrees=True).as_matrix()
        r = r_z.dot(r_x)
        r_y = R.from_euler('z', b['azim'] + 90, degrees=True).as_matrix()
        skel3d = (_skel3d.copy()).dot(r.T).dot(r_y)
        torso_height = np.linalg.norm(skel3d[0] - skel3d[8])
        skel3d = skel3d / torso_height * torso_target
    else:
        skel3d = _skel3d.copy()
    
    # shift by pelvis
    skel3d -= skel3d[0]
    skel3d += pelv3d

    new_3d_locations, skel_options = cast_skel_into_scene(b, y_off, skel2d_cast, skel3d, target_limb_lengths, torso_target)
    return new_3d_locations, skel3d, skel_options, torso_target

def cast_neck_and_pelv(b, skel2d_cast, y_off):
    cam_pos = b['cam_pos']
    
    pelv3d = ray_cast_point(skel2d_cast[0], cam_pos, b['image_plane'], z_off=y_off, axis=1)
    neck3d = ray_cast_point(skel2d_cast[8], cam_pos, b['image_plane'], z_off=y_off, axis=1)
    return neck3d, pelv3d

def cast_skel_into_scene(b, y_off, skel2d_cast, _skel3d_prev, _target_limb_lengths, torso_target):
    '''
    3d cast skeleton with known 2d pose
    '''
    # cast the pelvis to y=-.?

    cam_pos = b['cam_pos']
    
    neck3d, pelv3d = cast_neck_and_pelv(b, skel2d_cast, y_off)
    
    skel3d_prev = _skel3d_prev.copy()
    skel3d_prev -= skel3d_prev[0]
    skel3d_prev += pelv3d
    
    # iterate over timeframes. take the minimum one.
    new_3d_locations = {8: neck3d, 0: pelv3d}
    b_to_len = {}
    # cast joints into scene and draw it!
    limb = limb_defs['shoulder']
    for limb_name, limb in limb_defs.items():
        for limb_a, limb_b in limb:
            endpoint3d = ray_cast_point(skel2d_cast[limb_b], cam_pos, b['image_plane'], z_off=0)

            # determine the 2 candidates for the location of this limb-end!
            startpt = new_3d_locations[limb_a]
            limb_len = _target_limb_lengths[limb_name] * torso_target
            p1, t1, p2, t2 = find_points_with_distance_to_line(cam_pos, endpoint3d, startpt, limb_len)
            
            b_to_len[limb_b] = limb_len

            # now we gotta pick one the points that makes it into our new skeleton. 
            # take the one closer to the 3d prediction :/
            if np.linalg.norm(skel3d_prev[limb_b] - p1) < np.linalg.norm(skel3d_prev[limb_b] - p2):
                pp = p1
            else:
                pp = p2

            new_3d_locations[limb_b] = pp
    
    # == generate all options ==
    all_options = {}
    for name, chain in limb_chains.items():
        # append a list of just the root into the options
        options = [[new_3d_locations[chain[0]]]]
            
        # now append both next options for every step to the root
        for i, ch in enumerate(chain[1:]):
            # grow the chain one limb at a time!
            new_opts = []
            for o in options:
                c0 = o.copy()
                c1 = o.copy()
                startpt = o[-1]
                
                limb_b = ch
                limb_len = b_to_len[limb_b]
                endpoint3d = ray_cast_point(skel2d_cast[limb_b], cam_pos, b['image_plane'], z_off=0)
                p1, t1, p2, t2 = find_points_with_distance_to_line(cam_pos, endpoint3d, startpt, limb_len)
                
                c0.append(p1)
                c1.append(p2)
                new_opts.append(c0)
                new_opts.append(c1)
            options = new_opts
        
        
        all_options[name] = options
                
    return new_3d_locations, all_options


def skel_dict_to_array(skel_dict):
    output = np.zeros((17, 3))
    for k, v in skel_dict.items():
        output[k, :] = v
    return output