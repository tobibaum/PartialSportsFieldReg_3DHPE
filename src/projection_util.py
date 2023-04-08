from scipy.spatial.transform import Rotation as R
import math
from hough_util import *
import tensorflow as tf
import time
import itertools
from pandas import DataFrame
from collections import defaultdict

from stadium_def import *
from scipy.optimize import minimize

# ignore the divide by zero caused by meanshift algo.
np.seterr(all='ignore')

h, w = 720, 1280
tab10 = plt.get_cmap('tab10')
tab20 = plt.get_cmap('tab20')
# construct 3D scene and project down to create 2D view.
n_lanes = 6
n_orth = 3
n_vert = 3
lane_width = 1.22
lanes_main = np.array([[[0, i*lane_width, 0], [10, i*lane_width, 0]] for i in range(n_lanes)])
lanes_vert = np.array([[[1.8, (2.5+i)*lane_width, 0], [1.8, (2.5+i)*lane_width, 1.6]] for i in range(n_vert)])
lanes_orth = np.array([[[j+5, 0, 0], [j+5, 10, 0]] for j in range(n_orth)])
lanes = np.vstack((lanes_main, lanes_vert, lanes_orth))

ANK_RIGHT_IND = 3
ANK_LEFT_IND = 6
NECK_IND = 8
PELVIS_IND = 0

def azim_to_vec_tf(az, el, degrees=True):
    if degrees:
        az = az / 180 * math.pi
        el = el / 180 * math.pi
    # deriv: https://math.stackexchange.com/questions/1150232/finding-the-unit-direction-vector-given-azimuth-and-elevation
    sina = tf.sin(az)
    cosa = tf.cos(az)
    sinb = tf.sin(el)
    cosb = tf.cos(el)
    return tf.stack([sina*cosb, cosa*cosb, sinb])

def azel_from_view_dir(v):
    c = np.sqrt(v[0]**2+v[1]**2)
    
    az_ = np.degrees(np.arctan(v[0]/v[1]))
    el_ = np.degrees(np.arctan(v[2]/c))
    return az_, el_

def azel_from_vec(view_target_, cam_vec_, degrees=True):
    if not degrees:
        raise
    v = view_target_ - cam_vec_
    return azel_from_view_dir(v)

def azim_to_vec(az, el, degrees=True):
    if degrees:
        az = az / 180 * np.pi
        el = el / 180 * np.pi
    else:
        print('WARNING. azim_to_vec degrees=False')
    # deriv: https://math.stackexchange.com/questions/1150232/finding-the-unit-direction-vector-given-azimuth-and-elevation
    sina = np.sin(az)
    cosa = np.cos(az)
    sinb = np.sin(el)
    cosb = np.cos(el)
    return np.array([sina*cosb, cosa*cosb, sinb])

def rotation_from_azelro(azim, elev, roll):
    r_pan = R.from_euler('z', azim, degrees=True).as_matrix()
    r_tilt = R.from_euler('x', elev, degrees=True).as_matrix()
    r_roll = R.from_euler('y', roll, degrees=True).as_matrix()
    r = r_pan.T.dot(r_tilt).dot(r_roll)
    return r

def rotation_from_view(view_dir, ignore_flip=False):
    azim, elev = azel_from_view_dir(view_dir)
    r_pan = R.from_euler('z', azim, degrees=True).as_matrix()
    r_tilt = R.from_euler('x', elev, degrees=True).as_matrix()
    r = r_pan.T.dot(r_tilt)
    # flip around horizontal axis! cuz images have an origin of (0, 720) in plt
    if not ignore_flip:
        r[:, 2] *= -1
    return r

def compute_projection(camx, camy, camz, azim, elev, fov, F, _lanes=lanes_new,
                       roll=0, w=1280, h=720, **kwargs):
    #if len(kwargs) > 0:
    #    print('[WARN] comp-proj ignore:', kwargs.keys())
    view_dir = azim_to_vec(azim, elev, degrees=True)
    view_dir = view_dir / np.linalg.norm(view_dir)
    
    # modify camera parameters
    cam_pos = np.array([camx, camy, camz])
    ratio = w/h #16/9
    
    r = rotation_from_view(view_dir, ignore_flip=True)
    r2 = R.from_euler('y', roll, degrees=True).as_matrix()    
    r = r.dot(r2)
    
    # construct the viewport
    lf = np.tan(fov / 180 * np.pi / 2) * F
    bt = lf / ratio
    base_pts = np.array([[lf, F, bt],
                         [lf, F, -bt],
                         [-lf, F, -bt],
                         [-lf, F, bt]
                        ])
    image_plane = base_pts.dot(r.T) + cam_pos
    
    shift = np.array([w,h])/2
    
    # easier method! with just matrix operations and no plane intersections :roll_eyes:
    lines_2ds = []
    lanes_rots = []
    for sub_lanes in _lanes:
        lanes_rot = (sub_lanes - cam_pos).dot(r)
        # TODO!!! something is still fishy here?! with clipping and such!
        
        # clip area behind camera
        lanes_clipped = []
        for lane in lanes_rot:
            l_clip = clip_segment(lane)
            lanes_clipped.append(l_clip)
            
        pts3d = np.vstack(lanes_clipped)
        # actual projection!
        pts2d = pts3d / (pts3d[:, 1, np.newaxis] + 1e-6)
        
        lines_2d = []
        a = 0
        for lane in lanes_clipped:
            # unstack the points!
            n_seg = lane.shape[0]
            b = a + n_seg
            l2d = pts2d[a:b, [0, 2]]/ (2*lf)*w + shift
            lines_2d.append(l2d)
            a = b
            
        lines_2ds.append(lines_2d)
        lanes_rots.append(lanes_rot)

    # == vanishing points! ==
    # x-direction = v0
    # y-direction = v1
    # these are the lines of only 2 point, i.e., straight lines.
    t = (cam_pos[2] - view_dir[2]) / view_dir[2]
    vp_target = cam_pos - view_dir*(1+t)
    tl = vp_target + [-1, -1, 0]
    tr = vp_target + [-1, 1, 0]
    bl = vp_target + [1, -1, 0]
    br = vp_target + [1, 1, 0]
    vp_pts = np.array([tl,tr,bl,br])
    vp_rot = (vp_pts - cam_pos).dot(r)
    vp2d = vp_rot / (vp_rot[:, 1, np.newaxis] + 1e-6)
    vp2d = vp2d[:, [0, 2]] / (2*lf)*w + shift
    v0 = line_intersection([vp2d[0], vp2d[2]], [vp2d[1], vp2d[3]])
    v1 = line_intersection([vp2d[0], vp2d[1]], [vp2d[2], vp2d[3]])
    if v0 is None:
        v0 = [-1, -1]
    if v1 is None:
        v1 = [-1, -1]
    # == vanishing points! ==
    
    res_dict = {'cam_pos': cam_pos, 
                'view_dir': view_dir,
                'base_pts': base_pts[:,[0,2]] / (2*lf)*w + shift,
                'image_plane': image_plane,
                'lines_2ds': lines_2ds,
                'lanes_rots': lanes_rots,
                'lf': lf,
                'w': w,
                'r': r,
                'v0': np.array(v0),
                'v1': np.array(v1)
               }
    
    return res_dict

def draw_base(ax1, base_pts):
    #lf = base_pts[0][0]
    #shift = np.array([w,h])/2
    for j, bp in enumerate(base_pts):
        if j >= len(base_pts)-1:
            j = -1
        bp_line = np.vstack((bp, base_pts[j+1]))#[:, [0,2]]
        #ax1.plot(*(bp_line/(lf*2)*w + shift).T, linewidth=2, color='black')
        ax1.plot(*bp_line.T, linewidth=2, color='black')

def draw2d(ax1, lines_2ds, base_pts=None, color=None, trans_fun=None, **kwargs):
    # projection plot
    for lines_2d in lines_2ds:
        for i, l in enumerate(lines_2d):
            col = color if color is not None else tab20(i)
            if not 'linewidth' in kwargs:
                kwargs['linewidth'] = 2
            if trans_fun is not None:
                l = trans_fun(l)
            
            ax1.plot(*l.T, c=col, **kwargs)
    
    if base_pts is not None:
        draw_base(ax1, base_pts)

    ax1.set_ylim(-.35*w, w*.85)
    ax1.set_xlim(-.1*w, w*1.1)
    
def draw_scene(res_dict, ax=None, ax1=None, ax2=None, 
               _lanes=lanes, w=1280, h=720, fake_F=1, **kwargs):
    # compute for given params
    cam_pos = res_dict['cam_pos']
    view_dir = res_dict['view_dir']
    image_plane = res_dict['image_plane'] 
    lines_2ds = res_dict.get('lines_2ds', None)
    base_pts = res_dict.get('base_pts', None)
    
    # === vizualization ===
    # 3d and projected 2d lines (in 3d)
    for sub_lanes in _lanes:
        for i, l in enumerate(sub_lanes):
            #ax.plot3D(*l.T, linewidth=3, c=tab20(i))
            ax.plot3D(*l.T, linewidth=2, c='blue')
            
    # draw the camera
    ax.scatter(*cam_pos, s=50, color='orange')#tab10(0))
    ax.plot3D(*np.vstack((cam_pos, cam_pos+view_dir*fake_F)).T, linewidth=5, c='green')#tab10(0))
    for j, ip in enumerate(image_plane):
        ax.plot3D(*np.vstack((cam_pos, ip + (ip-cam_pos)*(fake_F-1))).T, linewidth=3, color='black')
        if j >= len(image_plane)-1:
            j = -1
        vec0 = cam_pos + (ip - cam_pos)*fake_F
        vec1 = cam_pos + (image_plane[j+1] - cam_pos)*fake_F
        ax.plot3D(*np.vstack((vec0, vec1)).T, linewidth=3, color='black')

    ax.set_xlim(0, 25)
    ax.set_ylim(-10, 15)
    ax.set_zlim(-5, 5)
    #ax.axis('auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if ax1 is not None and lines_2ds is not None:
        draw2d(ax1, lines_2ds, base_pts)
        
    return res_dict

def horiz_dist_from_proto_seg(l0_raw, l1_raw):
    # calc distance in recorded image
    # fix two angles and a point        
    l0 = np.array([l0_raw[0], h-l0_raw[1], l0_raw[2], h-l0_raw[3]])
    l1 = np.array([l1_raw[0], h-l1_raw[1], l1_raw[2], h-l1_raw[3]])

    # and the distance of intersections
    p0, p1 = l0[:2], l0[2:]
    s = p0[1] / (p0[1] - p1[1])
    intersect0 = p0 + s*(p1-p0)
    
    p0, p1 = l1[:2], l1[2:]
    s = p0[1] / (p0[1] - p1[1])
    intersect1 = p0 + s*(p1-p0)
    return np.abs(intersect0 - intersect1)[0]

# calculate horizontal distance between lanes
def horizontal_distance(pv0, l1):
    raise
    pvb = line_intersection(l1, [pv0, [pv0[0]+100, pv0[1]]])
    return (pvb - pv0)[0]

def angled_distance(pv0, l1, alpha):
    pvb = angled_intersect(pv0, l1, alpha)
    return np.linalg.norm((pvb - pv0))

def angled_intersect(pv0, l1, alpha):
    pvb = line_intersection(l1, [pv0, [pv0[0]+100, pv0[1]+100*alpha]])
    return pvb

def vertical_distance(pv0, l1):
    pvb = line_intersection(l1, [pv0, [pv0[0], pv0[1]+100]])
    return (pvb - pv0)[1]


def project_lines(_params, _lines, w=1280, h=720):
    cam_pos = _params['cam_pos']
    r = _params['r']
    fov = _params['fov']
    
    F = _params.get('F', 1)
    lf = np.tan(fov / 180 * np.pi / 2) * F
    
    lanes_rot = (_lines - cam_pos).dot(r)
    pts3d = lanes_rot.reshape(-1, 3)
    pts2d = pts3d / pts3d[:, 1, np.newaxis]
    lines_flat = pts2d.reshape(_lines.shape)
    lines_2d = lines_flat[..., [0,2]]
    
    lines_2d = lines_2d / (2*lf) * w
    lines_2d += (w//2, h//2)
    return lines_2d

def ray_cast_point(pt2d, cam_pos, image_plane3d, z_off=0.1, axis=2, w=1280, h=720):
    # draw 2d point on 3d image plane.
    # what is the ratio on the image that the point is moved?
    # go same ratios on 3d image plane.
    # points in image plane are 0-3 clockwise from bottom left.
    # z_off is the height above the ground. in case of feet, we intersect w/ 
    # the ankle which is an approx 10cm above ground.
    xd = pt2d[0] / w
    yd = pt2d[1] / h

    # bottom left start point + x/y-ratios times the direction vectors
    pt2d_3dplane = (1-xd-yd)*image_plane3d[2] + yd*image_plane3d[3] + xd*image_plane3d[1]

    l0 = cam_pos
    l = pt2d_3dplane - cam_pos
    
    p0 = [0, 0, 0]
    n = [0, 0, 0]
    p0[axis] = z_off
    n[axis] = 1
    d = (l0 - p0).dot(n) / (l.dot(n))
    return l0 - l*d

def knee_angle_from_lines(_lines_2d, limb_ids=[6,7]):
    # calc knee angle in resulting projection
    # line to body-part map:
    # 6: ankle-knee (left)
    # 7: knee-hip (left)
    ankle = _lines_2d[limb_ids[0]][0]
    knee = _lines_2d[limb_ids[0]][1]
    hip = _lines_2d[limb_ids[1]][1]
    calf = knee - ankle
    thigh = knee - hip
    knee_ang_disp = np.arccos(calf.dot(thigh) / (np.linalg.norm(thigh) * np.linalg.norm(calf)))
    return knee, np.degrees(knee_ang_disp)


def get_timesteps_from_curve(_Xfoot, _frames, band_width=9):
    # sliding min of size bandwidth
    # bandwidth: max of 200 spm = 3.333 step/s ~> 9 frames/step
    M = len(_Xfoot)
    ms = []
    for i in range(0, M-(band_width+1)):
        m = np.argmin(_Xfoot[i:i+band_width])
        ms.append(m + i)
    cnt = Counter(ms)
    cand = np.array(cnt.most_common(int(M/band_width)))
    vlines = cand[cand[:, 1] == band_width]
    # translate these minima to the actual frame where the athlete appeared
    vlines = np.array([[_frames[k], v] for k, v in vlines])
    return vlines

def vp_to_azim_elev_roll(_df, target_v0, v_name='v0', plot_it=False, group_by='azim'):
    target_line_v0 = np.vstack((target_v0, [w/2,h/2]))
    
    target_line = target_line_v0
    df_sel = _df

    g = df_sel.groupby([group_by, 'fov', 'F', 'dist']).groups.items()

    if plot_it:
        N = len(g)
        fig, axs = plt.subplots(nrows=N+1, figsize=(5, N*5))
        ax = axs[0]

    azim_elev_pairs = []
    for i, (param, idc) in enumerate(g):
        df_subsel = df_sel.loc[idc]
        v0 = np.vstack(df_subsel[v_name])

        if plot_it:
            ax.scatter(*v0.T)
            ax.plot(*v0.T, c='lightblue')

        # get the base line of these v0s connecting first and last
        # do these interesect at all? if so, get a better fit
        v_line = np.vstack((v0[0], v0[-1]))
        inters = line_intersection(target_line, v_line)
        s0 = intersection_ratio(v_line, inters)
        s0b = intersection_ratio(target_line, inters)
        
        # actually if the s0b overshoots, that's fine!
        if (0<=s0<=1):# and 0<=s0b<=1:
            # intersect and interpolate, which elev/azims work.
            # check them against the tool, whether those would actually do it.
            # then, zoom in! create larger grid and check what the deviation from straight
            # lines looks like in those smaller cases.

            # find the indices that are closests to the intersection point
            # determine which values on this iso-line are closest
            inds = np.argsort(np.sum((v0 - inters)**2, 1))
            j = 0
            s1 = -1
            while s1<0 or s1>1:
                # due to the curvature of the actual iso-line of a certain azim/elev
                # that we want to fit, there might be a situation, where the closest
                # points aren't the correct ones. if so, shift the search window
                chose_inds = [j, j+1]
                v0_sub = v0[chose_inds]
                # take the line-segment between the closest points and interpolate
                inters2 = line_intersection(target_line, v0_sub)
                s1 = intersection_ratio(v0_sub, inters2)
                j+=1
                if j+1 >= len(v0):
                    break
            dat_pts = np.array(df_subsel[['azim', 'elev', 'roll']].iloc[chose_inds])
            dat_res = (1-s1)*dat_pts[0] + s1*dat_pts[1]
            azim_elev_pairs.append(dat_res)

            if plot_it:
                ax.plot(*v_line.T, linewidth=4)

                ax2 = axs[i+1]
                ax2.scatter(*v0.T)
                ax2.scatter(*inters)
                ax2.scatter(*inters2, c='green')
                ax2.scatter(*v0_sub.T, c='red', alpha=.8)
                ax2.set_ylabel('%.2f %.2f'%(s0, s0b))

                ax2.plot(*v_line.T)
                xlim = ax2.get_xlim()
                ax2.scatter(*target_line[0], color='darkblue')
                ax2.plot(*target_line.T, c='gray', linewidth=2, alpha=.7)
                xlim = ax2.set_xlim(xlim)
                

    if plot_it:    
        ax.plot((0, 1280), (0, 0), c='black')
        ax.plot((0, 0), (720, 0), c='black')
        ax.plot((1280, 1280), (720, 0), c='black')
        ax.plot((0, 1280), (720, 720), c='black')

        ax.plot(*target_line.T, c='gray', linewidth=2, alpha=.7)
        ax.scatter(*target_v0, color='darkblue')
        ax.set_xlim(-10000, 10000)
        ax.set_ylim(0, 3500)

    if len(azim_elev_pairs) != 0:
        return np.vstack(azim_elev_pairs)

def proj_proto_to_points(_v1, _proj0, _proj1, _proto0, _proto1):
    # construct helper points on new lines
    # where the inner most projected line crosses the right image border
    i0 = np.array(line_intersection(_proj0, [[w, 0], [w, h]]))
    
    # use the vanishing point. this is per definition the location, where the lenghts
    # are the same. trying it by ratio caused problems with clipping!
    i1 = line_intersection([_v1, i0], _proj1)
    
    # get the proto intersection when we would y-shift the grid
    p0 = np.array(line_intersection(_proto0, [i0, i1]))
    p1 = np.array(line_intersection(_proto1, [i0, i1]))

    return i0, i1, p0, p1

def ray_cast_dist(i0, i1, p0, p1, cam_pos, rd):
    rc_i0 = ray_cast_point(i0, cam_pos, rd['image_plane'], z_off=0)
    rc_i1 = ray_cast_point(i1, cam_pos, rd['image_plane'], z_off=0)
    rc_p0 = ray_cast_point(p0, cam_pos, rd['image_plane'], z_off=0)
    rc_p1 = ray_cast_point(p1, cam_pos, rd['image_plane'], z_off=0)
    dist_p0_p1 = np.linalg.norm(rc_p0 - rc_p1)
    dist_i0_i1 = np.linalg.norm(rc_i0 - rc_i1)
    return dist_i0_i1 / dist_p0_p1

def azim_from_v0_proto_no_shift(df, proto_l0, proto_l1, v0, k=None, proj_indA=0, proj_indB=2,
                       skel_store_dict={}, debug=False):
    
    proto0 = np.array([proto_l0[0], h-proto_l0[1], proto_l0[2], h-proto_l0[3]])
    proto1 = np.array([proto_l1[0], h-proto_l1[1], proto_l1[2], h-proto_l1[3]])

    az_el = vp_to_azim_elev_roll(df, v0)
        
    if az_el is None:
        if debug:
            print('== azel empty ==')
        return None

    rds = {}
    scale_infos = defaultdict(list)

    for ae_i, ae in enumerate(az_el):
        try:
            # adapt view-target, s.t. the lane cross lays exactly at the proto-line cross.
            dist = 140
            view_target = [0, 0, 0]
            broken = False

            # == 1. adapt for lane width
            # project 
            dist_fact = 2
            # iterate a few more times, so as to minimize the offsets
            for l_iter in range(4):
                cam_pos = view_target - azim_to_vec(ae[0], ae[1], degrees=True)*dist
                rd = fov_from_azelro(v0, ae, camx=cam_pos[0], camy=cam_pos[1], camz=cam_pos[2])
                proj_lines = rd['lines_2ds'][1][::2]
                if len(proj_lines[0]) == 1:
                    #print('break 1', l_iter)
                    broken = True
                    break
                #print(proj_lines)

                proj0 = proj_lines[proj_indA]
                proj1 = proj_lines[proj_indB]

                if len(proj1) == 1:
                    broken = True
                    break
                i0, i1, p0, p1 = proj_proto_to_points(rd['v1'], proj0, proj1, proto0, proto1)

                dist_off = ray_cast_dist(i0, i1, p0, p1, cam_pos, rd)
                dist = dist * dist_off

                cam_pos = view_target - azim_to_vec(ae[0], ae[1], degrees=True)*dist
                #rd2 = fov_from_azelro(v0, ae, camx=cam_pos[0], camy=cam_pos[1], camz=cam_pos[2])
                rd2 = compute_projection(*cam_pos, azim=ae[0], elev=ae[1], fov=rd['fov'], F=rd['F'], roll=ae[2])

                proj_lines2 = rd2['lines_2ds'][1][::2]
                if len(proj_lines2[0]) == 1:
                    #print("broken2", ae_i)
                    broken = True
                    break

                proj0b = proj_lines2[proj_indA]
                proj1b = proj_lines2[proj_indB]

                i3 = line_intersection([i0, i1], proj0b)
                rc_i3 = ray_cast_point(i3, cam_pos, rd2['image_plane'], z_off=0)
                rc_p0 = ray_cast_point(p0, cam_pos, rd2['image_plane'], z_off=0)
                dist_i3_p0 = np.linalg.norm(rc_i3 - rc_p0) * dist_off

                #i0b, i1b, p0b, p1b = proj_proto_to_points(proj0b, proj1b, proto0, proto1)
                #dist_off_b = ray_cast_dist(i0b, i1b, p0b, p1b)

                # distance before and after move!!
                if i3[0] > p0[0]:
                    view_target[1] += dist_i3_p0
                else:
                    view_target[1] -= dist_i3_p0

                # == 3. do not move by x for now. do this later with minimizing the
                #    recostruction error of all lines.

            if broken:
                #print('broken!!', ae_i)
                continue
            # == 4. update one last time
            cam_pos = view_target - azim_to_vec(ae[0], ae[1], degrees=True)*dist
            #rd = fov_from_azelro(v0, ae, camx=cam_pos[0], camy=cam_pos[1], camz=cam_pos[2])
            fov = rd['fov']
            F = rd['F']
            rd = compute_projection(*cam_pos, azim=ae[0], elev=ae[1], fov=rd['fov'], F=rd['F'], roll=ae[2])

            rd['view_target'] = view_target
            rd['dist_fact'] = dist_fact
            rd['dist'] = dist
            rd['fov'] = fov
            rd['F'] = F

            # determine projected athlete height vs.
            # 0. get the location at which to project the athlete. intersection of foot and groundplane.
            if k is not None and len(skel_store_dict) > 0 and \
                k in skel_store_dict['frame_to_athid_anchor']:
                frame_vline = skel_store_dict['frame_to_athid_anchor'][k]
                for row in frame_vline:
                    _ath_id, anch, skel_arr_id = row
                    letes_array2d = skel_store_dict['ath_to_skel2d'][_ath_id]
                    pos2d = letes_array2d[skel_arr_id].copy()
                    pos2d[:, 1] *= -1
                    pos2d[:, 1] += h
                    pt2d = pos2d[anch]
                    azim = rd
                    pt3d_ankle = ray_cast_point(pt2d, azim['cam_pos'], azim['image_plane'])
                    #foot straight up lines and their projection back to 2d
                    ank_straight_line = np.array([[pt3d_ankle, pt3d_ankle + [0, 0, 1.66]]])
                    pl = project_lines(azim, ank_straight_line)
                    proj_height = np.linalg.norm(pl[0][0] - pl[0][1])
                    # calc the 2d stretch of the athlete
                    pelv_neck_height_2d = np.linalg.norm(pos2d[anch] - pos2d[NECK_IND])
                    scale_infos[ae_i].append({'ath': _ath_id, 'proj': proj_height, 
                                              'actual': pelv_neck_height_2d, 'pt3d': pt3d_ankle,
                                              'project_line': pl})
            rd['azim'] = ae[0]
            rd['elev'] = ae[1]
            rd['roll'] = ae[2]
            rds[ae_i] = rd
        except Exception as e:
            raise e
            print('==ERR==')
            print(e)
            print(ae_i, ae)
            print('/==ERR==')
            continue

    return {'rds': rds, 'v0measured': v0,
            'proto_l0': proto_l0, 'proto_l1': proto_l1, 'scale_infos': scale_infos}
    
def fov_from_azelro(v0target, azel, fov=1, F=1, camx=-10, camy=-4, camz=2, w=1280, h=720):
    v0_dist = 100
    fov_new = fov
    i = 0
    while v0_dist>10 and i<10:
        cent = np.array([w//2, h//2])
        res_dict = compute_projection(camx, camy, camz, azim=azel[0], elev=azel[1], 
                       fov=fov_new, F=F, roll=azel[2])
        v0 = res_dict['v0']
        v1 = res_dict['v1']
        d0 = np.sqrt(np.sum((v0target - cent)**2))
        d1 = np.sqrt(np.sum((v0 - cent)**2))
        v0_dist = np.sqrt(np.sum((v0target - v0)**2)) 
        d = d0/d1
        # TODO: make this more elegant!!!!
        # DNGR! this isn't quite right!!!
        fov_new = fov_new/d
        i+=1
        # if we spin here to much, skip! take best guess.
        #print(dist, d)
    #print('--')
    
    params = {'camx': camx, 
              'camy': camy, 
              'camz': camz, 
              'azim': azel[0],
              'elev': azel[1],
              'fov': fov_new,
              'F': F,
              'roll': azel[2]
             }
              
    res_dict_new = compute_projection(**params)
    res_dict_new.update(params)
    return res_dict_new

def smooth(arr, win_width=5, smooth_fun=np.mean):
    arr_new = []
    for i in range(0, len(arr)-win_width):
        arr_new.append(smooth_fun(arr[i:i+win_width], 0))
    return np.array(arr_new)

def roll_from_vps(_v0, _v1, roll=0):
    # compute the necessary camera roll to make vps horizontal
    d = _v0 - _v1
    roll_beta = np.arctan(np.abs(d[0] / d[1]))
    roll_alpha = np.pi/2 - roll_beta
    
    if _v0[1] > _v1[1]:
        roll_alpha = roll_alpha*-1
        
    roll_new = roll - roll_alpha / np.pi * 180
    return roll_new

def create_look_up_grid(ranges=None, no_roll=False):
    if ranges == None:
        ranges = {'azim': list(np.arange(-89, -1, 2)),
          'elev': list(np.arange(-45, 1, 1)),
          'dist': [1],
          'fov': [20],#list(np.arange(1, 10, 2)),
          'F': [1]#list(np.arange(1, 10, 2))
         }

    N_opts = np.prod([len(v) for v in ranges.values()])
    all_opts = itertools.product(*ranges.values())
    res = []
    for opts in tqdm(all_opts, total=N_opts):
        params = dict(zip(ranges.keys(), opts))
        # always point camera at somewhat center of the scene
        view_target = np.array([7, 0, 0])

        cam_vec = view_target - azim_to_vec(params['azim'], params['elev'], degrees=True) * params['dist']

        params['camx'] = cam_vec[0]
        params['camy'] = cam_vec[1]
        params['camz'] = cam_vec[2]
        params_in = params.copy()
        del params_in['dist']
        roll = params_in.get('roll', 0)

        try:
            res_dict_pre = compute_projection(_lanes=lanes_new, **params_in)
        except Exception as e:
            print(e)
            raise e
            continue

        # now do the same again with the proper rotation to make it horizontal
        if no_roll:
            res_dict = res_dict_pre
            params['roll'] = 0
        else:
            roll = roll_from_vps(res_dict_pre['v0'], res_dict_pre['v1'])
            params['roll'] = roll
            params_in['roll'] = roll
            res_dict = compute_projection(**params_in)

        #if res_dict['v1'] is not None:
        #    lines_2d = res_dict['lines_2d']
        #    # how much are the vertical lines tilted in the view?
        #    xdiff = np.mean(lines_2d[n_lanes:-n_orth][:, 0, 0] - lines_2d[n_lanes:-n_orth][:, 1, 0])
        #    res_dict['xdiff'] = xdiff
#
        params.update(res_dict)
        res.append(params)

    return DataFrame(res)

def angle_between(center, p0, p1):
    u = center - p0
    v = center - p1
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    
    dot = u.dot(v)
    ang = np.degrees(np.arccos(np.abs(dot)))
    if dot < 0:
        ang = 180 - ang
    return ang

def clip_segment(_p):
    p_good = _p[:, 1] > 0
    
    where = np.where(p_good==False)[0]
    
    p_clip = _p[p_good]
    if len(where) > 0:
        # first occurrence in chain that needs to be clipped.
        to_clip = where[0]
        p0 = _p[to_clip-1]
        p1 = _p[to_clip]
        pvec = p1 - p0
        
        if pvec[1] != 0:
            # dont clip to exactly on the boundary for numerical reasons!
            # clip plane is y=.1
            t = (-p0[1] + 0.1) / pvec[1]
            p_new = p0 + t*(p1 - p0)

            # it can happen that parts of the curve leave the clipping
            # area and the curve comes back in after.
            # in that case: just discard the missing segments.
            if len(where) + where[0] >= len(p_good):
                p_clip = np.vstack((p_clip, p_new))
    return p_clip

def clip_rots(_lanes_rot):
    clipped = []
    for pts in _lanes_rot:
        pts_clipped = []
        for p in pts:
            pts_clipped.append(clip_segment(p))
        clipped.append(pts_clipped)
    return clipped

def interpolate(az_cands, cam_id):
    main = int(cam_id)
    r = cam_id - main
    
    if not main in az_cands:
        return None
        
    if r == 0 or not (main+1) in az_cands:
        return az_cands[main]
    
    cand0 = az_cands[main]
    cand1 = az_cands[main+1]
    
    cand_mix = {}
    for k, v in cand0.items():
        v2 = cand1[k]
        if k == 'view_target':
            v = np.array(v)
            v2 = np.array(v2)
            
        try:
            cand_mix[k] = v + r * (v2 - v)
        except Exception as e:
            continue
            
    rd_mix = compute_projection(*cand_mix['cam_pos'], **cand_mix)
    cand_mix['lines_2ds'] = rd_mix['lines_2ds']
    #cand_mix['view_target'] = rd_mix['view_target']
    
    return cand_mix

def interpolate_params(params_list, target_value, param_name='cam_posz'):
    '''
    increase means whether the value in question increas with a higher index.
    for cam_pos, increase=1 because a higher index up higher.
    '''
    if param_name == 'cam_posz':
        param_name = 'cam_pos'
        param_ind=2
        increase=1
    elif param_name == 'cam_posy':
        param_name = 'cam_pos'
        param_ind=1
        increase=1
    elif param_name == 'v1x':
        param_name = 'v1'
        param_ind=0
        increase=-1
    elif param_name == 'v2':
        param_ind=1
        increase=1
    elif param_name == 'elev':
        param_ind=None
        increase=1
    else:
        raise NotImplementedError
    
    param_keys, param_values = zip(*list(params_list.items()))
    params = np.vstack([a[param_name] for a in param_values])
    if param_ind is not None:
        params = params[:, param_ind]
        
    cam_id_min = np.nanargmin(np.abs(params - target_value))
    
    cp0 = params_list[param_keys[cam_id_min]][param_name]
    cp1 = params_list[param_keys[cam_id_min + 1]][param_name]
    if param_ind is not None:
        s = (target_value - cp0[param_ind]) / (cp1[param_ind] - cp0[param_ind])
    else:
        s = (target_value - cp0) / (cp1 - cp0)
    if s < 0 or s>1:
        s=0
    
    best = interpolate(params_list, cam_id_min + s)
    
    return best, cam_id_min+s

def shift_x(az_pair, x_off):
    azim = az_pair['azim']
    elev = az_pair['elev']
    fov = az_pair['fov']
    F = 1
    rot = az_pair['roll']
    # shift again
    cam_vec = az_pair['cam_pos'].copy()
    cam_vec[0] += x_off

    shifted = compute_projection(camx=cam_vec[0], camy=cam_vec[1], camz=cam_vec[2],
                     azim=azim, elev=elev, fov=fov, F=F, roll=rot, _lanes=lanes_new)
    same_params = ['dist', 'azim', 'elev', 'roll', 'fov', 'F']
    
    target = az_pair['view_target']
    target[0] += x_off
    shifted['view_target'] = target
    for p in same_params:
        shifted[p] = az_pair[p]
        
    return shifted

# == Test clipping ==
# fig = plt.figure()
# ax_clip = fig.add_subplot(111, projection='3d')

# for pts in res_dict['lanes_rots']:
#     for p in pts:
#         p_clip = clip_segment(p)
#         ax_clip.plot3D(*p_clip.T)
        
# xs = np.array([0, 25])
# ys = np.array([0, 10])
# zs = np.array([0, 0])

# n = 2
# x = np.linspace(0, 150, n)
# y = np.linspace(-10, 10, n)
# X, Y = np.meshgrid(x, y)
# Z = X*0

# ax_clip.plot_surface(X, Z, Y, rstride=12, cstride=12, color=(.25,.25,.25,.5))

# ax_clip.set_xlabel('x')
# ax_clip.set_ylabel('y')
# ax_clip.set_zlabel('z')

def create_projection_matrix_from_azim_pair(az_pair, width=1280, height=720):
    params = az_pair
    cx = width // 2
    cy = height // 2

    cam_pos = params['cam_pos']
    r = params['r']
    fov = params['fov']
    F = params['F']
    lf = np.tan(fov / 180 * np.pi / 2) * F
    f = w / (2*lf)

    switch = np.array([[1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]])

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    
    t = -r.T.dot(cam_pos[:, np.newaxis])
    Rt = np.hstack((r.T, t))
    P = K @ switch @ Rt

    return P

def project_3d_to_2d(_P, pts3d_in):
    pts3d_in_pad = np.hstack((pts3d_in, np.ones((pts3d_in.shape))))[..., :4]
    pts3d = (_P @ pts3d_in_pad.T).T
    pts2d = pts3d / pts3d[..., 2, np.newaxis]
    pts2d = pts2d[..., :2]
    return pts2d


def min_distance_to_line(l0, l1, x0):
    # find minimum between point, line with fraction t.
    t = -(l1 - l0).dot(l0 - x0) / np.linalg.norm(l1 - l0)**2
    d = np.linalg.norm(l0 + (l1 - l0) * t)
    return t, d

def find_exact_distance(t, l0, l1, x0, dt):
    # the line is determined by l0, l1
    d = np.linalg.norm((l0 + (l1 - l0) * t[0]) - x0)
    l = np.abs(d - dt)
    return l

def find_points_with_distance_to_line(l0, l1, x0, dt):
    '''
    function to determine location of ray-casting.
    we know the parent location of a certain limb and its length
    then, determine the end-point as the limb as the intersection
    of the ray and the correct distance.
    == params
    # l0:
    '''

    res = minimize(find_exact_distance, x0=1, tol=1e-8, method='Nelder-Mead', args=(l0, l1, x0, dt, ))
    t1 = res.x[0]
    p1 = l0 + (l1 - l0) * t1
    
    t_min, d_min = min_distance_to_line(l0, l1, x0)
    
    t_diff = t_min - t1
    t2 = t_min + t_diff

    p2 = l0 + (l1 - l0)*t2
    
    return p1, t1, p2, t2

def get_lane1_marker(rd):
    lines2d = rd['lines_2ds']
    l = lines2d[0][0][-1].copy()
    l[1] = h - l[1]
    return l

def groundtruth_to_b(groundtruthdict):
    # other params that b could be enhanced with!?
    # 'view_target', 'dist'
    
    campos = groundtruthdict['campos']
    camrot = groundtruthdict['camrot']
    fov = groundtruthdict['fov']
    el, az, ro = camrot
    b = compute_projection(*campos, azim=az, elev=el, fov=fov, F=1,roll=ro)
    view_target = ray_cast_point([w//2, h//2], campos, b['image_plane'], z_off=0)
    
    # add some other commonly requested data
    b['azim'] = az
    b['elev'] = el
    b['roll'] = ro
    b['fov'] = fov
    b['F'] = 1
    b['view_target'] = view_target
    b['dist'] = np.linalg.norm(view_target-campos)
    return b
