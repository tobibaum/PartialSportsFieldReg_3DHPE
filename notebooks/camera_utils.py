import numpy as np
from scipy.spatial.transform import Rotation as R

def draw_base(ax1, base_pts):
    #lf = base_pts[0][0]
    #shift = np.array([w,h])/2
    for j, bp in enumerate(base_pts):
        if j >= len(base_pts)-1:
            j = -1
        bp_line = np.vstack((bp, base_pts[j+1]))#[:, [0,2]]
        #ax1.plot(*(bp_line/(lf*2)*w + shift).T, linewidth=2, color='black')
        ax1.plot(*bp_line.T, linewidth=2, color='black')

def draw2d(ax1, lines_2ds, base_pts=None, color=None, trans_fun=None, w=1280, h=720, **kwargs):
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
    
def draw_scene(res_dict, _lanes, ax=None, ax1=None, ax2=None, 
                w=1280, h=720, fake_F=1, **kwargs):
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

def rotation_from_view(view_dir, ignore_flip=False):
    azim, elev = azel_from_view_dir(view_dir)
    r_pan = R.from_euler('z', azim, degrees=True).as_matrix()
    r_tilt = R.from_euler('x', elev, degrees=True).as_matrix()
    r = r_pan.T.dot(r_tilt)
    # flip around horizontal axis! cuz images have an origin of (0, 720) in plt
    if not ignore_flip:
        r[:, 2] *= -1
    return r

def get_lanes():
    # Track parameters
    len_straight = 84.39
    radius_curve = 36.5
    distance_lanes = 1.22
    n_lanes = 8
    n_curve_segments = 200
    
    # Straight lane segments (homestretch and opposite)
    stad_lanes = []
    for i in range(n_lanes):
        xs = [-15, len_straight]
        # homestretch
        y = -i * distance_lanes
        stad_lanes.append(np.array([[xs[0], y, 0], [xs[1], y, 0]]))
        # opposite stretch
        y = 2 * radius_curve + i * distance_lanes
        stad_lanes.append(np.array([[xs[0], y, 0], [xs[1], y, 0]]))
    
    # Curved segments
    curves = []
    for i in range(n_lanes):
        for x0 in [0, len_straight]:
            r = radius_curve + i * distance_lanes
            y0 = r - i * distance_lanes
            theta = np.linspace(np.pi/2., 3*np.pi/2., n_curve_segments)
            sign = 1 if x0 == 0 else -1
            x = sign * r * np.cos(theta) + x0
            y = r * np.sin(theta) + y0
            z = np.zeros_like(theta)
            curves.append([x, y, z])
    curves = np.array(curves)
    
    # Finish lines
    finish_lanes = []
    y0 = -n_lanes * distance_lanes
    y1 = distance_lanes
    for xo in [-1, -.5, -.1, 0, .1, .5, 1]:
        xs = len_straight + xo
        finish_lanes.append(np.array([[xs, y0, 0], [xs, y1, 0]]))
    
    # Result
    return [curves.transpose([0, 2, 1]), np.array(stad_lanes), finish_lanes]


def line_intersection(_line1, _line2):
    if len(_line1) == 4:
        line1 = [[_line1[0], _line1[1]], [_line1[2], _line1[3]]]    
    else:
        line1 = _line1
        
    if len(_line2) == 4:
        line2 = [[_line2[0], _line2[1]], [_line2[2], _line2[3]]]
    else:
        line2 = _line2
        
    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        #print('lines do not intersect')
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

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

def azel_from_view_dir(v):
    c = np.sqrt(v[0]**2+v[1]**2)
    
    az_ = np.degrees(np.arctan(v[0]/v[1]))
    el_ = np.degrees(np.arctan(v[2]/c))
    return az_, el_

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

def compute_projection(camx, camy, camz, azim, elev, fov, F, _lanes,
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