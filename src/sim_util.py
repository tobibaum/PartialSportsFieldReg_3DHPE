from tictoc.tictoc import TicTocer

from projection_util import *

MAX_INT = 2**31-1

FEAT_KEYS = ['azim','elev','fov','rot','target_x','target_y','dist']
MAX_VAL = 250
CANNY_LOW = 100
CANNY_HIGH = 200

param_ranges = {'azim': (-90.,90), 
                'elev': (-20., -1),
                'fov': (1, 40),
                'rot': (-10., 10),
                'target_x': (-20, 20),
                'target_y': (-8, 0),
                'dist': (20, 80)
               }

# feature parameters
# how many radii to use.
#scale_down image before taking
sd = 4
r_collection = list(range(100, 10000, 1250)) + [-1]
deg_skip = 5
bin_width = 5
ro_max = int(np.sqrt((w//sd)**2 + (h//sd)**2))
ro_bins = range(0, 2*ro_max+1, bin_width)

from stadium_def import *
        
def draw_canvas_from_lines(_lines2d, _sd=sd, line_width=1):
    canvas = np.zeros((h//_sd, w//_sd))
    for line_coll in _lines2d:
        for _line in line_coll:
            line = _line.copy()
            line[:, 1] = h - line[:, 1]

            line /= _sd
            for i_line in range(len(line) - 1):
                s, e = line[i_line], line[i_line+1]
                if np.any(np.abs(s)>MAX_INT/2) or np.any(np.abs(e)>MAX_INT/2):
                    continue
                try:
                    cv2.line(canvas, s.astype(int), e.astype(int), 255, line_width)
                except Exception as excp:
                    print('ERR s,e', s, e)
                    print(excp)
    return canvas

# accumulate hough lines!

from collections import defaultdict, Counter
import time


def convert_canvas_to_lines(_canvas, plot_it=False):
    if plot_it:
        plt.figure()
    
    t = time.time()
    # hough accumulation for straight line
    xs, ys = np.where(_canvas>0)
    canvas_acc = np.zeros((180 // deg_skip + 1, len(ro_bins)-1))
    # for every segment on a line
    ro_raw = []
    for theta in range(0, 180, deg_skip): 
        ang = np.radians(theta)

        ro_fit = (xs*np.cos(ang) + ys*np.sin(ang)).astype(int)
        ro_raw.append(ro_fit.copy())
        ro_fit += ro_max
        ro_fit = ro_fit[ro_fit < 2*ro_max]
        ro_fit = ro_fit[ro_fit > 0]
        

        hist = np.histogram(ro_fit, bins=ro_bins)[0]
        canvas_acc[theta//deg_skip] = hist
        
#         print(theta, hist[ro_max])
        
        if theta==104 or theta==106 or True:
            max_inds = np.where([ro_fit == ro_max])[1]
            plt.plot(ys[max_inds], xs[max_inds])
#             print(theta, hist[ro_max])
#             print('max_inds', max_inds)
#             print(max_inds)
#             print(Counter(ro_fit))
#             print('hist', np.argmax(hist))
        
#         hist[
        #cnt = Counter(ro_fit)
        #inds, vals = list(zip(*list(cnt.items())))
        #canvas_acc[theta][list(inds)] = vals
    t = time.time() - t
    print('min/max', np.min(ro_raw), np.max(ro_raw), ro_max)
    print('took %.2fs'%t)
    if plot_it:
        plt.imshow(_canvas)
    return canvas_acc

def illustratate_canvas_and_lines(_canvas, _canvas_acc, thresh=110):
    # draw lines from accumulator

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    
    theta, ro = np.where(_canvas_acc>thresh)
    theta_raw = theta.copy()
    ro_raw = ro.copy()
    theta = theta * deg_skip
    ro *= bin_width
    ro -= ro_max
    
    blame = np.zeros(_canvas_acc.shape)

    for i, (t, r) in enumerate(zip(theta, ro)):
        ang = np.radians(t)
        ca = np.cos(ang)
        sa = np.sin(ang)

        X = r / sa
        Y = r / ca
        pts = np.array([[X, 0], [0, Y]])

        col = 'red'
        if t>90:
            pts = np.array([[-X, 2*Y], [0, Y]])
            col='blue'
            if pts[0][1] < 50:
                col='purple'
#                 print(t, r, X, Y)
                
        #blame[t, r] = col
        circ = plt.Circle((ro_raw[i], theta_raw[i]), 2, color=col, fill=False)
        axs[1].add_patch(circ)
        
        axs[0].plot(*np.vstack(pts).T, c=col)

    axs[0].imshow(_canvas)
    axs[1].imshow(_canvas_acc)

# voting for curved hough
# https://de.wikipedia.org/wiki/Hough-Transformation
# https://de.wikipedia.org/wiki/Kreissegment
def convert_canvas_to_hough(_canvas, tt=TicTocer(), plot_time=False):
#     print('canv-shape', _canvas.shape)
    xs, ys = np.where(_canvas>0)
    H = np.sqrt(xs**2 + ys**2) + 1e-6
    beta = np.arcsin(ys / H)
    
    ang_to_sa = {}
    ang_to_ca = {}
    filled_lookup = False
    
    r_to_canvas = []
    tt.tic('full')
    for r in tqdm(r_collection, disable=not plot_time):
        tt.tic('per')
        r2 = r**2
        canvas_curve = np.zeros((180 // deg_skip, len(ro_bins)-1))
        for theta in range(0, 180, deg_skip):
            tt.tic('theta')
            
            tt.tic('calc')
            ang = np.radians(theta)
            tt.tic('trig')
            if filled_lookup:
                tt.tic('lookup')
                # ha!! i use dynamic programming. who knew.
                sa = ang_to_sa[ang]
                ca = ang_to_ca[ang]
                tt.toc('lookup')
            else:
                tt.tic('recalc')
                sa = np.sin(ang - beta)
                ca = np.cos(ang - beta)
                ang_to_sa[ang] = sa
                ang_to_ca[ang] = ca
                tt.toc('recalc')                
            s = H * sa
            tt.toc('trig')
            if r == -1:
                h_seg = 0
            else:
                s[s**2 > r2] = r
                h_seg = r - np.sqrt(r2 - s**2)
            ro_fit = (ca - h_seg/H) * H
            tt.toc('calc')
            
            tt.tic('filt')
            ro_fit = ro_fit.astype(int) + ro_max
            ro_fit = ro_fit[ro_fit < 2*ro_max]
            ro_fit = ro_fit[ro_fit > 0]
            tt.toc('filt')

            tt.tic('count')
            # TODO: this could be sped up!!
            hist = np.histogram(ro_fit, bins=ro_bins)[0]
            canvas_curve[theta//deg_skip] = hist
            
#             cnt = Counter(ro_fit)
#             if len(cnt) > 0:
#                 inds, vals = list(zip(*list(cnt.items())))
#                 canvas_curve[theta][list(inds)] = vals

            tt.toc('count')
            tt.toc('theta')
        r_to_canvas.append(canvas_curve)
        tt.toc('per')
        filled_lookup = True

    tt.toc('full')
    r_to_canvas = np.array(r_to_canvas)
    
    if plot_time:
        tt.print_timing_infos()
        
    return r_to_canvas.transpose([2,1,0])

def illustratate_canvas_and_curves(_canvas, _canvas_curve, _r, thresh=150, thresh_top_perc=-1):
    if _r == -1:
        return illustratate_canvas_and_lines(_canvas, _canvas_curve, thresh=thresh)
        
    fig, ax = plt.subplots()
    ax.imshow(_canvas)
    
    if thresh_top_perc != -1:
        # use the top percentage of lines.
        dats = _canvas_curve.ravel()
        dats = dats[dats != 0]
        thresh = np.quantile(dats, 1 - thresh_top_perc)

    # draw again for curves!
    thetas, ros = np.where(_canvas_curve>thresh)
    thetas = thetas * deg_skip
    ros *= bin_width
    ros -= ro_max
    
    for t, ro in zip(thetas, ros):
        ang = np.radians(t)
        ca = np.cos(ang)
        sa = np.sin(ang)

        # plot circle of radius r around center.
        # distance of center from corner: ro+r
        cx = (ro+_r) * sa
        cy = (ro+_r) * ca

        circ = plt.Circle((cx, cy), _r, color='r', fill=False)
        ax.add_patch(circ)

    #plt.xlim(-100, 750)
    #plt.ylim(400, -100)
    
def create_feature_from_params(azim=-70., elev=-4., dist=80., fov=5, rot=-2.75, target_x=-3, target_y=2, return_res=False):

    view_target = np.array([target_x, target_y, 0.])
    azel_vec = azim_to_vec(azim, elev, degrees=True)
    cam_vec = view_target - azel_vec * dist

    res_dict = compute_projection(camx=cam_vec[0], camy=cam_vec[1], 
                                  camz=cam_vec[2], azim=azim, elev=elev, 
                                  fov=fov, F=1, roll=rot, no_vanish=True, 
                                  _lanes=lanes_new)
    canvas = draw_canvas_from_lines(res_dict['lines_2ds'])

    if return_res:
        return convert_canvas_to_hough(canvas), canvas, res_dict
    else:
        return convert_canvas_to_hough(canvas), canvas
    
def hough_feat_to_img(_feat):
    return np.vstack(_feat.transpose([2,1,0])).T

def img_to_edges(img, _sd=sd):
    # == draw edges ==
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Second, process edge detection use Canny.

    gray = cv2.resize(gray, (w//_sd, h//_sd))
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return edges

def edges_to_hough(edges):
    max_val = MAX_VAL
    hough_feat = convert_canvas_to_hough(edges)
    hough_feat[hough_feat>max_val] = max_val
    hough_feat /= max_val
    return hough_feat


