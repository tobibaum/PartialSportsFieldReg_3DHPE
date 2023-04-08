import time
from tqdm import tqdm
import sys
import numpy as np
import cv2
import json
import pandas
from pytube import YouTube
import pickle
from sklearn.metrics import pairwise_distances

from tictoc.tictoc import TicTocer
import yt_utils

from matplotlib import pyplot as plt

def intersection_ratio(line, pt):
    # from a segment and a point that lies on that line,
    # find intersection point on the segment
    a, b = line
    s = ((pt-a) / (b-a))[0]
    return s

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

def get_median_vanish(_plines, _tt=TicTocer()):
    vs = []
    N = len(_plines)
    for i in range(N):
        l0 = _plines[i][0]
        for j in range(i+1, N-1):
            l1 = _plines[j][0]
            _tt.tic('van_inter')
            li = line_intersection(l0, l1)
            _tt.toc('van_inter')
            if li is not None:
                vs.append(li)
    vs = np.array(vs).T
    return np.median(vs, 1)

def draw_segs(img, _x0, _x1, col=(0,128,128), w=3):
    cv2.line(img, _x0.astype(int), _x1.astype(int), col, w)

def determine_vanish_points(plines, nplines, _tt=TicTocer()):
    # determine the main vanishing point from the track lanes
    _tt.tic('get v0')
    v0 = get_median_vanish(plines, _tt=_tt)
    x1 = v0
    _tt.toc('get v0')

    # compute intersection of horizontal line (from first vanishing point)
    # w/ potential orthogonal lines in scene
    if len(nplines) == 0:
        return v0, None
    
    inters = []
    _tt.tic('get v1')
    for line in nplines:
        _tt.tic('inter')
        li = line_intersection((v0, [v0[0]*100, v0[1]]), line[0])
        _tt.toc('inter')
        if li is not None:
            inters.append(li)
    _tt.toc('get v1')
    
    if len(inters) == 0:
        #print('no intersections')
        return v0, None
    
    v1 = np.mean(inters, 0)
    if v1[0] * v0[0] > 0:
        # both vanishing points are either left or right of the central image
        # line. unreasonable!!
        return v0, None
    return v0, v1
    
def draw_orthogonals(img, v1, start_line, dist=100):
    img_orth = img

    p0 = start_line[:2]
    p1 = start_line[2:]
    vec_line = p1-p0
    line_len = np.linalg.norm(vec_line)
    #for angle in range(-180, 180, 1):
    #    vec = np.array((np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)))
    for d in range(0, int(line_len), dist):
        y0 = v1
        y1 = p0+vec_line/line_len*d
        vec = y1 - y0
        y2 = y0 + vec*2
        #plot_seg(y0, y1, c='g')
        draw_segs(img_orth, y0, y1, col=(80,80,80), w=2)
        draw_segs(img_orth, y1, y2, col=(80,80,80), w=2)

    return img_orth

from sklearn.cluster import MeanShift
from collections import Counter

def get_angle(l):
    return np.arctan((l[2] - l[0]) / (l[3] - l[1]))

def get_lines_for_img(img, _tt=TicTocer(), arg_dict={}, pline_range=[-10, 100]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #kernel_size = 1
    kernel_size = arg_dict.get('kernel_size', 1)
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    #Second, process edge detection use Canny.
    low_threshold = arg_dict.get('low_threshold', 20)
    high_threshold = arg_dict.get('high_threshold', 210)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #Then, use HoughLinesP to get the lines. You can adjust the parameters for better performance.
    rho = arg_dict.get('rho', 1)  # distance resolution in pixels of the Hough grid
    theta = np.pi / 1440 # angular resolution in radians of the Hough grid
    threshold = arg_dict.get('threshold', 10)  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = arg_dict.get('min_line_length', 120)  # minimum number of pixels making up a line
    max_line_gap = arg_dict.get('max_line_gap', 10)  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    # get the main directions of the lines.
    all_angles = get_angle(lines[:, 0, :].T) / np.pi * 180
    range_check = (all_angles < pline_range[0]) + (all_angles > pline_range[1])
    all_angles[np.where(range_check)] = -180
    
    # meanshift with a bandwith of 5 degrees
    ANGLE_PLAY = 5
    clustering = MeanShift(bandwidth=ANGLE_PLAY).fit_predict(all_angles[:, np.newaxis])
    cnt = Counter(clustering)
    main_dir_cnt = cnt.most_common(2)

    pick_angles = all_angles[clustering == main_dir_cnt[0][0]]
    if pick_angles[0] == -180:
        main_lines = lines[clustering == main_dir_cnt[1][0]]
    else:
        main_lines = lines[clustering == main_dir_cnt[0][0]]
        
    if len(cnt) > 1:
        second_lines = lines[clustering == main_dir_cnt[1][0]]
    else:
        second_lines = []

    return main_lines, second_lines

def draw_lines_on_img(img, _plines, _nplines):
    img_out = img
    
    for lines, col in zip([_plines, _nplines], [(10, 212, 30), (210, 20, 40)]):
        for l in lines:
            p0 = l[0][:2]
            p1 = l[0][2:]
            cv2.line(img_out, p0, p1, col, 2)
    
    return img_out

def line_distance(_l0, _l1):
    # distance of points p3/p4 from line between p1-p2
    p1, p2 = _l0[:2], _l0[2:]
    p3, p4 = _l1[:2], _l1[2:]
    
    d = []
    for p in [p3, p4]:
        d.append(np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1))
        
    # the distances between start and enpoint of the segments are a little
    # different, cuz its not perfectly parallel. take average
    return np.mean(d)

def get_main_segments_from_lines(_lines, clustering_bandwidth=10, _tt=TicTocer()):
    # cluster line segments down to main lines.
    # assumption that these are the lanes. 
    # parameters: clustering bandwidth
    # approximately half the distance in pixels in the image between lanes

    _tt.tic('collect')
    pts = []
    for l in _lines:
        ang = get_angle(l)
        alph_l = np.pi/2-ang
        ca_l = np.cos(alph_l)
        sa_l = np.sin(alph_l)
        rot_l = np.array([[ca_l, -sa_l], [sa_l, ca_l]])

        seg = l[:2], l[2:]
        seg_r_l = np.array(seg).dot(rot_l)
        pt = seg_r_l[0, 1]
        pts.append(pt)

    pts = np.array(pts)
    _tt.toc('collect')
    
    _tt.tic('cluster')
    clusters = MeanShift(bandwidth=clustering_bandwidth).fit_predict(pts[:, np.newaxis])
    _tt.toc('cluster')
    
    _tt.tic('proto')
    _proto_segs = []
    _proto_pts = []
    for c in np.unique(clusters):
        mp = np.mean(pts[clusters == c])
        ml = np.mean(_lines[clusters==c], 0)
        _proto_segs.append(ml)
        _proto_pts.append(mp)

    order = np.argsort(_proto_pts)
    _tt.toc('proto')
    return np.array(_proto_segs)[order]

def seg_to_full_width(_img, _line):
    h, w, _ = _img.shape
    m0 = line_intersection(_line, [[0,0], [0,h]])
    m1 = line_intersection(_line, [[w,0], [w,h]])
    if m1[1] > h:
        m1 = line_intersection(_line, [[0, h], [w, h]])
    if m0[1] < 0:
        m0 = line_intersection(_line, [[0, 0], [w, 0]])
    return np.hstack((m0,m1))

# track lanes.
# match the segs to each other.
# backup idea: lukas-kanade on protoseg corners

def track_lanes(proto_segs):
    n_start = len(proto_segs[0])
    seg_id = n_start
    frame_to_proto_seg_to_id = {0: {k: k for k in range(n_start)}}
    frame_lane_id_to_proto_id = {0: {k: k for k in range(n_start)}}
    # initialize with all the first segments.
    active = dict(enumerate(proto_segs[0][:, 0]))

    max_distance = 150

    for k in range(1, len(proto_segs)):
        ps = proto_segs[k]

        keys, values = list(zip(*active.items()))

        p1 = np.vstack(values)
        p2 = ps[:, 0]

        dists = pairwise_distances(p1, p2)
        preferences = np.argsort(dists, 1)

        transitions = {}
        picked = set()
        for i, pref in enumerate(preferences):
            for p in pref:
                d = dists[i, p]
                # if we are too far off, reject!
                if p in picked or d > max_distance:
                    continue
                picked.add(p)
                transitions[i] = p
                break

        # are there any lanes we didn't assign yet? give them new ids.
        unpicked = set(range(len(p2))) - picked
        for p in unpicked:
            active[seg_id] = ps[p, 0]
            seg_id += 1

        for t, v in transitions.items():
            active[t] = ps[v, 0]

        frame_to_proto_seg_to_id[k] = {v: l for l, v in transitions.items()}
        frame_lane_id_to_proto_id[k] = transitions
    return frame_to_proto_seg_to_id, frame_lane_id_to_proto_id