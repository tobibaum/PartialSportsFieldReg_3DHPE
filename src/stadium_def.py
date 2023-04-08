import numpy as np

do_orthogonal = False
do_finish = True

# standard track dimensions.
# https://www.dimensions.com/element/track-and-field-400m-running-track
len_straight = 84.39
radius_curve = 36.5
distance_lanes = 1.22
n_lanes = 8
n_curve_segments = 200

# ====== stadium ======
# looking down the homestretch, x=straight, y=left
# the origin is the inner left curve of lane1
stad_lanes = []
colors = []
for i in range(n_lanes):
    xs = [-15, len_straight]
    # lanes on the homestretch
    y = -i*distance_lanes
    lane = np.array([[xs[0], y, 0], [xs[1], y, 0]])
    stad_lanes.append(lane)
    colors.append('blue')
    
    # lanes on opposite site
    y = 2*radius_curve + i*distance_lanes
    lane = np.array([[xs[0], y, 0], [xs[1], y, 0]])
    stad_lanes.append(lane)
    colors.append('blue')
    
orthogonal_lanes = []
y0 = -n_lanes*distance_lanes
y1 = 2*radius_curve + n_lanes*distance_lanes
for xs in np.linspace(-15, len_straight + 15, 30):
    lane = np.array([[xs, y0, 0], [xs, y1, 0]])
    orthogonal_lanes.append(lane)
    
finish_lanes = []
y0 = -n_lanes*distance_lanes
y1 = distance_lanes
for xo in [-1, -.5, -.1, 0, .1, .5, 1]:
    xs = len_straight + xo
    lane = np.array([[xs, y0, 0], [xs, y1, 0]])
    finish_lanes.append(lane)
    
curves = []
# draw curves
for i in range(n_lanes):
    for x0 in [0, len_straight]:
        r = radius_curve + i*distance_lanes
        y0 = r - i*distance_lanes
        # Theta varies only between pi/2 and 3pi/2. to have a half-circle
        theta = np.linspace(np.pi/2., 3*np.pi/2., n_curve_segments)
        sign = 1 if x0==0 else -1 # flip the curve outward if far corner
        x = sign*r*np.cos(theta) + x0
        y = r*np.sin(theta) + y0
        z = np.zeros_like(theta)
        curves.append([x,y,z])
        
curves = np.array(curves)
stad_lanes = np.array(stad_lanes)
lanes_new = [curves.transpose([0, 2, 1]), stad_lanes]
if do_orthogonal:
    lanes_new.append(orthogonal_lanes)
if do_finish:
    lanes_new.append(finish_lanes)
# ====== stadium ======

def draw_short_lanes(ax, scale=1, shift=0, nlanes=3, len_straight_short=15, **kwargs):
    lanes_short = []
    for i in range(nlanes):
        xs = [-10, len_straight_short]
        # lanes on the homestretch
        y = -i*distance_lanes*scale + shift
        lane = np.array([[xs[0], y, 0], [xs[1], y, 0]])
        lanes_short.append(lane)
        ax.plot3D(*lane.T, linewidth=2, c='black', zorder=0, **kwargs)