import numpy as np
N_JOINTS = 17
N_CHANNELS = 3

mapping = \
[['K4ABT_JOINT_PELVIS', 0],
 ['K4ABT_JOINT_SPINE_NAVEL', 7],
 ['K4ABT_JOINT_SPINE_CHEST', -1],
 ['K4ABT_JOINT_NECK', 8],
 ['K4ABT_JOINT_CLAVICLE_LEFT', -1],
 ['K4ABT_JOINT_SHOULDER_LEFT', 14],
 ['K4ABT_JOINT_ELBOW_LEFT', 15],
 ['K4ABT_JOINT_WRIST_LEFT', 16],
 ['K4ABT_JOINT_HAND_LEFT', -1],
 ['K4ABT_JOINT_HANDTIP_LEFT', -1],
 ['K4ABT_JOINT_THUMB_LEFT', -1],
 ['K4ABT_JOINT_CLAVICLE_RIGHT', -1],
 ['K4ABT_JOINT_SHOULDER_RIGHT', 11],
 ['K4ABT_JOINT_ELBOW_RIGHT', 12],
 ['K4ABT_JOINT_WRIST_RIGHT', 13],
 ['K4ABT_JOINT_HAND_RIGHT', -1],
 ['K4ABT_JOINT_HANDTIP_RIGHT', -1],
 ['K4ABT_JOINT_THUMB_RIGHT', -1],
 ['K4ABT_JOINT_HIP_LEFT', 1],
 ['K4ABT_JOINT_KNEE_LEFT', 2],
 ['K4ABT_JOINT_ANKLE_LEFT', 3],
 ['K4ABT_JOINT_FOOT_LEFT', -1],
 ['K4ABT_JOINT_HIP_RIGHT', 4],
 ['K4ABT_JOINT_KNEE_RIGHT', 5],
 ['K4ABT_JOINT_ANKLE_RIGHT', 6],
 ['K4ABT_JOINT_FOOT_RIGHT', -1],
 ['K4ABT_JOINT_HEAD', 10],
 ['K4ABT_JOINT_NOSE', 9],
 ['K4ABT_JOINT_EYE_LEFT', -1],
 ['K4ABT_JOINT_EAR_LEFT', -1],
 ['K4ABT_JOINT_EYE_RIGHT', -1],
 ['K4ABT_JOINT_EAR_RIGHT', -1]]

rev_map_name = {m[1]: m[0] for m in mapping}
remap = [m[1] for m in mapping]
rev_map = {m[1]: i for i, m in enumerate(mapping) if m[1]!=-1}
rev_idc = [rev_map[i] for i in range(N_JOINTS)]


mapping_metrab = \
[['K4ABT_JOINT_PELVIS', 23],
 ['K4ABT_JOINT_SPINE_NAVEL', 5],
 ['K4ABT_JOINT_SPINE_CHEST', 8],
 ['K4ABT_JOINT_NECK', 11],
 ['K4ABT_JOINT_CLAVICLE_LEFT', 12],
 ['K4ABT_JOINT_SHOULDER_LEFT', 15],
 ['K4ABT_JOINT_ELBOW_LEFT', 17],
 ['K4ABT_JOINT_WRIST_LEFT', 19],
 ['K4ABT_JOINT_HAND_LEFT', 21],
 ['K4ABT_JOINT_HANDTIP_LEFT', -1],
 ['K4ABT_JOINT_THUMB_LEFT', -1],
 ['K4ABT_JOINT_CLAVICLE_RIGHT', 13],
 ['K4ABT_JOINT_SHOULDER_RIGHT', 16],
 ['K4ABT_JOINT_ELBOW_RIGHT', 18],
 ['K4ABT_JOINT_WRIST_RIGHT', 20],
 ['K4ABT_JOINT_HAND_RIGHT', 22],
 ['K4ABT_JOINT_HANDTIP_RIGHT', -1],
 ['K4ABT_JOINT_THUMB_RIGHT', -1],
 ['K4ABT_JOINT_HIP_LEFT', 0],
 ['K4ABT_JOINT_KNEE_LEFT', 3],
 ['K4ABT_JOINT_ANKLE_LEFT', 6],
 ['K4ABT_JOINT_FOOT_LEFT', 9],
 ['K4ABT_JOINT_HIP_RIGHT', 1],
 ['K4ABT_JOINT_KNEE_RIGHT', 4],
 ['K4ABT_JOINT_ANKLE_RIGHT', 7],
 ['K4ABT_JOINT_FOOT_RIGHT', 10],
 ['K4ABT_JOINT_HEAD', 14],
 ['K4ABT_JOINT_NOSE', -1],
 ['K4ABT_JOINT_EYE_LEFT', -1],
 ['K4ABT_JOINT_EAR_LEFT', -1],
 ['K4ABT_JOINT_EYE_RIGHT', -1],
 ['K4ABT_JOINT_EAR_RIGHT', -1]]

# if left and right are switched, this is 4! else 0
#lr = 0
lr = 4

mapping_xsens = \
[['K4ABT_JOINT_PELVIS', 0],
 ['K4ABT_JOINT_SPINE_NAVEL', 3],
 ['K4ABT_JOINT_SPINE_CHEST', -1],
 ['K4ABT_JOINT_NECK', 5],
 
 ['K4ABT_JOINT_CLAVICLE_LEFT', 7+lr],
 ['K4ABT_JOINT_SHOULDER_LEFT', 8+lr],
 ['K4ABT_JOINT_ELBOW_LEFT', 9+lr],
 ['K4ABT_JOINT_WRIST_LEFT', 10+lr],
 ['K4ABT_JOINT_HAND_LEFT', -1],
 ['K4ABT_JOINT_HANDTIP_LEFT', -1],
 ['K4ABT_JOINT_THUMB_LEFT', -1],
 
 ['K4ABT_JOINT_CLAVICLE_RIGHT', 11-lr],
 ['K4ABT_JOINT_SHOULDER_RIGHT', 12-lr],
 ['K4ABT_JOINT_ELBOW_RIGHT', 13-lr],
 ['K4ABT_JOINT_WRIST_RIGHT', 14-lr],
 ['K4ABT_JOINT_HAND_RIGHT', -1],
 ['K4ABT_JOINT_HANDTIP_RIGHT', -1],
 ['K4ABT_JOINT_THUMB_RIGHT', -1],
 
 ['K4ABT_JOINT_HIP_LEFT', 15+lr],
 ['K4ABT_JOINT_KNEE_LEFT', 16+lr],
 ['K4ABT_JOINT_ANKLE_LEFT', 17+lr],
 ['K4ABT_JOINT_FOOT_LEFT', 18+lr],
 
 ['K4ABT_JOINT_HIP_RIGHT', 19-lr],
 ['K4ABT_JOINT_KNEE_RIGHT', 20-lr],
 ['K4ABT_JOINT_ANKLE_RIGHT', 21-lr],
 ['K4ABT_JOINT_FOOT_RIGHT', 22-lr],
 ['K4ABT_JOINT_HEAD', 6],
 ['K4ABT_JOINT_NOSE', -1],
 ['K4ABT_JOINT_EYE_LEFT', -1],
 ['K4ABT_JOINT_EAR_LEFT', -1],
 ['K4ABT_JOINT_EYE_RIGHT', -1],
 ['K4ABT_JOINT_EAR_RIGHT', -1]]

mapping_yolov7 = \
[['K4ABT_JOINT_PELVIS', -1],
 ['K4ABT_JOINT_SPINE_NAVEL', -1],
 ['K4ABT_JOINT_SPINE_CHEST', -1],
 ['K4ABT_JOINT_NECK', -1],
 
 ['K4ABT_JOINT_CLAVICLE_LEFT', -1],
 ['K4ABT_JOINT_SHOULDER_LEFT', 6],
 ['K4ABT_JOINT_ELBOW_LEFT', 8],
 ['K4ABT_JOINT_WRIST_LEFT', 10],
 ['K4ABT_JOINT_HAND_LEFT', -1],
 ['K4ABT_JOINT_HANDTIP_LEFT', -1],
 ['K4ABT_JOINT_THUMB_LEFT', -1],
 
 ['K4ABT_JOINT_CLAVICLE_RIGHT', -1],
 ['K4ABT_JOINT_SHOULDER_RIGHT', 5],
 ['K4ABT_JOINT_ELBOW_RIGHT', 7],
 ['K4ABT_JOINT_WRIST_RIGHT', 9],
 ['K4ABT_JOINT_HAND_RIGHT', -1],
 ['K4ABT_JOINT_HANDTIP_RIGHT', -1],
 ['K4ABT_JOINT_THUMB_RIGHT', -1],
 
 ['K4ABT_JOINT_HIP_LEFT', 12],
 ['K4ABT_JOINT_KNEE_LEFT', 14],
 ['K4ABT_JOINT_ANKLE_LEFT', 16],
 ['K4ABT_JOINT_FOOT_LEFT', -1],
 
 ['K4ABT_JOINT_HIP_RIGHT', 11],
 ['K4ABT_JOINT_KNEE_RIGHT', 13],
 ['K4ABT_JOINT_ANKLE_RIGHT', 15],
 ['K4ABT_JOINT_FOOT_RIGHT', -1],
 
 ['K4ABT_JOINT_HEAD', -1],
 ['K4ABT_JOINT_NOSE', 0],
 ['K4ABT_JOINT_EYE_LEFT', 2],
 ['K4ABT_JOINT_EAR_LEFT', 4],
 ['K4ABT_JOINT_EYE_RIGHT', 1],
 ['K4ABT_JOINT_EAR_RIGHT', 3]]


mapping_h36m = mapping.copy()
mapping_motionbert = mapping.copy()

xsens_kinect_inds = [m[1] for m in mapping_xsens]
metrab_kinect_inds = [m[1] for m in mapping_metrab]
yolo_kinect_inds = [m[1] for m in mapping_yolov7]

# to transform metrabs coco_19 into our h36m coords. fairly close match!!
coco_to_h36m = [2, 6, 7, 8, 12, 13, 14, 2, 0, 1, 18, 9, 10, 11, 3, 4, 5]

COCO_lines = [[6,5,4,0,7,8,10],
              [3,2,1,0],
              [13,12,11,8],
              [16,15,14,8]
             ]

coco_line_colors = {0: 'blue',
                    1: (.1, .4, 1),
                    2: (.2, .6, 1),
                    3: 'black',
                    4: 'black',
                    5: 'black',
                    6: 'red',
                    7: (1, .4, .4),
                    8: (1, .6, .6),
                    9: (.5, .0, .5),
                    10: (.4, .1, .4),
                    11: (.6, .4, .6),
                    12: (1, .5, 0),
                    13: (1, .5, .3),
                    14: (1, .7, .6)
                   }

skip_idc = [7, 9, 10]
good_idc = list(np.arange(17))
for si in skip_idc:
    good_idc.remove(si)

COCO_pairs = []
for l in COCO_lines:
    for k in range(len(l) - 1):
        COCO_pairs.append([l[k], l[k+1]])
        
def pos_to_coco_lines(pos):
    skel_lines = []
    for cp in COCO_pairs:
        skel_lines.append([pos[cp[0]], pos[cp[1]]])
    skel_lines = np.array(skel_lines)
    return skel_lines