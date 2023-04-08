import cv2
import os
import numpy as np
from pytube import YouTube


def vid_array_from_path(vid_path, t_start, np_name='', max_frames=500, force_fps=None):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if force_fps is not None:
        fps = force_fps
    print('extract %s (%d fps)'%(np_name, fps))

    frames = []
    ret, frame = cap.read()
    assert(ret)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    i=0
    j=0
    while(True):
        if i%1000==0:
            print(i, j)
        i+=1
        ret, frame = cap.read()
        if not ret:
            break

        if (i + 10) / fps >= t_start:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            j+=1
        if j>max_frames:
            break
    return np.array(frames)

def get_array_for_url(url, vid_dir='vids', np_dir='vids/np', size=None, max_frames=500, t_start=0, force=False, name_only=False, force_fps=None, fps=30):
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)
    is_local = not 'youtu' in url
    is_npy = url.endswith('npy')
    
    if is_npy:
        np_name = url
    else:
        if is_local:
            np_name = os.path.join(np_dir, url.split('/')[-1] + '_%d'%t_start + '.npy')
        else:
            yt = YouTube(url)
            vid_name = yt.title
            np_name = os.path.join(np_dir, vid_name.replace(' ', '_') + '_%d'%t_start + '.npy')
            
    if name_only:
        return np_name
        
    do_download = not os.path.exists(np_name) or force
    if do_download and not is_npy:
        if is_local:
            vid_path = url
        else:
            print('download video!')
            print(yt.streams)
            if size is not None:
                stream = yt.streams.filter(res='%dp'%size).filter(mime_type='video/mp4').filter(fps=fps).first()
                #stream = yt.streams.filter(res='%dp'%size).filter(mime_type='video/mp4').order_by('fps').asc().first()
            if size is None or stream is None:
                stream = yt.streams.order_by('resolution').desc().first()
            print(stream)
            vid_path = stream.download(vid_dir)
            
        vid_array = vid_array_from_path(vid_path, t_start, np_name, max_frames, force_fps=force_fps)
        np.save(np_name, vid_array)
    else:
        print('load %s'%np_name)
        vid_array = np.load(np_name)
    return vid_array

max_frame = 500
frame_size = 720

def load_bokeh_array_from_npy(img_array):
    C, M, N, _ = img_array.shape
    bokeh_array = np.empty((C, M, N), dtype=np.uint32)
    view = bokeh_array.view(dtype=np.uint8).reshape((C, M, N, 4))
    view[...,0] = img_array[...,0] # copy red channel
    view[...,1] = img_array[...,1] # copy blue channel
    view[...,2] = img_array[...,2] # copy green channel
    view[...,3] = 225
    bokeh_array = bokeh_array[:, ::-1]
    return bokeh_array

def resize_video(vid_array_in, _size=frame_size):
    import cv2
    N, H, W, C = vid_array_in.shape
    H_new = _size 
    W_new = int(W / H * H_new)
    new_frames = []
    for frame in vid_array_in:
        new_frames.append(cv2.resize(frame, (H_new, W_new)))
    vid_array_small = np.array(new_frames)
    return vid_array_small

def single_array_to_bokeh(arr):
    M, N, _ = arr.shape
    bokeh_array = np.empty((M, N), dtype=np.uint32)
    view = bokeh_array.view(dtype=np.uint8).reshape((M, N, 4))
    view[...,0] = arr[...,0] # copy red channel
    view[...,1] = arr[...,1] # copy blue channel
    view[...,2] = arr[...,2] # copy green channel
    view[...,3] = 225
    bokeh_array = bokeh_array[::-1]
    return bokeh_array
