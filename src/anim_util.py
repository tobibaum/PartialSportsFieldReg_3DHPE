from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def create_anim_from_array(_vid_array, figsize=None):
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(_vid_array[0], interpolation='none')

    def init():
        im.set_array(_vid_array[0])
        return [im]

    def animate(i):
        im.set_array(_vid_array[i])
        return [im]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(_vid_array), interval=30)
    plt.axis('off')
    plt.tight_layout()
    plt.close()
    return anim

# in case we want to save this!
# from IPython.display import clear_output
# def callback_fun(i, n):
#     print(f'\rSaving frame {i} of {n}',)
#     clear_output(wait=True)
# writervideo = animation.FFMpegWriter(fps=30)
# anim_path = 'anim.mp4'
# anim.save(anim_path, writer=writervideo, progress_callback=callback_fun)

# display:
# HTML(anim.to_html5_video())
