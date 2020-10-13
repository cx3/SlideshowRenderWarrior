# Python's STL imports - None
import os
# Third party imports

# RenderWarrior imports
from animations import transitions_classes as trans
from videoclip import RenderControllers


def main():
    media_dir = './media_files/source_images/'

    wizard = RenderControllers.WizardSlideshowZoomy(
        image_sources=[media_dir + f for f in os.listdir(media_dir)][::-1][:3],
        transitions_list=[
            trans.BlendingTransition,
        ] * 10,
        frame_resolution=(720, 480),
        fps=5,
        slide_seconds=4,
        transition_seconds=2,
    )

    input('Enter for start rendering project... ')

    wizard.render_project(
        avi='./media_files/RENDERED.avi',
        show_avi=True,
        del_frames=False,
    )
