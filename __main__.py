#!/usr/bin/python38

import sys
import rw_cli
from kivy_front import main_window

__version__ = '0.0.1a @ 30-08-2020, 5:38 PM'
__author__ = 'github.com/cx3 aleksander.starostka@gmail.com +48664916155'
__title__ = 'Image Slideshow RenderWarrior'
__package__ = 'RenderWarrior'

__all__ = [
    'effects',
    'animations',
    'videoclip'
]

if 'cli' not in sys.argv:
    rw_cli.main()
else:
    main_window.launch_app()
