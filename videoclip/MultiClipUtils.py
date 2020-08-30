from videoclip.ClipUtils import KeyedEffect, KeyedEffectsStack, StillImage, NextFrame


class MultiTrack(NextFrame):
    def __init__(self):
        self.tracks = {}
        raise NotImplementedError
