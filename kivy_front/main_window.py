from typing import Any

from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.uix.actionbar import ActionBar, ActionView, ActionGroup
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView

from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout
from kivy.uix.button import Button

from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy_garden.contextmenu import AppMenu

from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from common.utils import Callable, Callback


Builder.load_file('./kivy_front/tabbed_view.kv')
Builder.load_file('./kivy_front/tabbed_view.kv')


class TabbedView(TabbedPanel):
    ...


class TimelineView(BoxLayout):
    ...


class VideoPreview(BoxLayout):
    ...


class MainWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

        menu_bar = BoxLayout(orientation='horizontal', spacing=10, size_hint=(0.3, 0.03))
        menu_bar.add_widget(Button(text='File'))
        menu_bar.add_widget(Button(text='Edit'))
        menu_bar.add_widget(Button(text='Other'))

        self.menu_bar = menu_bar
        self.tab_view = TabbedView()

        right_side = AnchorLayout(anchor_x='right')
        right_side.add_widget(self.tab_view)

        self.add_widget(self.menu_bar)
        self.add_widget(right_side)


class RenderWarriorKivyApp(App):
    def build(self):
        return MainWindow(orientation='vertical', spacing=10)


def launch_app(**kwargs):
    return RenderWarriorKivyApp(**kwargs).run()
