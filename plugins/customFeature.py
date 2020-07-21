# import from plugins/feature_view_custom_grid.py
"""Show how to customize the subplot grid specifiction in the feature view."""

import re
from phy import IPlugin, connect
from phy.cluster.views import FeatureView


def my_grid():
    """In the grid specification, 0 corresponds to the best channel, 1
    to the second best, and so on. A, B, C refer to the PC components."""
    s = """
    time,0A 0A,0B   1A,1B   0B,1A
    0A,0C   time,1A 0A,1A   0A,1B
    1A,1C   0A,1C   time,0B 0B,1B
    0B,0C   1B,1C   1A,0C   time,1B
    """.strip()
    return [[_ for _ in re.split(' +', line.strip())] for line in s.splitlines()]


class CustomFeatureViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            if isinstance(view, FeatureView):
                # We change the specification of the subplots here.
                view.set_grid_dim(my_grid())
