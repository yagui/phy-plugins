# import from plugins/cluster_view_styling.py
"""Show how to customize the styling of the cluster view with CSS."""

from phy import IPlugin
from phy.cluster.supervisor import ClusterView


class ClusterViewStylingPlugin(IPlugin):
    def attach_to_controller(self, controller):
        # We add a custom CSS style to the ClusterView.
        ClusterView._styles += """

            /* This CSS selector represents all rows for good clusters. */
            table tr[data-group='mua'] {

                /* We change the text color. Many other CSS attributes can be changed,
                such as background-color, the font weight, etc. */
                color: orange;
            }

        """

