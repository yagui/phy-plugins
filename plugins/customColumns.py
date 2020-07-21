# import from plugins/custom_columns.py
"""Show how to customize the columns in the cluster and similarity views."""

from phy import IPlugin, connect

class myColumnsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            #controller.supervisor.columns = ['id', 'channel','firing_rate','n_spikes','depth','amplitude','Amplitude','ContamPct','KsLabel']
            controller.supervisor.columns = ['id', 'ch','fr','n_spikes']

