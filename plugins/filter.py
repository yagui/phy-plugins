from phy import IPlugin, connect

class FilterNotNoisePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            @gui.view_actions.add(alias='fnn')  # corresponds to `:fnn` snippet
            def filterNotNoise():
                """Filter clusters good and mua."""
                controller.supervisor.filter("group!='noise'")
