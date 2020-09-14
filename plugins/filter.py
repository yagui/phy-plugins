from phy import IPlugin, connect

filterCondition="group!='noise'"

class FilterNotNoisePlugin(IPlugin):
    def attach_to_controller(self, controller):

        self.filtering = False
        
        @connect
        def on_gui_ready(sender, gui):

            @gui.view_actions.add(alias='fnn',shortcut='ctrl+f')  # corresponds to `:fnn` snippet
            def filterNotNoise():
                """Filter clusters good and mua."""
                if self.filtering:
                    controller.supervisor.filter('')
                    controller.supervisor.similarity_view.filter('')
                    self.filtering = False
                else:
                    controller.supervisor.filter(filterCondition)
                    controller.supervisor.similarity_view.filter(filterCondition)
                    self.filtering = True
