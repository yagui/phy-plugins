
# You can also put your plugins in ~/.phy/plugins/.

from phy import IPlugin

# Plugin example:
#
# class MyPlugin(IPlugin):
#     def attach_to_cli(self, cli):
#         # you can create phy subcommands here with click
#         pass

c = get_config()
c.Plugins.dirs = [r'/home/ariel/.phy/plugins']
c.TemplateGUI.plugins = []  # list of plugin names ti load in the TemplateGUI
c.TemplateGUI.plugins += ['myColumnsPlugin']  # list of plugin names ti load in the TemplateGUI
c.TemplateGUI.plugins += ['CustomFeatureViewPlugin']  # list of plugin names ti load in the TemplateGUI
#c.TemplateGUI.plugins += ['showNSpikesChannelsPlugin']  # list of plugin names ti load in the TemplateGUI
c.TemplateGUI.plugins += ['SingleChannelViewPlugin']  # list of plugin names ti load in the TemplateGUI
c.TemplateGUI.plugins += ['ClusterViewStylingPlugin']  # change colors for good mua and noise
