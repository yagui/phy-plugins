
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

c.TemplateGUI.plugins += ['myColumnsPlugin']  # remove unnecesary columns
c.TemplateGUI.plugins += ['CustomFeatureViewPlugin']  # add 3 principal component
c.TemplateGUI.plugins += ['WaveformClusteringViewPlugin']  # Plugin to clean each cluster through removing waveforms
c.TemplateGUI.plugins += ['ClusterViewStylingPlugin']  # change colors for mua
c.TemplateGUI.plugins += ['kmSplitPlugin']  # kmeans, split in 2
c.TemplateGUI.plugins += ['FilterNotNoisePlugin']  # show only good and mua

