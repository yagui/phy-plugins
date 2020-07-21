# import from plugins/waveformCuttingView.py
"""Show how to write a custom OpenGL view. This is for advanced users only."""

import numpy as np

from phy.utils.color import selected_cluster_color

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.cluster.views.base import  LassoMixin
from phy.plot.visuals import PlotVisual,TextVisual
from phy import connect

from phylib.utils.geometry import range_transform
from phy.plot import PlotCanvas, NDC, extend_bounds

class SingleChannelView(LassoMixin,ManualClusteringView):
    """All OpenGL views derive from ManualClusteringView."""

    def __init__(self, model=None):
        """
        Typically, the constructor takes as arguments *functions* that take as input
        one or several cluster ids, and return as many Bunch instances which contain
        the data as NumPy arrays. Many such functions are defined in the TemplateController.
        """

        super(SingleChannelView, self).__init__()
        self.canvas.enable_axes()
        self.canvas.enable_lasso()

        """
        The View instance contains a special `canvas` object which is a `̀PlotCanvas` instance.
        This class derives from `BaseCanvas` which itself derives from the PyQt5 `QOpenGLWindow`.
        The canvas represents a rectangular black window where you can draw geometric objects
        with OpenGL.

        phy uses the notion of **Layout** that lets you organize graphical elements in different
        subplots. These subplots can be organized in several ways:

        * Grid layout: a `(n_rows, n_cols)` grid of subplots (example: FeatureView).
        * Boxed layout: boxes arbitrarily located (example: WaveformView, using the
          probe geometry)
        * Stacked layout: one column with `n_boxes` subplots (example: TraceView,
          one row per channel)

        In this example, we use the stacked layout, with one subplot per cluster. This number
        will change at each cluster selection, depending on the number of selected clusters.
        But initially, we just use 1 subplot.

        """
        self.canvas.set_layout('stacked', n_plots=1)
        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

        self.model = model

        """
        phy uses the notion of **Visual**. This is a graphical element that is represented with
        a single type of graphical element. phy provides many visuals:

        * PlotVisual (plots)
        * ScatterVisual (points with a given marker type and different colors and sizes)
        * LineVisual (for lines segments)
        * HistogramVisual
        * PolygonVisual
        * TextVisual
        * ImageVisual

        Each visual comes with a single OpenGL program, which is defined by a vertex shader
        and a fragment shader. These are programs written in a C-like language called GLSL.
        A visual also comes with a primitive type, which can be points, line segments, or
        triangles. This is all a GPU is able to render, but the position and the color of
        these primitives can be entirely customized in the shaders.

        The vertex shader acts on data arrays represented as NumPy arrays.

        These low-level details are hidden by the visuals abstraction, so it is unlikely that
        you'll ever need to write your own visual.

        In ManualClusteringViews, you typically define one or several visuals. For example
        if you need to add text, you would add `self.text_visual = TextVisual()`.

        """
        self.visual = PlotVisual()

        """
        For internal reasons, you need to add all visuals (empty for now) directly to the
        canvas, in the view's constructor. Later, we will use the `visual.set_data()` method
        to update the visual's data and display something in the figure.

        """
        self.canvas.add_visual(self.visual)
        self.cluster_id = None
        self.channel_id = None

    def on_select(self, cluster_ids=(), **kwargs):
        """
        The main method to implement in ManualClusteringView is `on_select()`, called whenever
        new clusters are selected.

        *Note*: `cluster_ids` contains the clusters selected in the cluster view, followed
        by clusters selected in the similarity view.

        """

        """
        This method should always start with these few lines of code.
        """
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        
        if self.cluster_id != cluster_ids[0]:
            self.cluster_id = cluster_ids[0]
            """
            We update the number of boxes in the stacked layout, which is the number of
            selected clusters.
            """

            """
            We obtain the waveforms for the first cluster selected.
            """
            self.setChannel(self.model.get_cluster_channels(self.cluster_id)[0])
    


    def plotWaveforms(self):

        self.spikes_id = self.model.get_cluster_spikes(self.cluster_id)
        wavefs = self.model.get_waveforms(self.spikes_id,channel_ids=[self.channel_id])[:,:]
        self.wavefs = wavefs[:,:,0]

        """
        We update the number of boxes in the stacked layout, which is the number of
        selected clusters.
        """
        self.canvas.stacked.n_boxes = 1

        """
        For performance reasons, it is best to use as few visuals as possible. In this example,
        we want 1 waveform template per subplot. We will use a single visual covering all
        subplots at once. This is the key to achieve good performance with OpenGL in Python.
        However, this comes with the drawback that the programming interface is more complicated.

        In principle, we would have to concatenate all data (x and y coordinates) of all subplots
        to pass it to `self.visual.set_data()` in order to draw all subplots at once. But this
        is tedious.

        phy uses the notion of **batch**: for each subplot, we set *partial data* for the subplot
        which just prepares the data for concatenation *after* we're done with looping through
        all clusters. The concatenation happens in the special call
        `self.canvas.update_visual(self.visual)`.

        We need to call `visual.reset_batch()` before constructing a batch.

        """
        self.visual.reset_batch()

        """
        We iterate through all selected clusters.
        """



        """
        In this example, we just keep the peak channel. Note that `bunch.template` is a
        2D array `(n_samples, n_channels)` where `n_channels` in the number of "best"
        channels for the cluster. The channels are sorted by decreasing template amplitude,
        so the first one is the peak channel. The channel ids can be found in
        `bunch.channel_ids`.
        """

        """
        We decide to use, on the x axis, values ranging from -1 to 1. This is the
        standard viewport in OpenGL and phy.
        """
        #x = np.linspace(-1., 1., wavefs.shape[1])
        x = np.tile(np.linspace(-1., 1., self.wavefs.shape[1]), (self.wavefs.shape[0], 1))

        """
        phy requires you to specify explicitly the x and y range of the plots.
        The `data_bounds` variable is a `(xmin, ymin, xmax, ymax)` tuple representing the
        lower-left and upper-right corners of a rectangle. By default, the data bounds
        of the entire view is (-1, -1, 1, 1), also called normalized device coordinates.
        Eventually, OpenGL uses this coordinate system for display, but phy provides
        a transform system to convert from different coordinate systems, both on the CPU
        and the GPU.

        Here, the x range is (-1, 1), and the y range is (m, M) where m and M are
        respectively the min and max of the template.
        """
        M=np.max(np.abs(self.wavefs))
        self.data_bounds = (-1, -M, +1, M)

        """
        This function gives the color of the i-th selected cluster. This is a 4-tuple with
        values between 0 and 1 for RGBA: red, green, blue, alpha channel (transparency,
        1 by default).
        """
        color = selected_cluster_color(0)

        """
        The plot visual takes as input the x and y coordinates of the points, the color,
        and the data bounds.
        There is also a special keyword argument `box_index` which is the subplot index.
        In the stacked layout, this is just an integer identifying the subplot index, from
        top to bottom. Note that in the grid view, the box index is a pair (row, col).
        """
        self.visual.add_batch_data(
            x=x, y=self.wavefs, color=color, data_bounds=self.data_bounds, box_index=0)

       # Add channel labels.
        #if not self.do_show_labels:
        #    return
        self.text_visual.reset_batch()
        label = '{a}'.format(a=self.channel_id)
        self.text_visual.add_batch_data(
                pos=[-1, 0],
                text=str(label),
                anchor=[-1.25, 0],
                box_index=0,
            )
        self.canvas.update_visual(self.text_visual)


        #ax_db = self.data_bounds
        #hpos = np.tile([[-1, 0, 1, 0]], (1, 1))
        #self.line_visual.add_batch_data(
        #    pos=hpos,
        #    color=self.ax_color,
        #    data_bounds=ax_db,
        #    box_index=0,
        #)

#        # Vertical ticks every millisecond.
#        steps = np.arange(np.round(self.wave_duration * 1000))
#        # A vline every millisecond.
#        x = .001 * steps
#        # Scale to [-1, 1], same coordinates as the waveform points.
#        x = -1 + 2 * x / self.wave_duration
#        # Take overlap into account.
#        x = _overlap_transform(x, offset=bunch.offset, n=bunch.n_clu, overlap=self.overlap)
#        x = np.tile(x, len(channel_ids_loc))
#        # Generate the box index.
#        box_index = _index_of(channel_ids_loc, self.channel_ids)
#        box_index = np.repeat(box_index, x.size // len(box_index))
#        assert x.size == box_index.size
#        self.tick_visual.add_batch_data(
#            x=x, y=np.zeros_like(x),
#            data_bounds=ax_db,
#            box_index=box_index,
#        )
#



        """
        After the loop, this special call automatically builds the data to upload to the GPU
        by concatenating the partial data set in `add_batch_data()`.
        """
        self.canvas.update_visual(self.visual)

        """
        After updating the data on the GPU, we need to refresh the canvas.
        """
        self.canvas.update()
    

    def setChannel(self,channel_id):
        self.channel_id = channel_id
        self.plotWaveforms()

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or not len(self.cluster_ids)):  # pragma: no cover
            return np.array([], dtype=np.int64)
        # Get all points from all clusters.
        pos = []
        spike_ids = []
        x = np.linspace(-1., 1., self.wavefs.shape[1])

        # each item is a Bunch with attribute `pos` et `spike_ids`
        #bunchs = self.get_clusters_data(load_all=True)
        #if bunchs is None:
        #    return
        for idx,spike in enumerate(self.spikes_id):
            points = np.c_[x,self.wavefs[idx]]
            pos.append(points)
            spike_ids.append([spike]*len(x))
        if not pos:  # pragma: no cover
            logger.warning("Empty lasso.")
            return np.array([])
        pos = np.vstack(pos)
        pos = range_transform([self.data_bounds], [NDC], pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])
    
    def on_select_channel(self, sender=None, channel_id=None, key=None, button=None):
        """Respond to the click on a channel from another view, and update the
        relevant subplots."""
        print(channel_id)
        if self.channel_id != channel_id:
            self.setChannel(channel_id)
#        channels = self.channel_ids
#        if channels is None:
#            return
#        if len(channels) == 1:
#            self.plot()
#            return
#        assert len(channels) >= 2
#        # Get the axis from the pressed button (1, 2, etc.)
#        if key is not None:
#            d = np.clip(len(channels) - 1, 0, key - 1)
#        else:
#            d = 0 if button == 'Left' else 1
#        # Change the first or second best channel.
#        old = channels[d]
#        # Avoid updating the view if the channel doesn't change.
#        if channel_id == old:
#            return
#        channels[d] = channel_id
#        # Ensure that the first two channels are different.
#        if channels[1 - min(d, 1)] == channel_id:
#            channels[1 - min(d, 1)] = old
#        assert channels[0] != channels[1]
#        # Remove duplicate channels.
#        self.channel_ids = _uniq(channels)
#        logger.debug("Choose channels %d and %d in feature view.", *channels[:2])
#        # Fix the channels temporarily.
#        self.plot(fixed_channels=True)
#        self.update_status()
 
class SingleChannelViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_my_view():
            return SingleChannelView(model=controller.model)

            @connect(sender=view)
            def on_select_channel(sender, channel_id=None, key=None, button=None):
                # Update the Selection object with the channel id clicked in the waveform view.
                self.selection.channel_id = channel_id
                


        controller.view_creator['SingleChannelView'] = create_my_view

        # Open a view if there is not already one.
        #controller.at_least_one_view('SingleChannelView')

