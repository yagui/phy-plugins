# Ariel Burman - 2020
# 

import numpy as np

from phy.utils.color import selected_cluster_color

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.cluster.views.base import  LassoMixin,ScalingMixin
from phy.plot.visuals import PlotVisual,TextVisual
from phy import connect

from phylib.utils.geometry import range_transform
from phy.plot import PlotCanvas, NDC


class WaveformClusteringView(LassoMixin,ManualClusteringView):

    default_shortcuts = {
        'next_channel': 'f',
        'previous_channel': 'r',
    }

    def __init__(self, model=None):

        super(WaveformClusteringView, self).__init__()
        self.canvas.enable_axes()
        self.canvas.enable_lasso()
        
        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual, exclude_origins=(self.canvas.panzoom,))

        self.model = model

        self.gain = 0.195
        self.Fs = 30  # kHz

        self.visual = PlotVisual()

        self.canvas.add_visual(self.visual)
        self.canvas.panzoom.zoom = self.canvas.panzoom._default_zoom = (.97, .95)
        self.canvas.panzoom.pan = self.canvas.panzoom._default_pan = (-.01, 0)
        self.cluster_ids = None
        self.cluster_id = None
        self.channel_ids = None
        self.wavefs = None
        self.current_channel_idx = None
    
    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        
        if self.cluster_id != cluster_ids[0]:
            self.cluster_id = cluster_ids[0]

            self.channel_ids = self.model.get_cluster_channels(self.cluster_id)
            self.spike_ids = self.model.get_cluster_spikes(self.cluster_id)
            self.wavefs = self.model.get_waveforms(self.spike_ids,channel_ids=self.channel_ids)

            self.setChannelIdx(0)
        

    def plotWaveforms(self):

        Nspk,Ntime,Nchan = self.wavefs.shape

        self.visual.reset_batch()
        self.text_visual.reset_batch()

        x = np.tile(np.linspace(-Ntime/2/self.Fs, Ntime/2/self.Fs, Ntime), (Nspk, 1))

        M=np.max(np.abs(self.wavefs[:,:,self.current_channel_idx]))
        #print(M*self.gain)
        if M*self.gain<100:
            M = 10*np.ceil(M*self.gain/10)
        elif M*self.gain<1000:
            M = 100*np.ceil(M*self.gain/100)
        else:
            M = 1000*np.floor(M*self.gain/1000)
        self.data_bounds = (x[0][0], -M, x[0][-1], M)

        colorwavef = selected_cluster_color(0)
        colormedian = selected_cluster_color(3)#(1,156/256,0,.5)#selected_cluster_color(1)
        colorstd = (0,1,0,1)#selected_cluster_color(2)
        colorqtl = (1,1,0,1)

        if Nspk>100:
            medianCl = np.median(self.wavefs[:,:,self.current_channel_idx],axis=0)
            stdCl = np.std(self.wavefs[:,:,self.current_channel_idx],axis=0)
            q1 = np.quantile(self.wavefs[:,:,self.current_channel_idx],.01,axis=0,interpolation='higher')
            q9 = np.quantile(self.wavefs[:,:,self.current_channel_idx],.99,axis=0,interpolation='lower')

        self.visual.add_batch_data(
                x=x, y=self.gain*self.wavefs[:,:,self.current_channel_idx], color=colorwavef, data_bounds=self.data_bounds, box_index=0)

        #stats
        if Nspk>100:
            x1 = x[0]
            self.visual.add_batch_data(
                    x=x1, y=self.gain*medianCl, color=colormedian, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=self.gain*(medianCl+3*stdCl), color=colorstd, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=self.gain*(medianCl-3*stdCl), color=colorstd, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=self.gain*q1, color=colorqtl, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=self.gain*q9, color=colorqtl, data_bounds=self.data_bounds, box_index=0)

        #axes
        self.text_visual.add_batch_data(
                pos=[.9, .98],
                text='[uV]',
                anchor=[-1, -1],
                box_index=0,
            )
        
        self.text_visual.add_batch_data(
                pos=[-1, -.95],
                text='[ms]',
                anchor=[1, 1],
                box_index=0,
            )

        label = 'Ch {a}'.format(a=self.channel_ids[self.current_channel_idx])
        self.text_visual.add_batch_data(
                pos=[-.98, .98],
                text=str(label),
                anchor=[1, -1],
                box_index=0,
            )
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.text_visual)
        self.canvas.axes.reset_data_bounds(self.data_bounds)

        self.canvas.update()
    
    def setChannel(self,channel_id):
        self.channel_ids
        itemindex = np.where(self.channel_ids==channel_id)[0]
        if len(itemindex):
            self.setChannelIdx(itemindex[0])

    def setChannelIdx(self,channel_idx):
        self.current_channel_idx = channel_idx
        self.plotWaveforms()

    def setNextChannelIdx(self):
        if self.current_channel_idx == len(self.channel_ids)-1:
            return
        self.setChannelIdx(self.current_channel_idx+1)

    def setPrevChannelIdx(self):
        if self.current_channel_idx == 0:
            return
        self.setChannelIdx(self.current_channel_idx-1)

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or not len(self.cluster_ids)):  # pragma: no cover
            return np.array([], dtype=np.int64)

        pos = []
        spike_ids = []

        Ntime = self.wavefs.shape[1]
        x = np.linspace(-Ntime/2/self.Fs, Ntime/2/self.Fs, Ntime)

        for idx,spike in enumerate(self.spike_ids):
            points = np.c_[x,self.gain*self.wavefs[idx,:,self.current_channel_idx]]
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

        # Return all spikes not lassoed, so the selected cluster is still the same we are working on
        spikes_to_remove = np.unique(spike_ids[ind])
        keepspikes=np.isin(self.spike_ids,spikes_to_remove,assume_unique=True,invert=True)
        A=self.spike_ids[self.spike_ids != spikes_to_remove]
        if len(A)>0:
            return self.spike_ids[keepspikes]
        else:
            return np.array([], dtype=np.int64)
    
 
class WaveformClusteringViewPlugin(IPlugin):

    def attach_to_controller(self, controller):
        def create_my_view():
            return WaveformClusteringView(model=controller.model)

        controller.view_creator['WaveformClusteringView'] = create_my_view

        @connect
        def on_select_channel(sender,channel_id,key,button):
            self.view.setChannel(channel_id)

        @connect
        def on_view_attached(view, gui):
            if isinstance(view, WaveformClusteringView):

                @view.dock.add_button(icon='f105')
                def nextChannelType(checked):
                    # The checked argument is only used with buttons `checkable=True`
                    view.setNextChannelIdx()

                # The icon unicode can be found at https://fontawesome.com/icons?d=gallery
                @view.dock.add_button(icon='f104')
                def prevChannelType(checked):
                    # The checked argument is only used with buttons `checkable=True`
                    view.setPrevChannelIdx()

                @view.actions.add(shortcut='f')
                def next_channel():
                    """Select Previous Channel Index"""
                    v=gui.get_view('WaveformClusteringView')
                    if v:
                        v.setNextChannelIdx()

                @view.actions.add(shortcut='r')
                def prev_channel():
                    """Select Next Channel Index"""
                    v=gui.get_view('WaveformClusteringView')
                    if v:
                        v.setPrevChannelIdx()
