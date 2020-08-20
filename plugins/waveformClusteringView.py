# Ariel Burman - 2020
# 2020

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

#    default_shortcuts = {
#        'next_channel': 'alt+d',
#        'previous_channel': 'alt+e',
#    }

    def __init__(self, model=None):

        super(WaveformClusteringView, self).__init__()
        self.canvas.enable_axes()
        self.canvas.enable_lasso()
        
        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

        self.model = model

        self.visual = PlotVisual()

        self.canvas.add_visual(self.visual)
        self.canvas.panzoom.zoom = self.canvas.panzoom._default_zoom = (.94, .98)
        self.canvas.panzoom.pan = self.canvas.panzoom._default_pan = (0.03, 0)
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

        x = np.tile(np.linspace(-1., 1., Ntime), (Nspk, 1))

        M=np.max(np.abs(self.wavefs[:,:,self.current_channel_idx]))
        self.data_bounds = (-1, -M, +1, M)

        colorwavef = selected_cluster_color(0)
        colormedian = selected_cluster_color(3)#(1,156/256,0,.5)#selected_cluster_color(1)
        colorstd = (0,1,0,1)#selected_cluster_color(2)
        colorqtl = (1,1,0,1)

        x1 = np.linspace(-1., 1., Ntime)
        if Nspk>100:
            medianCl = np.median(self.wavefs[:,:,self.current_channel_idx],axis=0)
            stdCl = np.std(self.wavefs[:,:,self.current_channel_idx],axis=0)
            q1 = np.quantile(self.wavefs[:,:,self.current_channel_idx],.01,axis=0,interpolation='higher')
            q9 = np.quantile(self.wavefs[:,:,self.current_channel_idx],.99,axis=0,interpolation='lower')

        self.visual.add_batch_data(
                x=x, y=self.wavefs[:,:,self.current_channel_idx], color=colorwavef, data_bounds=self.data_bounds, box_index=0)

        #axes
        self.visual.add_batch_data(
                x=x1, y=np.zeros((1,Ntime)), color=(1,1,1,1), data_bounds=self.data_bounds, box_index=0)

        self.visual.add_batch_data(
                x=np.array([-1,-1]), y=np.array([-M,M]), color=(1,1,1,1), data_bounds=self.data_bounds, box_index=0)
        self.visual.add_batch_data(
                x=np.array([-.36,-.36]), y=np.array([-M,M]), color=(1,1,1,.5), data_bounds=self.data_bounds, box_index=0)
        self.visual.add_batch_data(
                x=np.array([.36,.36]), y=np.array([-M,M]), color=(1,1,1,.5), data_bounds=self.data_bounds, box_index=0)
        
        #stats
        if Nspk>100:
            self.visual.add_batch_data(
                    x=x1, y=medianCl, color=colormedian, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=medianCl+3*stdCl, color=colorstd, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=medianCl-3*stdCl, color=colorstd, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=q1, color=colorqtl, data_bounds=self.data_bounds, box_index=0)
            self.visual.add_batch_data(
                    x=x1, y=q9, color=colorqtl, data_bounds=self.data_bounds, box_index=0)

        #axes label
        if M*0.195<1000:
            axescalelabel = '{a:.1f} uV'.format(a=M*0.195)
        else:
            axescalelabel = '{a:.1f} mV'.format(a=M*0.000195)

        self.text_visual.add_batch_data(
                pos=[-1, 1],
                text=axescalelabel,
                anchor=[1.25, 0],
                box_index=0,
            )
        
        self.text_visual.add_batch_data(
                pos=[-1, -1],
                text='-'+axescalelabel,
                anchor=[1.25, 0],
                box_index=0,
            )

        label = '{a}'.format(a=self.channel_ids[self.current_channel_idx])
        self.text_visual.add_batch_data(
                pos=[-1, 0],
                text=str(label),
                anchor=[-1.25, 0],
                box_index=0,
            )
        self.canvas.update_visual(self.text_visual)

        self.canvas.update_visual(self.visual)

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

        x = np.linspace(-1., 1., self.wavefs.shape[1])

        for idx,spike in enumerate(self.spike_ids):
            points = np.c_[x,self.wavefs[idx,:,self.current_channel_idx]]
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
        def on_gui_ready(sender, gui):

            self.view = gui.get_view('WaveformClusteringView')

            @controller.supervisor.actions.add(shortcut='f')
            def next_channel():
                v=gui.get_view('WaveformClusteringView')
                if v:
                    v.setNextChannelIdx()

            @controller.supervisor.actions.add(shortcut='r')
            def prev_channel():
                v=gui.get_view('WaveformClusteringView')
                if v:
                    v.setPrevChannelIdx()

        @connect
        def on_select_channel(sender,channel_id,key,button):
            self.view.setChannel(channel_id)


