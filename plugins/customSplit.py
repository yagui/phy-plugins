from phy import IPlugin, connect


def k_means(x,N):
    """Cluster an array into N subclusters, using the K-means algorithm."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=N).fit_predict(x)


class kmSplitPlugin(IPlugin):
    def attach_to_controller(self, controller):

        self.N = 2

        @connect
        def on_gui_ready(sender, gui):

            @controller.supervisor.actions.add(alias='cs',shortcut='s')
            def custom_split():
                """Split using the K-means clustering algorithm on the template amplitudes
                of the first cluster."""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)

                # We get the spike ids and the corresponding spike template amplitudes.
                # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = bunchs[0].spike_ids
                y = bunchs[0].amplitudes

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                labels = k_means(y.reshape((-1, 1)),self.N)
                print(labels)
                assert spike_ids.shape == labels.shape

                # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)


            @controller.supervisor.actions.add(alias='kmn',shortcut='alt+k',prompt=True, prompt_default=lambda: self.N)
            def setKMeansN(N):
                """Set number of clusters to use for KMeans."""
                if type(N) is int:
                    if N>1:
                        self.N = N
                        gui.status_message = 'KMeans N set to {a:d}'.format(a=self.N)
                    else:
                        gui.status_message = 'Error: N should be an Integer greater than 1'
                elif type(N) is str:
                    try:
                        a = int(N)
                        if a>1:
                            self.N = a
                            gui.status_message = 'KMeans N set to {a:d}'.format(a=self.N)
                        else:
                            gui.status_message = 'Error: N should be an Integer'
                    except:
                        gui.status_message = 'Error: input is not a number'
                else:
                    print(type(N))
