import hyperspy.api as hs
import numpy as np
import pickle
import argparse as ap
import h5py

from sklearn.cluster import HDBSCAN

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.colors import LinearSegmentedColormap

from mendeleev import element
import element_cmap
import csv


FIELDS = ('file', 'path', 'comp', 'size_original', 'size_binned', 'size_valid', 'live_time (s)', 'px_dwell (us)', 'phase_id', 'phase_points', 'phase_live_time (s)')

phase_cmap = element_cmap.prep_phase_colormap()
element_cmap.prep_elemental_colormaps()

def phase_map_plot(data, ax):
    """
    Plot phase map with proper colormap.
    
    -1 -- invalid points
    0--N -- phases

    Parameters
    ----------
    data : 2D array
        DESCRIPTION.
    ax : axis
        DESCRIPTION.

    Returns
    -------
    im : AxesImage
        DESCRIPTION.

    """
    
    # select subset of colormap, according to the number of phases
    cmap_cur = LinearSegmentedColormap.from_list('cmap_cur', phase_cmap.colors[np.min(data)+1 : np.max(data)+2], np.max(data)-np.min(data)+1)
    
    if np.max(data) != np.min(data):
        im = ax.imshow(data, cmap = cmap_cur, vmin= np.min(data) - 0.5, vmax=np.max(data) + 0.5)
        plt.colorbar(im,
                      ticks=np.arange(np.min(data), np.max(data) + 1),
                      shrink = 0.75,
                      aspect = 15)
    else:
        im = ax.imshow(data, cmap = 'Set1')

    return im


def process_EDSatlas(fname, h5_path, element_list = None, binning = 1, use_fov = False, quiet = False):
    
    f = h5py.File(fname, "r")
    atlas = f[h5_path]

    atlas_pars = []

    for i, comp in enumerate(atlas.keys()):
        print(h5_path+'/'+comp)
        mapa = EDSmap(fname, h5_path+'/'+comp, element_list)
        result, cl_params = mapa.process(binning, use_fov, quiet)
        atlas_pars.extend(result)
           
    with open(fname+'.csv','w', newline='', encoding='utf-8') as fcsv:
        wr = csv.writer(fcsv, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(FIELDS)
        wr.writerows(atlas_pars)

def process_EDSmap(fname, h5_path, element_list = None, binning = 1, use_fov = False, quiet = False):
    eds_map = EDSmap(fname, h5_path, element_list)
    eds_map.process(binning, use_fov, quiet)


class EDSmap:
    """
    Class representing a single EDS map.
       
    """
    
    def __init__(self, fname, h5_path, element_list = None):
        
        # load data
        self.load_from_edax_h5(fname, h5_path, element_list)

        # if previous settings were saved, load them, assign defaults otherwise
        try:
            fpickle = open(self.barefile + "/" + str(self.comp) + "_cl_params.pickle", 'rb')
        except FileNotFoundError:
            self.cl_params = {"min_samples" : 4,
                              "min_cluster_size" : 200,
                              "cutoff": 50,
                              "components": 3}
        else:
            with fpickle:
                self.cl_params = pickle.load(fpickle)
                
    
    def rebin(self, binning):
        
        self.eds = self.eds.rebin(scale=[binning, binning, 1])
        self.fov = self.fov.rebin(scale=[binning, binning])
        
        self.eds.metadata.set_item("size_binned", self.eds.isig[0].data.size)
        self.sem_meta.Detector.EDS.set_item("live_time",
            self.sem_meta.Detector.EDS.get_item('live_time') / binning**2)
        self.sem_meta.set_item("Detector.EDS.pixel_dwell",
            round(self.sem_meta.Detector.EDS.get_item('live_time') / self.eds.metadata.get_item('size_binned'),6))

        
    def _xray_lines_cmap_list(self):
        xray_lines = self.eds.metadata.Sample.xray_lines
        cmap_list = ["cmap_" + line.split("_")[0] for line in xray_lines]
        return cmap_list
    
    
    def process(self, binning, use_fov, quiet, dead_time = 0.3):
        self.use_fov = use_fov
        
        # rebinning to improve phase discrimination
        if (binning != 1):
            self.rebin(binning)
        
        if quiet:
            # do not show GUI and use the saved parameters
            self.decompose_phases()
            self.cluster_phases()
        else:            
            # run decomposition (do-while "decompose")
            repeat = "decompose"
            while repeat == "decompose":
                self.decompose_phases()
            
                # run interactive ID, plot result (do-while "cluster")
                repeat = "cluster"
                while repeat == "cluster":
                    self.cluster_phases()
                    # show GUI and wait for button
                    repeat = self.cluster_gui()

        result = self.export_phase_spectra(dead_time)
        
        # write individual results to CSV file
        with open(self.barefile+"/"+str(self.comp)+'.csv','w', newline='', encoding='utf-8') as fcsv:
            wr = csv.writer(fcsv, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(FIELDS)
            wr.writerows(result)
        
        return result, self.cl_params
    

    def load_from_edax_h5(self, fname, h5_path, element_list = None):
        f = h5py.File(fname, "r")
        
        livemap_path = '/Live Map 1/'

        spd_dts =   f[h5_path + livemap_path + 'SPD']
        meta_host = f[h5_path + livemap_path + 'HOSTPARAMS']
        meta_map =  f[h5_path + livemap_path + 'MAPIMAGEIPR']
        meta_spc =  f[h5_path + livemap_path + 'SPC']

        spd_raw = spd_dts[()]
        self.eds = hs.signals.Signal1D(spd_raw)
        
        fov_dts = f[h5_path + '/FOVIMAGE']
        fov_raw = fov_dts[()].reshape(self.eds.isig[0].data.shape)
        self.fov = hs.signals.BaseSignal(fov_raw)
        
        self.eds.set_signal_type("EDS_SEM")
        self.eds.change_dtype("float32")

        # manual reading and assignment of metadata
        self.eds.metadata.set_item('Sample.description', meta_spc["SpectrumLabel"][0])
        self.eds.metadata.set_item("comp_number", h5_path.rsplit('/',1)[-1])
        self.eds.metadata.set_item("size_original", spd_raw[:,:,0].size)
        self.eds.metadata.set_item("size_binned", self.eds.isig[0].data.size)
        
        # abbreviation for the nasty long path within the metadata tree
        self.sem_meta = self.eds.metadata.Acquisition_instrument.SEM
        
        self.sem_meta.set_item("beam_energy", meta_host["KV"][0])
        self.sem_meta.set_item("beam_current", meta_host["BeamCurrent"][0])
        self.sem_meta.set_item("magnification", meta_host["Magnification"][0])
        self.sem_meta.set_item("working_distance", meta_host["WD"][0])
        
        self.sem_meta.set_item("pixel_x", meta_map["MicronsPerPixelX"][0])
        self.sem_meta.set_item("pixel_y", meta_map["MicronsPerPixelY"][0])
        self.sem_meta.set_item("pixel_x_units", "um")
        self.sem_meta.set_item("pixel_y_units", "um")
        
        self.sem_meta.Stage.set_item("rotation", meta_host["Rotation"][0])
        self.sem_meta.Stage.set_item("tilt_alpha", meta_host["Tilt"][0])
        self.sem_meta.Stage.set_item("x", meta_host["StageXPosition"][0])
        self.sem_meta.Stage.set_item("y", meta_host["StageYPosition"][0])
        self.sem_meta.Stage.set_item("z", meta_host["StageZPosition"][0])        

        self.sem_meta.Detector.EDS.set_item("azimuth_angle", meta_spc["AzimuthAngle"][0])
        self.sem_meta.Detector.EDS.set_item("elevation_angle", meta_spc["ElevationAngleActual"][0])
        self.sem_meta.Detector.EDS.set_item("live_time", meta_spc["LiveTime"][0])
        self.sem_meta.Detector.EDS.set_item("energy_resolution_MnKa", meta_spc["DetectorResoultion"][0])
        self.sem_meta.Detector.EDS.set_item("eVpch", meta_spc["evPch"][0])
        self.sem_meta.Detector.EDS.set_item("eVpch_units", "eV")
        self.sem_meta.Detector.EDS.set_item("pixel_dwell", round(meta_spc["LiveTime"][0] / self.eds.metadata.get_item('size_binned'), 6))
        
        # axes calibration
        self.eds.axes_manager[0].name = 'x'
        self.eds.axes_manager[0].units = 'um'
        self.eds.axes_manager[0].scale = meta_map["MicronsPerPixelX"][0]

        self.eds.axes_manager[1].name = 'y'
        self.eds.axes_manager[1].units = 'um'
        self.eds.axes_manager[1].scale = meta_map["MicronsPerPixelY"][0]

        self.eds.axes_manager[-1].name = 'E'
        self.eds.axes_manager['E'].units = 'keV'
        self.eds.axes_manager['E'].scale = self.sem_meta.Detector.EDS.eVpch / 1000.   # eV per channel

        if element_list == None:
            # use elements from file
            z_list = meta_spc["AtomicNumberOfPeakIds"][0,:]
            z_list = z_list[z_list != 0].tolist()

            element_list = [element(z_list[i]).symbol for i in range(len(z_list))]

        self.eds.add_elements(element_list)
        self.eds.add_lines(lines = (), only_one = False)

        self.file = fname
        self.barefile = fname.rstrip('.edaxh5')
        self.path = h5_path
        self.comp = self.eds.metadata.get_item('comp_number')

        f.close()
    
    def plot_eds(self):
        eds_maps = self.eds.get_lines_intensity()       
        
        fig_eds= plt.figure(figsize = (12,9))
        hs.plot.plot_images(eds_maps,
                            axes_decor='off',
                            tight_layout = True,
                            suptitle = "",
                            per_row = 4,
                            cmap = self._xray_lines_cmap_list(),
                            fig = fig_eds)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        
        fig_eds.show()
    
    def decompose_phases(self):
        # phase decomposition
        
        self.eds.decomposition(algorithm = "sklearn_pca",
                                output_dimension = 8)
        
        self.eds.pca_variance = self.eds.get_explained_variance_ratio()
        
        self.eds.decomposition(algorithm="NMF",
                                output_dimension = self.cl_params["components"],
                                max_iter = 200)
                
        self.dec_loads = np.array(np.reshape(self.eds.get_decomposition_loadings(), (self.cl_params["components"], -1)))
        self.dec_loads_sum = np.sum(self.dec_loads, axis = 0)
        self.dec_loads = self.dec_loads / self.dec_loads_sum
        self.dec_loads = np.concatenate(( self.dec_loads, np.atleast_2d(self.dec_loads_sum)), axis = 0)
        
        # add FoV it to the decomposed signals
        if self.use_fov:
            self.to_cluster = np.concatenate((np.reshape(self.fov.data, (self.fov.data.size, 1)), self.dec_loads.T), axis=1)
        else:
            self.to_cluster = self.dec_loads.T

        
    def cluster_phases(self):
        
        # do cluster analysis either from decomposed signals, or from all signals (FoV included) and plot the result
        
        hdbs = HDBSCAN(allow_single_cluster= True,
                       cluster_selection_method= 'eom',
                       min_cluster_size = self.cl_params["min_cluster_size"],
                       min_samples = self.cl_params["min_samples"],
                       metric = 'seuclidean',
                       metric_params={'V' : self.to_cluster.var(axis=0)},
                       n_jobs = -1)
                  
        hdbs.fit(self.to_cluster)
        
        # number of identified phases
        self.n_phases = np.max(hdbs.labels_) + 1
        
        # create map of phase indices
        phase_map_raw = hdbs.labels_.reshape(self.eds.isig[0].data.shape)
               
        self.ph_num_pts = np.zeros(self.n_phases, dtype="int")        
        ph_spc_raw = np.zeros((self.n_phases, self.eds.data.shape[-1]))
        
        # all points start as invalid
        valid_mask_map = np.zeros_like(phase_map_raw, dtype = bool)
        
        for i in range(self.n_phases):
            # current phase mask
            mask = (phase_map_raw == i)
            
            # filter points belonging to current phase, minor included
            self.ph_num_pts[i] = int(np.sum(mask))
            ph_spc_raw[i,:] = self.eds.data[mask,:].sum(0)
            
            # add to final mask only if number of points is larger than "cutoff"
            if (self.ph_num_pts[i] > self.cl_params["cutoff"]):
                valid_mask_map = np.logical_or(valid_mask_map, mask)
        
            
        # sort phases based on number of points and reindex phase map
        ph_order_desc       = np.argsort(-self.ph_num_pts, kind = 'stable')
        ph_order_desc_inv   = np.argsort(ph_order_desc)
        self.ph_num_pts     = self.ph_num_pts[ph_order_desc]
        ph_spc_raw          = ph_spc_raw[ph_order_desc,:]
        self.phase_map      = ph_order_desc_inv[phase_map_raw]        
        
        # cutoff minor phases
        valid_mask_ph = (self.ph_num_pts > self.cl_params["cutoff"])
        self.ph_num_pts_clustered   = int(np.sum(self.ph_num_pts))              # clustered points
        self.ph_num_pts_valid       = int(np.sum(self.ph_num_pts[valid_mask_ph])) # valid points        
        self.phase_map_valid = np.where(valid_mask_map, self.phase_map, -1) # filetered phase map with invalid points as -1
        
        print("Phases:", self.ph_num_pts)
        print("Total / Clustered / Valid:", self.eds.metadata.get_item('size_binned'), self.ph_num_pts_clustered, self.ph_num_pts_valid)
        print()
        
        self.ph_spc = hs.signals.Signal1D(ph_spc_raw)
        self.ph_spc_total = hs.signals.Signal1D(self.eds.sum((0,1)))
        self.ph_spc_valid = hs.signals.Signal1D(np.sum(ph_spc_raw[valid_mask_ph,:], axis=0))
        
        for s in [self.ph_spc, self.ph_spc_total, self.ph_spc_valid]:
            s.set_signal_type("EDS_SEM")
            s.metadata.add_dictionary(self.eds.metadata.as_dictionary())
            
            s.axes_manager[-1].name = 'E'
            s.axes_manager['E'].units = 'keV'
            s.axes_manager['E'].scale = self.sem_meta.Detector.EDS.eVpch / 1000.   # eV per channel
    
    def cluster_gui(self):
        
        def save_cl_params():
            self.cl_params["min_cluster_size"] = sl_cls.val
            self.cl_params["min_samples"] = sl_smp.val
            self.cl_params["cutoff"] = sl_cut.val
            self.cl_params["components"] = sl_comp.val
        
        def recluster(event):
            self.repeat = "cluster"
            save_cl_params()
            plt.close("all")
        
        def redecompose(event):
            self.repeat = "decompose"
            save_cl_params()
            plt.close("all")
        
        def eds(event):
            self.plot_eds()
            
        def go_on(event):
            self.repeat = None
            plt.close("all")
        
        fig, ax = plt.subplots(2,3, figsize=(18, 9), gridspec_kw={'width_ratios': [2, 2, 1]})
        
        phase_map_plot(self.phase_map_valid, ax[0,0])
            
        ax[0,0].axis('off')        
        ax[0,1].imshow(self.fov, cmap='Greys_r')   
        ax[0,1].axis('off')
        
        ax[1,0].remove()
        ax[1,0] = fig.add_subplot(2,3,4,projection='3d')
        ax[1,0].view_init(elev=30, azim=45, roll=0)
        subsample = np.random.choice(range(self.eds.isig[0].data.size),min(self.eds.isig[0].data.size, 10000))
        
        if self.dec_loads.shape[0] == 1:
            self.dec_loads = np.vstack( (self.dec_loads[0,:],
                                         np.zeros_like(self.dec_loads[0,:]),
                                         np.zeros_like(self.dec_loads[0,:])) )
        elif self.dec_loads.shape[0] == 2:
            self.dec_loads = np.vstack( (self.dec_loads[:2,:],
                                         np.zeros_like(self.dec_loads[0,:])) )
                                       
        ax[1,0].scatter(self.dec_loads[0,subsample], 
                      self.dec_loads[1,subsample], 
                      self.dec_loads[2,subsample], 
                      c = self.phase_map_valid.flatten()[subsample], 
                       cmap = (LinearSegmentedColormap.from_list('cmap_cur',
                                                                phase_cmap.colors[np.min(self.phase_map_valid)+1:np.max([np.max(self.phase_map_valid)+2],0)],
                                                                np.max([np.max(self.phase_map_valid)-np.min(self.phase_map_valid)+1],0)) if np.max(self.phase_map_valid)!=np.min(self.phase_map_valid) else 'Set1'),
                      marker = '.',
                      s = self.dec_loads_sum[subsample]
                      )
        ax[1,1].plot(self.eds.pca_variance, 'o', ms = 4)
        ax[0,2].remove()
        ax[1,2].remove()
        
        
        # hard cutoff - phases with less then "hard cutoff" points will not be exported
        ax_cut = fig.add_axes([0.77, 0.15, 0.03, 0.75])
        sl_cut = Slider(
            ax=ax_cut,
            label="hard cutoff",
            orientation="vertical",
            valmin=10,
            valmax=1000,
            valinit=self.cl_params["cutoff"],
            valstep=1
        )
        
        # control elements of clustering parameters
        ax_cls = fig.add_axes([0.83, 0.15, 0.03, 0.75])
        sl_cls = Slider(
            ax=ax_cls,
            label='min_cluster_size',
            orientation="vertical",
            valmin=10,
            valmax=self.eds.metadata.get_item('size_binned'),
            valinit=self.cl_params["min_cluster_size"],
            valstep = 10
        )
        
        ax_smp = fig.add_axes([0.89, 0.15, 0.03, 0.75])
        sl_smp = Slider(
            ax=ax_smp,
            label="min_samples",
            orientation="vertical",
            valmin=2,
            valmax=200,
            valinit=self.cl_params["min_samples"],
            valstep=1
        )
        
        # decomposition dimension
        ax_comp = fig.add_axes([0.95, 0.15, 0.03, 0.75])
        sl_comp = Slider(
            ax=ax_comp,
            label="components",
            orientation="vertical",
            valmin=2,
            valmax=6,
            valinit=self.cl_params["components"],
            valstep=1
        )
        
        ax_b1 = fig.add_axes([0.770, 0.05, 0.05, 0.05])
        ax_b2 = fig.add_axes([0.825, 0.05, 0.05, 0.05])
        ax_b3 = fig.add_axes([0.880, 0.05, 0.05, 0.05])
        ax_b4 = fig.add_axes([0.935, 0.05, 0.05, 0.05])
        b1 = Button(ax_b1, 'Cluster', hovercolor='0.975')
        b2 = Button(ax_b2, 'Decompose', hovercolor='0.975')
        b3 = Button(ax_b3, 'Save', hovercolor='0.975')
        b4 = Button(ax_b4, 'Elements', hovercolor='0.975')
        
        b1.on_clicked(recluster)
        b2.on_clicked(redecompose)
        b3.on_clicked(go_on)
        b4.on_clicked(eds)
        fig.subplots_adjust(left=0,right=0.99,top=0.99,bottom=0.0,hspace=0.0,wspace=0.0)  
        ax[1,1].set_position([0.45,0.05,0.3,0.4])
        plt.show()
        
        return self.repeat
        
        
    def export_phase_spectra(self, dead_time = 0):
        """
        Export spectra of all clusters found within map.

        Parameters
        ----------
        dead_time : float
            Estimated dead time of the detector. Used for better calculation of the cluster spectrum live time (default 0)
 
        Returns
        -------
        None.
        """

        live_time = self.sem_meta.Detector.EDS.live_time
        size_original = self.eds.metadata.get_item('size_original')
        size_binned = self.eds.metadata.get_item('size_binned')
        px_dwell = self.sem_meta.get_item('Detector.EDS.pixel_dwell')

        result_pars = []
        
        spc_name = self.barefile+'/'+str(self.comp)+"_ph_total.msa"       
        self.ph_spc_total.save(spc_name, overwrite=True, encoding = 'utf8')
        spc_temp = hs.load(spc_name)
        spc_temp.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", round((1-dead_time) * px_dwell * size_binned, 3))
        spc_temp.save(spc_name, overwrite=True, encoding = 'utf8')
        result_pars.append((self.file,
                            self.path,
                            str(self.comp),
                            size_original,
                            size_binned,
                            int(self.ph_num_pts_valid),
                            live_time,
                            px_dwell*1e6,
                            -1,
                            size_binned,
                            round((1-dead_time) * px_dwell * size_binned,3)))
        
        spc_name = self.barefile+'/'+str(self.comp)+"_ph_valid.msa"          
        self.ph_spc_valid.save(spc_name, overwrite=True, encoding = 'utf8')
        spc_temp = hs.load(spc_name)
        spc_temp.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", round((1-dead_time) * px_dwell * self.ph_num_pts_valid,3))
        spc_temp.save(spc_name, overwrite=True, encoding = 'utf8')
        result_pars.append((self.file,
                            self.path,
                            str(self.comp),
                            size_original,
                            size_binned,
                            int(self.ph_num_pts_valid),
                            live_time,
                            px_dwell*1e6,
                            -1,
                            int(self.ph_num_pts_valid),
                            round((1-dead_time) * px_dwell * self.ph_num_pts_valid,3)))
                
        
        for phase in self.ph_spc:
            i = self.ph_spc.axes_manager.indices[0]
            
            #  do not export phases with less than "cutoff" points
            if self.ph_num_pts[i] < self.cl_params["cutoff"]:
                continue
         
            spc_name = self.barefile+'/'+str(self.comp)+"_ph_"+str(i)+".msa"
            
            phase.save(spc_name, overwrite=True, encoding = 'utf8')
            spc_temp = hs.load(spc_name)
            spc_temp.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", round((1-dead_time) * px_dwell * self.ph_num_pts[i],3))
            spc_temp.save(spc_name, overwrite=True, encoding = 'utf8')
            
            result_pars.append((self.file,
                                self.path,
                                str(self.comp),
                                size_original,
                                size_binned,
                                int(self.ph_num_pts_valid),
                                live_time,
                                px_dwell*1e6,
                                i,
                                int(self.ph_num_pts[i]),
                                round((1-dead_time) * px_dwell * self.ph_num_pts[i],3)))
            
        plt.figure(1)
        phase_map_plot(self.phase_map_valid, plt.gca())
        plt.axis('off')
        plt.title(self.barefile+"_"+str(self.comp))
        plt.savefig(self.barefile+"/"+str(self.comp)+".png", bbox_inches = 'tight')
        
        plt.figure(2)
        plt.imshow(self.fov, cmap='Greys_r')
        plt.axis('off')
        plt.title(self.barefile+"_"+str(self.comp)+"_fov")
        plt.savefig(self.barefile+"/"+str(self.comp)+"_fov.png", bbox_inches = 'tight')
        
        with open(self.barefile + "/" + str(self.comp) + "_cl_params.pickle", 'wb') as fpickle:
            pickle.dump(self.cl_params, fpickle)
        
        plt.close("all")
        return(result_pars)

    def savedata(self):
        with open((self.barefile+"/"+str(self.comp)+"_ph_map.pickle"), 'wb+') as f:
            pickle.dump(self.phase_map, f)

        self.ph_spc.save(self.barefile+"/"+str(self.comp)+"_ph_spc.hspy", overwrite=True)
        self.ph_spc.inav[0].save(self.barefile+"/"+str(self.comp)+"_ph_spc.msa", overwrite=True, encoding = 'utf8')


def export_std_to_msa(fname, current):
    """
    Export standard spectra from *.edaxh5 file to msa files needed for NTSA-II. Path within h5 file is hardcoded

    Parameters
    ----------
    fname : string
        File name containing the standard spectra.
    current : float
        Beam current in nA for proper calibration of the spectrum.

    Returns
    -------
    None.

    """
    
    filesplit = fname.rsplit(sep = '\\', maxsplit = 1)
    filepath = filesplit[0]
    barefile = filesplit[-1].rstrip('.edaxh5')
    
    f = h5py.File(fname, "r")

    atlas = f["/"+barefile+"/"]
    spc_path = 'Area 1/Selected Area 1'
    
    for std in atlas.keys():
        
        dts_host = f["/".join([barefile, std, spc_path, 'HOSTPARAMS'])]
        dts_spc =  f["/".join([barefile, std, spc_path, 'SPC'])]
    
        spc = hs.signals.Signal1D(dts_spc["SpectrumCounts"][0])
        
        spc.set_signal_type("EDS_SEM")
        spc.change_dtype("float32")
    
        # manual reading of metadata
        spc.metadata.set_item("Compound", std)
    
        spc.metadata.Acquisition_instrument.SEM.set_item("beam_energy", dts_host["KV"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("beam_current", dts_host["BeamCurrent"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("magnification", dts_host["Magnification"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("working_distance", dts_host["WD"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("Stage.rotation", dts_host["Rotation"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("Stage.tilt_alpha", dts_host["Tilt"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("Stage.x", dts_host["StageXPosition"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("Stage.y", dts_host["StageYPosition"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("Stage.z", dts_host["StageZPosition"][0])
        spc.metadata.Acquisition_instrument.SEM.set_item("beam_current", current)
        
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("azimuth_angle", dts_spc["AzimuthAngle"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("elevation_angle", dts_spc["ElevationAngleActual"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", dts_spc["LiveTime"][0])

        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("energy_resolution_MnKa", dts_spc["DetectorResoultion"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch", dts_spc["evPch"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch_units", "eV")
        
        spc.axes_manager[-1].name = 'E'
        spc.axes_manager['E'].units = 'keV'
        spc.axes_manager['E'].scale = spc.metadata.Acquisition_instrument.SEM.Detector.EDS.eVpch / 1000.   # eV per channel
        
        print(spc.metadata)
        
        spc.save('/'.join([filepath, barefile]) + "_" + std + ".msa", overwrite=True, encoding = 'utf8')
        
    f.close()


if __name__ == "__main__":
    
    parser = ap.ArgumentParser(prog = "EDS phase clustering tool",
                               description = "A tool for phase clustering of EDAX EDS maps, using Non-negative Matrix Factorization for signal decomposition and HDBSCAN for clustering.",
                               usage = "eds_phanal.py [-h] filename h5path [-a | -m] [-e ELEMENTS [ELEMENTS ...]] [-b BINNING] [-q] [-f]")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--atlas', action = 'store_true', help = "Process all maps within a single H5 group.")
    group.add_argument('-m', '--map',   action = 'store_true', help = "Process a single map.")
    
    parser.add_argument('filename', help = "Path to the data file")
    parser.add_argument('h5path', help = "Path within the H5 file to the map (--map) or to the group of maps (--atlas)")

    parser.add_argument('-e', '--elements', nargs = '+', help = "List of chemical element symbols to be used; the list from H5 file is used if not provided expicitly.")
    parser.add_argument('-b', '--binning', type = int, default=1, help = "Spatial binning")
    parser.add_argument('-q', '--quiet', action = 'store_true', help = "Does not open GUI, uses previously saved parameters from processing.")
    parser.add_argument('-f', '--fov', action = 'store_true', help = "Add SEM signal (FoV) to the decomposed EDS and use it for clustering (may be beneficial when phases are well distinguishable on SEM).")
    
    args = parser.parse_args()
    
    if args.atlas:
        process_EDSatlas(args.filename, args.h5path, args.elements, args.binning, args.fov, args.quiet)
        
    elif args.map:
        process_EDSmap(args.filename, args.h5path, args.elements, args.binning, args.fov, args.quiet)

