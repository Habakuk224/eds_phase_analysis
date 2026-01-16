import hyperspy.api as hs
import numpy as np
import pickle
import argparse as ap
import h5py
import os
import sys

from hdbscan import HDBSCAN

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QSpinBox, QLabel, QDialog, QGridLayout, QPushButton, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mendeleev import element
import element_cmap
import csv


FIELDS = ('file', 'path', 'comp', 'size_original', 'size_binned', 'size_valid', 'live_time (s)', 'px_dwell (us)', 'phase_id', 'phase_points', 'phase_live_time (s)')
plt.rcParams['figure.constrained_layout.use'] = True
element_cmap.prep_elemental_colormaps()


class SliderSpinbox(QWidget):
    """
    A combined slider + spinbox widget.
    """

    value_changed = Signal(float)

    def __init__(self, label, vmin, vmax, initial, step, parent=None):
        super().__init__(parent)

        self._vmin = vmin
        self._vmax = vmax
        self._step = step

        layout = QVBoxLayout(self)
        if label:
            layout.addWidget(QLabel(label))

        # Spinbox
        self.spin = QSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setSingleStep(1)
        self.spin.setValue(initial)

        # Slider (integer scale for smooth movement)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(vmin, vmax)
        self.slider.setValue(initial)

        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)

        # Synchronization flags
        self._updating = False

        # Connect slider <-> spinbox
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, val):
        if self._updating:
            return
        self._updating = True

        self.spin.setValue(val)
        self.value_changed.emit(val)
        self._updating = False

    def _from_spin(self, val):
        if self._updating:
            return
        self._updating = True

        self.slider.setValue(val)
        self.value_changed.emit(val)
        self._updating = False

    @property
    def value(self):
        """Current numeric value."""
        return self.spin.value()

    @value.setter
    def value(self, val):
        """Set value programmatically."""
        val = max(self._vmin, min(self._vmax, val))
        self.spin.setValue(val)


class InfoWindow(QDialog):
    def __init__(self, edsmap, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Elemental EDS")
        layout = QVBoxLayout(self)

        self.setMinimumSize(1200, 800)
        self.setModal(False)  # non-blocking, user can close independently

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.canvas)

        # Initial plot
        edsmap.plot_eds(self.fig)
        self.canvas.draw_idle()


class MainClusterDialog(QDialog):
    """
    2x3 layout dialog:
      - Rightmost column: 4 SliderSpinboxes + 4 buttons below
      - Left/middle columns: reserved for plots or other widgets
    """

    def __init__(self, map):
        super().__init__()
        self.setWindowTitle(f"Clustering: {map.barefile} — field {map.comp}")
        self.map = map
        self.user_closed = False

        self.resize(1500, 800)
        main_layout = QHBoxLayout(self)

        # --- Left / middle columns (can hold plots) ---
        grid_layout = QGridLayout()
        self.figs = []
        self.canvases = []
        for row in range(2):
            for col in range(2):
                fig = Figure()
                canvas = FigureCanvas(fig)

                self.canvases.append(canvas)
                self.figs.append(fig)

                grid_layout.addWidget(canvas, row, col)

        main_layout.addLayout(grid_layout, stretch=4)

        # --- Right column: sliders + buttons ---
        right_col = QGridLayout()

        # Four SliderSpinboxes
        self.sliders = []
        slider_paremeters = [{"label": "Components", "vmin" : 2, "vmax": 6, "initial" : self.map.cl_params["components"]},
                             {"label": "Min. cluster size", "vmin" : 1, "vmax": self.map.eds.metadata.get_item('size_binned'), "initial" : self.map.cl_params["min_cluster_size"]},
                             {"label": "Min. samples", "vmin" : 2, "vmax": 200, "initial" : self.map.cl_params["min_samples"]},
                             {"label": "Cutoff", "vmin" : 0, "vmax": 1000, "initial" : self.map.cl_params["cutoff"]}]

        for i, pars in enumerate(slider_paremeters):
            s = SliderSpinbox(**pars, step=1)
            self.sliders.append(s)
            right_col.addWidget(s, 0, i)

        # Four buttons below sliders
        btn_decompose = QPushButton("Decompose")
        btn_cluster = QPushButton("Cluster")
        btn_another = QPushButton("Elements")
        btn_save = QPushButton("Save")
        self.chbx_fov = QCheckBox("Use Field of View")

        btn_decompose.clicked.connect(lambda: self._on_button("decompose"))
        btn_cluster.clicked.connect(lambda: self._on_button("cluster"))
        btn_another.clicked.connect(lambda: self._on_button("elements"))
        btn_save.clicked.connect(lambda: self._on_button("save"))  # Save closes dialog

        right_col.addWidget(btn_decompose, 1, 0)
        right_col.addWidget(btn_cluster, 1, 1)
        right_col.addWidget(btn_another, 1, 2)
        right_col.addWidget(btn_save, 1, 3)

        right_col.addWidget(self.chbx_fov, 2, 0, 1, 4)

        main_layout.addLayout(right_col, stretch=1)

        self.eds_window = None  # keep reference

        # Initial plot
        self._update_plot()

    def _on_button(self, action):
        """Triggered by any of the four buttons"""

        self.result_action = action

        self.map.cl_params["components"] = self.sliders[0].value
        self.map.cl_params["min_cluster_size"] = self.sliders[1].value
        self.map.cl_params["min_samples"] = self.sliders[2].value
        self.map.cl_params["cutoff"] = self.sliders[3].value
        self.map.cl_params["use_fov"] = self.chbx_fov.isChecked()

        # Call model functions depending on button
        if action == "decompose":
            self.map.decompose_phases()
            self.map.cluster_phases()

        elif action == "cluster":
            self.map.cluster_phases()

        elif action == "elements":
            if self.eds_window is None:
                self.eds_window = InfoWindow(self.map, parent=self)
            self.eds_window.show()
            self.eds_window.raise_()  # bring to front
            self.eds_window.activateWindow()  # focus

        elif action == "save":
            if self.eds_window is not None:
                self.eds_window.close()

            self.accept()

        self._update_plot()

    def _update_plot(self):
        """Redraw plot using current parameters or last computation"""

        for fig in self.figs:
            fig.clear()

        ax_fov = self.figs[0].add_subplot()
        ax_fov.imshow(self.map.fov, cmap='Greys_r')
        ax_fov.axis('off')

        self.map.phase_map_plot(self.map.phase_map_valid, self.figs[1])

        ax_dec = self.figs[2].add_subplot(projection='3d')
        ax_dec.view_init(elev=30, azim=45, roll=0)

        subsample = np.random.choice(range(self.map.eds.isig[0].data.size), min(self.map.eds.isig[0].data.size, 10000))

        if self.map.dec_loads.shape[0] == 1:
            self.map.dec_loads = np.vstack((self.map.dec_loads[0, :],
                                        np.zeros_like(self.map.dec_loads[0, :]),
                                        np.zeros_like(self.map.dec_loads[0, :])))
        elif self.map.dec_loads.shape[0] == 2:
            self.map.dec_loads = np.vstack((self.map.dec_loads[:2, :],
                                        np.zeros_like(self.map.dec_loads[0, :])))

        ax_dec.scatter(self.map.dec_loads[0, subsample],
                       self.map.dec_loads[1, subsample],
                       self.map.dec_loads[2, subsample],
                       c=self.map.phase_map_valid.flatten()[subsample],
                       s=self.map.dec_loads_sum[subsample],
                       norm=self.map.norm,
                       cmap=self.map.cmap,
                       marker='.'
                       )

        ax_tree = self.figs[3].add_subplot()
        self.map.cluster_tree.plot(select_clusters=True,
                                   selection_palette=self.map.cmap(self.map.norm(self.map.ph_order_desc_inv[1:])),
                                   axis=ax_tree)
        for canvas in self.canvases:
            canvas.draw_idle()

    def closeEvent(self, event):
        self.user_closed = True
        event.accept()

def process_EDSatlas(fname, h5_path, element_list = None, binning = 1, quiet = False):
    
    f = h5py.File(fname, "r")
    atlas = f[h5_path]

    atlas_pars = []

    app = QApplication.instance()  # check if QApplication exists
    if app is None:
        app = QApplication(sys.argv)

    for i, comp in enumerate(atlas.keys()):
        print(h5_path+'/'+comp)
        mapa = EDSmap(fname, h5_path+'/'+comp, element_list)
        result, cl_params = mapa.process(binning, quiet)
        atlas_pars.extend(result)
           
    with open(f"{fname}.csv",'w', newline='', encoding='utf-8') as fcsv:
        wr = csv.writer(fcsv, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(FIELDS)
        wr.writerows(atlas_pars)


def process_EDSmap(fname, h5_path, element_list = None, binning = 1, quiet = False):

    app = QApplication.instance()  # check if QApplication exists
    if app is None:
        app = QApplication(sys.argv)

    eds_map = EDSmap(fname, h5_path, element_list)
    result, cl_params = eds_map.process(binning, quiet)

    with open(f"{fname}_{eds_map.comp}.csv",'w', newline='', encoding='utf-8') as fcsv:
        wr = csv.writer(fcsv, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerow(FIELDS)
        wr.writerows(result)


class EDSmap:
    """
    Class representing a single EDS map.
       
    """
    
    def __init__(self, fname, h5_path, element_list = None):
        
        # load data
        self.load_from_edax_h5(fname, h5_path, element_list)

        # if previous settings were saved, load them, assign defaults otherwise
        try:
            fpickle = open(os.path.join(self.filedir, self.barefile, f"{self.comp}_cl_params.pickle"), 'rb')
        except FileNotFoundError:
            self.cl_params = {"min_samples" : 4,
                              "min_cluster_size" : 200,
                              "cutoff": 50,
                              "components": 3,
                              "use_fov": False}
        else:
            with fpickle:
                self.cl_params = pickle.load(fpickle)
                
    
    def rebin(self, binning):
        
        self.eds = self.eds.rebin(scale=[binning, binning, 1])
        eds_size_binned = self.eds.isig[0].data.size
        self.eds.metadata.set_item("size_binned", eds_size_binned)

        fov_size = self.fov.data.size
        if fov_size > eds_size_binned:
            fov_binning = int(np.sqrt(fov_size / eds_size_binned))
            self.fov = self.fov.rebin(scale=[fov_binning, fov_binning])
        
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time",
            self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.get_item('live_time') / binning**2)

        self.eds.metadata.Acquisition_instrument.SEM.set_item("Detector.EDS.pixel_dwell",
            round(self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.get_item('live_time') / self.eds.metadata.get_item('size_binned'),6))

        self.eds.metadata.Acquisition_instrument.SEM.set_item('pixel_x', self.eds.metadata.Acquisition_instrument.SEM.get_item('pixel_x') * binning)
        self.eds.metadata.Acquisition_instrument.SEM.set_item('pixel_y', self.eds.metadata.Acquisition_instrument.SEM.get_item('pixel_y') * binning)
        
    def _xray_lines_cmap_list(self):
        xray_lines = self.eds.metadata.Sample.xray_lines
        cmap_list = ["cmap_" + line.split("_")[0] for line in xray_lines]
        return cmap_list
    
    
    def process(self, binning, quiet, dead_time = 0.3):
        
        # rebinning to improve phase discrimination
        if binning != 1:
            self.rebin(binning)

        self.decompose_phases()
        self.cluster_phases()

        if not quiet:
            dlg = MainClusterDialog(self)
            dlg.exec()

            if dlg.user_closed:
                sys.exit(0)

        # export phase spectra to msa files
        result = self.export_phase_spectra(dead_time)

        # export all elemental maps
        self.export_eds()
        
        # write individual results to CSV file
        with open(os.path.join(self.filedir, self.barefile, f"{self.comp}.csv"),'w', newline='', encoding='utf-8') as fcsv:
            wr = csv.writer(fcsv, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(FIELDS)
            wr.writerows(result)
        
        return result, self.cl_params
    

    def load_from_edax_h5(self, fname, h5_path, element_list = None):
        f = h5py.File(fname, "r")
        
        livemap_path = '/Live Map 1/'

        spd_dts =   f[h5_path + livemap_path + 'SPD']
        fov_dts =   f[h5_path + '/FOVIMAGE']
        meta_host = f[h5_path + livemap_path + 'HOSTPARAMS']
        meta_map =  f[h5_path + livemap_path + 'MAPIMAGEIPR']
        meta_spc =  f[h5_path + livemap_path + 'SPC']

        spd_raw = spd_dts[()]
        self.eds = hs.signals.Signal1D(spd_raw)
        print(f"EDS of size {self.eds.isig[0].data.shape} loaded")

        fov_raw = fov_dts[()]
        self.fov = hs.signals.BaseSignal(np.reshape(fov_raw, (fov_dts.attrs["PixelHeight"][0],fov_dts.attrs["PixelWidth"][0]))).T
        print(f"FoV of size {self.fov.data.shape} loaded")

        self.eds.set_signal_type("EDS_SEM")
        self.eds.change_dtype("float32")

        # manual reading and assignment of metadata
        self.eds.metadata.set_item('Sample.description', meta_spc["SpectrumLabel"][0])
        self.eds.metadata.set_item("comp_number", h5_path.rsplit('/',1)[-1])
        self.eds.metadata.set_item("size_original", spd_raw[:,:,0].size)
        self.eds.metadata.set_item("size_binned", self.eds.isig[0].data.size)
        
        self.eds.metadata.Acquisition_instrument.SEM.set_item("beam_energy", meta_host["KV"][0])
        self.eds.metadata.Acquisition_instrument.SEM.set_item("beam_current", meta_host["BeamCurrent"][0])
        self.eds.metadata.Acquisition_instrument.SEM.set_item("magnification", meta_host["Magnification"][0])
        self.eds.metadata.Acquisition_instrument.SEM.set_item("working_distance", meta_host["WD"][0])
        
        self.eds.metadata.Acquisition_instrument.SEM.set_item("pixel_x", meta_map["MicronsPerPixelX"][0])
        self.eds.metadata.Acquisition_instrument.SEM.set_item("pixel_y", meta_map["MicronsPerPixelY"][0])
        self.eds.metadata.Acquisition_instrument.SEM.set_item("pixel_x_units", "um")
        self.eds.metadata.Acquisition_instrument.SEM.set_item("pixel_y_units", "um")
        
        self.eds.metadata.Acquisition_instrument.SEM.Stage.set_item("rotation", meta_host["Rotation"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Stage.set_item("tilt_alpha", meta_host["Tilt"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Stage.set_item("x", meta_host["StageXPosition"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Stage.set_item("y", meta_host["StageYPosition"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Stage.set_item("z", meta_host["StageZPosition"][0])

        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("azimuth_angle", meta_spc["AzimuthAngle"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("elevation_angle", meta_spc["ElevationAngleActual"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", meta_spc["LiveTime"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("energy_resolution_MnKa", meta_spc["DetectorResoultion"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch", meta_spc["evPch"][0])
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch_units", "eV")
        self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("pixel_dwell", round(meta_spc["LiveTime"][0] / self.eds.metadata.get_item('size_binned'), 6))


        # axes calibration
        self.eds.axes_manager[0].name = 'x'
        self.eds.axes_manager[0].units = 'um'
        self.eds.axes_manager[0].scale = meta_map["MicronsPerPixelX"][0]

        self.eds.axes_manager[1].name = 'y'
        self.eds.axes_manager[1].units = 'um'
        self.eds.axes_manager[1].scale = meta_map["MicronsPerPixelY"][0]

        self.eds.axes_manager[-1].name = 'E'
        self.eds.axes_manager['E'].units = 'keV'
        self.eds.axes_manager['E'].scale = self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.eVpch / 1000.   # eV per channel

        if element_list == None:
            # use elements from file
            z_list = meta_spc["AtomicNumberOfPeakIds"][0,:]
            z_list = z_list[z_list != 0].tolist()

            element_list = [element(z_list[i]).symbol for i in range(len(z_list))]

        self.eds.add_elements(element_list)
        self.eds.add_lines(lines = (), only_one = False)

        self.file = fname
        self.filedir, self.basename = os.path.split(fname)
        self.barefile = self.basename.removesuffix('.edaxh5').removesuffix(".h5")

        self.path = h5_path
        self.comp = self.eds.metadata.get_item('comp_number')

        f.close()

    def export_eds(self):
        eds_maps = self.eds.get_lines_intensity()
        cmap_list = self._xray_lines_cmap_list()

        print(os.path.join(self.filedir, self.barefile, self.comp))
        os.makedirs(os.path.join(self.filedir, self.barefile, self.comp), exist_ok=True)

        for i, eds_map in enumerate(eds_maps):
            line = eds_map.metadata.Sample.xray_lines[0]
            px = eds_map.metadata.Acquisition_instrument.SEM.pixel_x
            plt.imsave(os.path.join(self.filedir, self.barefile, self.comp, f"{line}_px{px:.3g}um.tiff"), eds_map, cmap=cmap_list[i])


    def plot_eds(self, fig):

        eds_maps = self.eds.get_lines_intensity()
        cmap_list = self._xray_lines_cmap_list()

        hs.plot.plot_images(eds_maps,
                            axes_decor='off',
                            suptitle = "",
                            per_row = 4,
                            cmap = cmap_list,
                            fig = fig)

    
    def decompose_phases(self):
        # phase decomposition

        self.eds.decomposition(algorithm="NMF",
                                output_dimension = self.cl_params["components"],
                                max_iter = 200)
                
        self.dec_loads = np.array(np.reshape(self.eds.get_decomposition_loadings(), (self.cl_params["components"], -1)))
        self.dec_loads_sum = np.sum(self.dec_loads, axis = 0)
        # self.dec_loads = self.dec_loads / self.dec_loads_sum
        # self.dec_loads = np.concatenate(( self.dec_loads, np.atleast_2d(self.dec_loads_sum)), axis = 0)
        
        # add FoV it to the decomposed signals

        fov_eds_factor = (self.fov.data.shape[0] / self.eds.isig[0].data.shape[0],
                          self.fov.data.shape[1] / self.eds.isig[0].data.shape[1])

        if self.cl_params["use_fov"] & (fov_eds_factor != (1,1)):

            # rebin FoV if the sizes are compatible
            try:
                fov_rebinned = self.fov.rebin(scale=fov_eds_factor)
                self.to_cluster = np.concatenate((np.reshape(fov_rebinned.data, (fov_rebinned.data.size, 1)), self.dec_loads.T),
                                                 axis=1)
            except:
                print("Cannot rebin FoV into EDS shape, FoV is NOT used.")
                self.to_cluster = self.dec_loads.T
        else:
            self.to_cluster = self.dec_loads.T

        
    def cluster_phases(self):
        
        # do cluster analysis either from decomposed signals, or from all signals (FoV included) and plot the result
        
        # hdbs = HDBSCAN(allow_single_cluster= True,
        #                cluster_selection_method= 'leaf',
        #                min_cluster_size = self.cl_params["min_cluster_size"],
        #                min_samples = self.cl_params["min_samples"],
        #                metric = 'seuclidean',
        #                metric_params={'V' : self.to_cluster.var(axis=0)},
        #                n_jobs = -1)

        hdbs = HDBSCAN(allow_single_cluster= True,
                       cluster_selection_method= 'eom',
                       min_cluster_size = self.cl_params["min_cluster_size"],
                       min_samples = self.cl_params["min_samples"],
                       metric = 'seuclidean',
                       core_dist_n_jobs = -1,
                       V = self.to_cluster.var(axis=0)
        )

        clusterer = hdbs.fit(self.to_cluster)
        self.cluster_tree = clusterer.condensed_tree_

        # number of identified phases
        self.n_phases = np.max(hdbs.labels_) + 1
        
        # create map of phase indices
        phase_map_raw = hdbs.labels_.reshape(self.eds.isig[0].data.shape)

        self.ph_num_pts = np.zeros(self.n_phases, dtype="int")        
        ph_spc_raw = np.zeros((self.n_phases, self.eds.data.shape[-1]))

        # points with -1 are invalid
        valid_mask_map = (phase_map_raw != -1)
        
        for i in range(self.n_phases):
            # current phase mask
            mask = (phase_map_raw == i)
            
            # filter points belonging to current phase, minor included
            self.ph_num_pts[i] = int(np.sum(mask))
            ph_spc_raw[i,:] = self.eds.data[mask,:].sum(0)
            
            # add to final mask only if number of points is larger than "cutoff"
            if self.ph_num_pts[i] < self.cl_params["cutoff"]:
                valid_mask_map[mask] = False

        # sort phases based on number of points and reindex phase map
        self.ph_order_desc       = np.argsort(self.ph_num_pts, kind = 'stable')[::-1]
        self.ph_order_desc_inv   = np.insert(np.argsort(self.ph_order_desc),0,-1)
        self.ph_num_pts     = self.ph_num_pts[self.ph_order_desc]
        ph_spc_raw          = ph_spc_raw[self.ph_order_desc,:]
        self.phase_map      = self.ph_order_desc_inv[phase_map_raw+1]

        # cutoff minor phases
        valid_mask_ph = (self.ph_num_pts > self.cl_params["cutoff"])
        self.ph_num_pts_clustered   = int(np.sum(self.ph_num_pts))              # clustered points
        self.ph_num_pts_valid       = int(np.sum(self.ph_num_pts[valid_mask_ph])) # valid points        
        self.phase_map_valid = np.where(valid_mask_map, self.phase_map, -1) # filetered phase map with invalid points as -1
        self.n_phases_valid = np.max(self.phase_map_valid) + 1 # number of valid phases (after cutoff)


        print("Phases:", self.ph_num_pts)
        print("Total / Clustered / Valid:", self.eds.metadata.get_item('size_binned'), self.ph_num_pts_clustered, self.ph_num_pts_valid)
        print()

        # preparation of phase colormap
        base = plt.cm.Set1.colors*(self.n_phases_valid % 9 + 1)  # preserve Set1 indexing
        self.cmap = ListedColormap(base[:self.n_phases_valid])
        self.cmap.set_under('k')

        self.bounds = np.arange(-0.5, self.n_phases_valid + 0.5, 1.0)
        self.norm = BoundaryNorm(self.bounds, self.cmap.N)


        self.ph_spc = hs.signals.Signal1D(ph_spc_raw)
        self.ph_spc_total = hs.signals.Signal1D(self.eds.sum((0,1)))
        self.ph_spc_valid = hs.signals.Signal1D(np.sum(ph_spc_raw[valid_mask_ph,:], axis=0))
        
        for s in [self.ph_spc, self.ph_spc_total, self.ph_spc_valid]:
            s.set_signal_type("EDS_SEM")
            s.metadata.add_dictionary(self.eds.metadata.as_dictionary())
            
            s.axes_manager[-1].name = 'E'
            s.axes_manager['E'].units = 'keV'
            s.axes_manager['E'].scale = self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.eVpch / 1000.   # eV per channel


    def phase_map_plot(self, data, fig):
        """
        Plot phase map with proper colormap.

        -1 -- invalid points
        0--N -- phases

        Parameters
        ----------
        data : 2D array
            DESCRIPTION.

        Returns
        -------
        im : AxesImage
            DESCRIPTION.

        """

        axs = fig.subplots(1, 2, width_ratios=(15,1))
        axs[1].set_aspect(self.n_phases_valid + 1)

        # if np.max(data) != np.min(data):
        im = axs[0].imshow(data, cmap=self.cmap, norm=self.norm)

        cbar = fig.colorbar(im,
                            cax=axs[1],
                           ticks = np.arange(0, self.n_phases_valid + 1),
                           extend="min")
        axs[1].minorticks_off()
        axs[1].set_title("Phase")

        # else:
        #     im = axs[0].imshow(data, cmap='Set1')

        axs[0].axis('off')

        return im


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

        live_time = self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time
        size_original = self.eds.metadata.get_item('size_original')
        size_binned = self.eds.metadata.get_item('size_binned')
        px_dwell = self.eds.metadata.Acquisition_instrument.SEM.Detector.EDS.get_item('pixel_dwell')

        print(self.eds.metadata)

        result_pars = []
        
        spc_name = os.path.join(self.filedir, self.barefile, f"{self.comp}_ph_total.msa")
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
        
        spc_name = os.path.join(self.filedir, self.barefile, f"{self.comp}_ph_valid.msa")
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
         
            spc_name = os.path.join(self.filedir, self.barefile, f"{self.comp}_ph_{str(i)}.msa")
            
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
            
        fig1 = plt.figure(1)
        self.phase_map_plot(self.phase_map_valid, fig1)
        fig1.savefig(os.path.join(self.filedir, self.barefile, f"{self.comp}.png"), bbox_inches='tight', pad_inches=0)
        fig1.suptitle(f"{self.barefile}_{str(self.comp)}")
        fig1.savefig(os.path.join(self.filedir, self.barefile, f"{self.comp}_t.png"), bbox_inches='tight', pad_inches=0)
        
        plt.figure(2)
        plt.imshow(self.fov, cmap='Greys_r')
        plt.axis('off')
        plt.savefig(os.path.join(self.filedir, self.barefile, f"{self.comp}_fov.png"), bbox_inches='tight', pad_inches=0)
        plt.title(f"{self.barefile}_{str(self.comp)}_fov")
        plt.savefig(os.path.join(self.filedir, self.barefile, f"{self.comp}_t_fov.png"), bbox_inches = 'tight', pad_inches=0)
        
        with open(os.path.join(self.filedir, self.barefile, f"{self.comp}_cl_params.pickle"), 'wb') as fpickle:
            pickle.dump(self.cl_params, fpickle)
        
        plt.close("all")
        return result_pars

if __name__ == "__main__":
    
    parser = ap.ArgumentParser(prog = "EDS phase clustering tool",
                               description = "A tool for phase clustering of EDAX EDS maps, using Non-negative Matrix Factorization for signal decomposition and HDBSCAN for clustering.",
                               usage = "eds_phanal.py [-h] filename h5path [-a | -m] [-e ELEMENTS [ELEMENTS ...]] [-b BINNING] [-q]")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--atlas', action = 'store_true', help = "Process all maps within a single H5 group.")
    group.add_argument('-m', '--map',   action = 'store_true', help = "Process a single map.")
    
    parser.add_argument('filename', help = "Path to the data file")
    parser.add_argument('h5path', help = "Path within the H5 file to the map (--map) or to the group of maps (--atlas)")

    parser.add_argument('-e', '--elements', nargs = '+', help = "List of chemical element symbols to be used; the list from H5 file is used if not provided expicitly.")
    parser.add_argument('-b', '--binning', type = int, default=1, help = "Spatial binning")
    parser.add_argument('-q', '--quiet', action = 'store_true', help = "Does not open GUI, uses previously saved parameters from processing.")

    args = parser.parse_args()
    
    if args.atlas:
        process_EDSatlas(args.filename, args.h5path.rstrip('/'), args.elements, args.binning, args.quiet)
        
    elif args.map:
        process_EDSmap(args.filename, args.h5path.rstrip('/'), args.elements, args.binning, args.quiet)

