import hyperspy.api as hs
import argparse as ap
import h5py

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

    filesplit = fname.rsplit(sep='\\', maxsplit=1)
    filepath = filesplit[0]
    barefile = filesplit[-1].rstrip('.edaxh5')

    f = h5py.File(fname, "r")

    atlas = f[list(f.keys())[0]]

    def find_spc(name):
        if '/SPC' in name:
            return name

    for std_name, std in atlas.items():

        dts_spc = std[std.visit(find_spc)]
        dts_host = dts_spc.parent['HOSTPARAMS']

        spc = hs.signals.Signal1D(dts_spc["SpectrumCounts"][0])

        spc.set_signal_type("EDS_SEM")
        spc.change_dtype("float32")

        # manual reading of metadata
        spc.metadata.set_item("Compound", std_name)

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
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("elevation_angle",
                                                                      dts_spc["ElevationAngleActual"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("live_time", dts_spc["LiveTime"][0])

        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("energy_resolution_MnKa",
                                                                      dts_spc["DetectorResoultion"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch", dts_spc["evPch"][0])
        spc.metadata.Acquisition_instrument.SEM.Detector.EDS.set_item("eVpch_units", "eV")

        spc.axes_manager[-1].name = 'E'
        spc.axes_manager['E'].units = 'keV'
        spc.axes_manager['E'].scale = spc.metadata.Acquisition_instrument.SEM.Detector.EDS.eVpch / 1000.  # eV per channel

        print(spc.metadata)

        spc.save('/'.join([filepath, barefile]) + "_" + std_name + ".msa", overwrite=True, encoding='utf8')

    f.close()

if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Export standard spectra from *.edaxh5 file to msa files needed for NTSA-II. Path within h5 file is hardcoded")

    parser.add_argument('filename', help="Path to the standard data file")
    parser.add_argument('current', help="current used in nA (for proper calibration)", type = float)

    args = parser.parse_args()

    export_std_to_msa(args.filename, args.current)