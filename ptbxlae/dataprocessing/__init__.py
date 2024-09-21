import wfdb
import matplotlib.pyplot as plt
import ecg_plot


def load_single_record(id: int = 1, lowres: bool = True):
    id_str = "{:05d}".format(id)
    prefix_dir = f"{id_str[:2]}000"

    if lowres:
        return wfdb.rdsamp(f"../data/records100/{prefix_dir}/{id_str}_lr")
    else:
        return wfdb.rdsamp(f"../data/records500/{prefix_dir}/{id_str}_hr")


def plot_raw_data(sig, sigmeta, savefile: str = None):
    ecg_plot.plot(
        sig.transpose()[:, :250], sample_rate=sigmeta["fs"], columns=4, title=None
    )
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    if savefile:
        ecg_plot.save_as_png(savefile)
    else:
        ecg_plot.show()


def plot_single_record(id: int = 1, lowres: bool = True, savefile: str = None):
    sig, sigmeta = load_single_record(id=id, lowres=lowres)
    plot_raw_data(sig, sigmeta)
