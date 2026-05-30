from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = r''
    settings.got10k_lmdb_path = r'G:\UniMod1K-main\SPT\data\got10k_lmdb'
    settings.got10k_path = r'G:\UniMod1K-main\SPT\data\got10k'
    settings.got_packed_results_path = r''
    settings.got_reports_path = r''
    settings.lasot_lmdb_path = r'G:\UniMod1K-main\SPT\data\lasot_lmdb'
    settings.lasot_path = r'G:\UniMod1K-main\SPT\data\lasot'
    settings.network_path = r'G:\UniMod1K-main\SPT\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = r'G:\UniMod1K-main\SPT\data\nfs'
    settings.otb_path = r'G:\UniMod1K-main\SPT\data\OTB2015'
    settings.prj_dir = r'G:\UniMod1K-main\SPT'
    settings.result_plot_path = r'G:\UniMod1K-main\SPT\test/result_plots'
    settings.results_path = r'G:\UniMod1K-main\SPT\test/tracking_results'    # Where to store tracking results
    settings.save_dir = r'G:\UniMod1K-main\SPT'
    settings.segmentation_path = r'G:\UniMod1K-main\SPT\test/segmentation_results'
    settings.tc128_path = r'G:\UniMod1K-main\SPT\data\TC128'
    settings.tn_packed_results_path = r''
    settings.tpl_path = r''
    settings.trackingnet_path = r'G:\UniMod1K-main\SPT\data\trackingNet'
    settings.uav_path = r'G:\UniMod1K-main\SPT\data\UAV123'
    settings.vot_path = r'G:\UniMod1K-main\SPT\data\VOT2019'
    settings.youtubevos_dir = r''
    settings.unimod1k_path = r"D:\sequences_RGBD1K" 

    return settings

