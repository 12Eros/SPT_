class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = r''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + r'/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_models = self.workspace_dir + r'/pretrained_models/'
        self.lasot_dir = r''
        self.got10k_dir = r''
        self.trackingnet_dir = r''
        self.coco_dir = r''
        self.lvis_dir = r''
        self.sbd_dir = r''
        self.imagenet_dir = r''
        self.imagenetdet_dir = r''
        self.ecssd_dir = r''
        self.hkuis_dir = r''
        self.msra10k_dir = r''
        self.davis_dir = r''
        self.youtubevos_dir = r''
        self.unimod1k_dir = r'G:\UniMod1K-main\SPT\data\RGBD1K_train_labelled'
        self.unimod1k_dir_nlp = r'G:\UniMod1K-main\SPT\data\nlps_train'
