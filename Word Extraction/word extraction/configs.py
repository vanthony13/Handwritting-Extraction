import stow
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = "C:\\Users\\Vitoria\\Desktop\\mltu-main (copy)\\Word Extraction\\word extraction\\Models\\1_image_to_word\\model"
        self.vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝŸñß¿¡.,;:!?áàâãéèêíïóôõöúüçíóúÁÀÂÃÉÈÊÍÏÓÔÕÖÚÜÇÍÓÚ"
        self.height = 96
        self.width = 1408
        self.max_text_length = 0
        self.batch_size = 32
        self.learning_rate = 0.001
        self.train_epochs = 100
        self.train_workers = 20
        self.checkpoint_path = stow.join(self.model_path, datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"), "checkpoint")
