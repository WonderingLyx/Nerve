import logging
import os 
import json

class Logger:
    def __init__(self, logger_name:str, log_path:str, mode='a'):
        # create a logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # create a handler to write record
        f_handler = logging.FileHandler(log_path, encoding='utf-8', mode=mode)
        f_handler.setLevel(logging.INFO)

        # create a handler to console
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)

        # define the output format
        formatter = logging.Formatter("%(asctime)s - [%(filename)s -->line:%(lineno)d] - %(levelname)s: %(message)s")
        f_handler.setFormatter(formatter)
        c_handler.setFormatter(formatter)

        # add handler to logger
        self.logger.addHandler(f_handler)
        self.logger.addHandler(c_handler)
    
    def get_logger(self):
        return self.logger


    def get_log_path(self, config: dict) -> str:
        log_dir = config['log_path']
        task_name = config['task_name']
        model_name = config['model_name']

        log_path = os.path.join(log_dir,'model_trained',task_name,model_name,'out.log')

        return log_path
