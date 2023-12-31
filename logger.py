# logger.py
import logging
import os
class Logger:
    def __init__(self, log_name, output_dir):

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # create file handler which logs even info messages
        fh = logging.FileHandler(os.path.join(output_dir, log_name), 'w')

        fh.setLevel(logging.INFO)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger