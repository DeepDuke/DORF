#! /usr/bin/python2.7
import os
import logging

class MyLogger:
    def __init__(self, log_file_path='.', log_file_name='test.log', level='DEBUG'):
        # Init logger
        self._logger = logging.getLogger(log_file_name.split('.')[0])
        
        # Create file handler
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
            
        log_file = log_file_path + '/' + log_file_name
        file_handler = logging.FileHandler(log_file, mode='w')  # first clear then rewrite
        
        # Create stream handler
        stream_handler = logging.StreamHandler()
        
        # Set log level
        if level == 'INFO':
            self._logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
            stream_handler.setLevel(logging.INFO)
        elif level == 'DEBUG':
            self._logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        elif level == 'WARNING':
            self._logger.setLevel(logging.WARNING)
            file_handler.setLevel(logging.WARNING)
            stream_handler.setLevel(logging.WARNING)
        elif level == 'ERROR':
            self._logger.setLevel(logging.ERROR)
            file_handler.setLevel(logging.ERROR)
            stream_handler.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.CRITICAL)
            file_handler.setLevel(logging.CRITICAL)
            stream_handler.setLevel(logging.CRITICAL)
        
        # Create Colorful formatter
        format = '[%(levelname)s] %(asctime)s - %(name)s:\n%(message)s\n'
        # grey = "\x1b[38;20m"
        # green = "\x1b[32;20m"
        # yellow = "\x1b[33;20m"
        # red = "\x1b[31;20m"
        # bold_red = "\x1b[31;1m"
        # reset = "\x1b[0m"
        
        # COLORFUL_FORMATS = {
        #     'DEBUG': grey + format + reset,
        #     'INFO': green + format + reset,
        #     'WARNING': yellow + format + reset,
        #     'ERROR': red + format + reset,
        #     'CRITICAL': bold_red + format + reset
        # }

        # formatter = logging.Formatter(COLORFUL_FORMATS[level])
        formatter = logging.Formatter(format)
        
        # Set formatter for handlers
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers into logger
        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)
    
    def INFO(self, message):
        return self._logger.info(message)
    
    def DEBUG(self, message):
        return self._logger.debug(message)
    
    def WARNING(self, message):
        return self._logger.warning(message)

    def ERROR(self, message):
        return self._logger.error(message)
    
    def CRITICAL(self, message):
        return self._logger.critical(message)
    
    