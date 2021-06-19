import logging

_logger = logging.getLogger(__name__)

def log_experiment_settings(settings, is_test=False):
    _logger.info('')
    if is_test:
        _logger.info('=== TEST SETTINGS ===')
    else:
        _logger.info('=== TRAIN SETTINGS ===')

    for key, value in settings.items():
        _logger.info(f'{key}={value}')
    _logger.info('')
