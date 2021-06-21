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


def log_epoch_metrics(epoch, logs):
    """
    Log the metrics of a training epoch
    """
    _logger.info('')
    _logger.info(f'--- Epoch {epoch + 1} ---')
    for metric in logs:
        _logger.info(f'{metric}={logs[metric]}')
    _logger.info('')


def get_dataset_length(dataset):
    """
    Returns the length of a dataset represented by a dict.
    """
    first_dict_key = next(iter(dataset))
    length = len(dataset[first_dict_key])
    return length


def get_steps_per_epoch(dataset_length, batch_size):
    """
    Returns the steps per epoch for a given dataset and a batch size
    """
    steps_per_epoch = dataset_length // batch_size
    if steps_per_epoch < 1:
        steps_per_epoch = 1
    return steps_per_epoch
