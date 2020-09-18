class LearningRateScheduler:
    def __init__(self, lr=0.01, decay=0.95):
        self.lr = lr
        self.decay = decay

    def update(self, n):
        if n > 1:
            self.lr *= self.decay
        return self.lr


class AlsEarlyStopping:
    def __init__(self, n_iter=2, min_epoch=15, anneal_times=4):
        """
        :param n_iter: if error is increasing for n_iter -> stop
        :param min_epoch: don't stop before min_epcoch
        """
        self.n_iter = n_iter
        self.last_value = 0
        self.consecutive_increasing_errors = 0
        self.min_epoch = min_epoch

    def stop(self, epoch, error):
        if epoch >= self.min_epoch:
            if error > self.last_value:
                self.consecutive_increasing_errors += 1
            if self.consecutive_increasing_errors >= self.n_iter:
                return True
        self.last_value = error
        return False


class SgdEarlyStopping:
    def __init__(self, n_iter=2, min_epoch=15, anneal_times=4):
        """
        :param n_iter: if error is increasing for n_iter -> stop
        :param min_epoch: don't stop before min_epcoch
        :param anneal_times : number of times to anneal before stopping to train
        """
        self.n_iter = n_iter
        self.last_value = 0
        self.consecutive_increasing_errors = 0
        self.min_epoch = min_epoch
        self.annealing_counter = 0
        self.anneal_times = anneal_times

    def stop(self, mf, epoch, error):
        if epoch >= self.min_epoch:
            if self.annealing_counter == self.anneal_times:
                return True
            if error > self.last_value:
                self.consecutive_increasing_errors += 1
            if self.consecutive_increasing_errors >= self.n_iter:
                # anneal
                mf.lr.lr *= 0.1
                self.annealing_counter += 1
                self.consecutive_increasing_errors = 0
                print('#' * 50 + 'learning rate annealed')
        self.last_value = error
        return False


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attributes(self, **kwargs):
        self._set_attributes(kwargs)
