from timeit import default_timer


class Timer(object):
    """ class written to time various actions in the scraper without having to awkwardly insert 4-5 lines of code
    at near-random places for each timer, and to allow for better timing
    :param name: String, name for the timer to help with identification when using multiple timers
    """

    def __init__(self, name):
        self.name = name
        self._start_timer = None
        self._stop_timer = None

    def start(self):
        self._start_timer = default_timer()

    def stop(self):
        self._stop_timer = default_timer()

    @property
    def total_running_time_short(self):
        running_time = self._stop_timer - self._start_timer
        return self.name+" : %.3f" % running_time
    # try adding functions to time parts of the loop off of this one timer

    @property
    def total_running_time_long(self):
        running_time = self._stop_timer - self._start_timer
        minutes, seconds = divmod(running_time, 60)
        hours, minutes = divmod(minutes, 60)
        return "Running time: %d:%d:%d. \n" % (hours, minutes, seconds)