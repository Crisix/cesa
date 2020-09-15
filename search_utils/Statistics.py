import time


class Timer:
    def __init__(self):
        self.duration = 0.
        self.last_start = None

    def resume(self):
        if self.last_start is not None:
            raise ValueError("timer already running")
        self.last_start = time.time()

    def pause(self):
        self.duration += time.time() - self.last_start
        self.last_start = None
        return self.duration

    def is_paused(self):
        return self.last_start is None

    def __enter__(self):
        self.resume()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pause()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # return str(datetime.timedelta(seconds=self.duration))
        return f"{self.duration:.2f}"


class Statistics:
    def __init__(self, original_sentence):
        self.tried_examples = 0
        self.tried_sentences = None
        self.total_sentences = None

        self.total_duration = Timer()
        self.merging_duration = Timer()
        self.find_matching_words_duration = Timer()

        self.original_sentence = original_sentence
        self.original_classification = None
        self.query = None

    def all_timers_stopped(self):
        return self.merging_duration.is_paused() and \
               self.total_duration.is_paused() and \
               self.find_matching_words_duration.is_paused()

    def add(self, stats):
        self.tried_examples += stats.tried_examples
        self.merging_duration.duration += stats.merging_duration.duration
        self.find_matching_words_duration.duration += stats.find_matching_words_duration.duration
