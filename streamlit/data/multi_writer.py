class MultiWriter:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for w in self.writers:
            w.write(text)
            w.flush()

    def flush(self):
        for w in self.writers:
            w.flush()