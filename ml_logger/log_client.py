import os
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession
from ml_logger.serdes import serialize, deserialize
from ml_logger.server import LogEntry, LoadEntry, LoggingServer, ALLOWED_TYPES


class LogClient:
    local_server = None
    prefix = None

    def __init__(self, url: str = None, prefix=None, max_workers=None):
        if url.startswith("file://"):
            self.local_server = LoggingServer(data_dir=url[6:])
        elif os.path.isabs(url):
            self.local_server = LoggingServer(data_dir=url)
        elif url.startswith('http://'):
            self.url = url
        else:
            # todo: add https://, and s3://
            raise TypeError('log url need to begin with `/`, `file://` or `http://`.')
        if prefix:
            assert not os.path.isabs(prefix), "prefix can not have leading slash`/`"
            self.prefix = prefix
        if max_workers:
            self.session = FuturesSession(ThreadPoolExecutor(max_workers=max_workers))
        else:
            self.session = FuturesSession()

    def _get(self, key, dtype):
        if self.local_server:
            return self.local_server.load(key, dtype)
        else:
            # todo: make the json serialization more robust. Not priority b/c this' client-side.
            json = LoadEntry(key, dtype)._asdict()

            # note: reading stuff from the server is always synchronous via the result call.
            res = self.session.get(self.url, json=json).result()
            result = deserialize(res.text)
            return result

    def _post(self, key, data, dtype):
        # todo: make this asynchronous
        if self.local_server:
            self.local_server.log(key, data, dtype)
        else:
            # todo: make the json serialization more robust. Not priority b/c this' client-side.
            json = LogEntry(key, serialize(data), dtype)._asdict()
            self.session.post(self.url, json=json)
            # todo: verify request success. Otherwise looses data.

    # Reads binary data
    def read(self, key):
        return self._get(key, dtype="read")

    # Reads binary data
    def read_pkl(self, key):
        return self._get(key, dtype="read_pkl")

    def read_np(self, key):
        return self._get(key, dtype="read_np")

    # appends data
    def log(self, key, data):
        self._post(key, data, dtype="log")

    # appends text
    def log_text(self, key, text):
        self._post(key, text, dtype="text")

    # sends out images
    def send_image(self, key, data):
        assert data.dtype in ALLOWED_TYPES, "image data must be one of {}".format(ALLOWED_TYPES)
        self._post(key, data, dtype="image")
