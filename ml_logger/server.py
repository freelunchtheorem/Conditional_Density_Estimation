import os
# todo: switch to dill instead
import pickle
from collections import namedtuple
from params_proto import cli_parse, Proto, BoolFlag

from ml_logger.serdes import deserialize, serialize
import numpy as np

LogEntry = namedtuple("LogEntry", ['key', 'data', 'type'])
LoadEntry = namedtuple("LogEntry", ['key', 'type'])
ALLOWED_TYPES = (np.uint8,)  # ONLY uint8 is supported.


class LoggingServer:
    def __init__(self, data_dir):
        assert os.path.isabs(data_dir)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        print('logging data to {}'.format(data_dir))

    configure = __init__

    def serve(self, port):
        from japronto import Application
        self.app = Application()
        self.app.router.add_route('/', self.log_handler, method='POST')
        self.app.router.add_route('/', self.read_handler, method='GET')
        # todo: need a file serving url
        self.app.run(port=port, debug=Params.debug)

    def read_handler(self, req):
        load_entry = LoadEntry(**req.json)
        print("loading: {} type: {}".format(load_entry.key, load_entry.type))
        res = self.load(load_entry.key, load_entry.type)
        data = serialize(res)
        return req.Response(text=data)

    def log_handler(self, req):
        log_entry = LogEntry(**req.json)
        print("writing: {} type: {}".format(log_entry.key, log_entry.type))
        data = deserialize(log_entry.data)
        self.log(log_entry.key, data, log_entry.type)
        return req.Response(text='ok')

    def load(self, key, dtype):
        if dtype == 'read':
            abs_path = os.path.join(self.data_dir, key)
            try:
                with open(abs_path, 'rb') as f:
                    return f.decode('utf-8')
            except FileNotFoundError as e:
                return e
        elif dtype == 'read_text':
            abs_path = os.path.join(self.data_dir, key)
            try:
                with open(abs_path, 'r') as f:
                    return f.decode('utf-8')
            except FileNotFoundError as e:
                return e
        elif dtype == 'read_pkl':
            abs_path = os.path.join(self.data_dir, key)
            try:
                with open(abs_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError as e:
                return e
        elif dtype == 'read_np':
            abs_path = os.path.join(self.data_dir, key)
            try:
                with open(abs_path, 'rb') as f:
                    return np.load(f)
            except FileNotFoundError as e:
                return e
        elif dtype == 'read_image':
            raise NotImplemented('reading images is not implemented.')

    def log(self, key, data, dtype):
        if dtype == "log":
            abs_path = os.path.join(self.data_dir, key)
            try:
                with open(abs_path, 'ab') as f:
                    pickle.dump(data, f)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(abs_path))
                with open(abs_path, 'ab') as f:
                    pickle.dump(data, f)
        elif dtype.startswith("text"):
            abs_path = os.path.join(self.data_dir, key)
            if "." not in key:
                abs_path = abs_path + ".md"
            try:
                with open(abs_path, "a+") as f:
                    f.write(data)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(abs_path))
                with open(abs_path, "a+") as f:
                    f.write(data)
        elif dtype.startswith("image"):
            abs_path = os.path.join(self.data_dir, key)
            if "." not in key:
                abs_path = abs_path + ".png"
            from PIL import Image
            assert data.dtype in ALLOWED_TYPES, "image datatype must be one of {}".format(ALLOWED_TYPES)
            if len(data.shape) == 3 and data.shape[-1] == 1:
                data.resize(data.shape[:-1])
            im = Image.fromarray(data)
            try:
                im.save(abs_path)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(abs_path))
                im.save(abs_path)
        elif dtype.startswith("video"):
            # todo: mpeg can be appended directly, making it a nice format.
            raise NotImplementedError


@cli_parse
class Params:
    data_dir = Proto("/tmp/logging-server", help="The directory for saving the logs")
    port = Proto(8081, help="port for the logging server")
    debug = BoolFlag(False, help='boolean flag for printing out debug traces')


if __name__ == '__main__':
    import pkg_resources

    v = pkg_resources.get_distribution("ml_logger").version
    print('running ml_logger.server version {}'.format(v))
    server = LoggingServer(data_dir=Params.data_dir)
    server.serve(port=Params.port)
