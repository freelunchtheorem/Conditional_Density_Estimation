import base64
from cloudpickle import cloudpickle


def deserialize(code):
    data = cloudpickle.loads(base64.b64decode(code))
    return data


def serialize(data):
    code = cloudpickle.dumps(data)
    return base64.b64encode(code).decode("utf-8")
