import yaml
import json

class Config(object):

    @classmethod
    def parse(clss, fpath):
        with open(fpath, 'r') as data:
            entries = yaml.safe_load(data)
        return clss(entries)

    def __init__(self, entries):
        proc_entries = self._parse(entries)
        self.__dict__.update(proc_entries)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        res = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                v = json.loads(str(v))
            res[k] = v
        return json.dumps(res, indent=2, sort_keys=True)

    @classmethod
    def _parse(clss, entries):
        proc_entries = {}
        for k, v in entries.items():
            proc_entries[k] = clss(v) if type(v) is dict else v
        return proc_entries

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def to_dict(self):
        return json.loads(str(self))

    def to_yaml(self):
        return yaml.safe_dump(self.to_dict())