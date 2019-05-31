import logging
import six
import pandas as pd

class LogSelector(object):
    def __init__(self, path, log):
        self.path = path
        self.log = log

    def select(self, name):
        return LogSelector(self.path + [name], self.log)

    def debug(self, k, v):
        if self.level_is_at_least(logging.DEBUG):
            self.log.notify(logging.DEBUG, self.path + [k], v)

    def info(self, k, v):
        if self.level_is_at_least(logging.INFO):
            self.log.notify(logging.INFO, self.path, k, v)

    def __getattr__(self, name):
        return LogSelector(self.path + [name], self.log)

    def __getitem__(self, name):
        return LogSelector(self.path + [name], self.log)

    def __repr__(self):
        d = self.data()
        if hasattr(d, 'values'):
            val = six.next(six.itervalues(d))
            if isinstance(val, dict):
                return repr(d.keys())
        return repr(d)

    def __iter__(self):
        return iter([LogSelector(self.path + [key], self.log) for key in self.data()])

    def __len__(self):
        return len(self.data())

    def data(self):
        current = self.log.data
        for i, p in enumerate(self.path):
            if p == '_':
                prefix = self.path[:i]
                suffix = self.path[i+1:]
                stuff = [(k, LogSelector(prefix + [k] + suffix, self.log).data()) for k,v in current.items()]
                return dict(stuff)
            elif p == '__':
                prefix = self.path[:i]
                suffix = self.path[i+1:]
                stuff = []
                for k1, v in current.items():
                    m = LogSelector(prefix + [k1] + suffix, self.log).data()
                    for k2 in m:
                        stuff.append(((k1,k2),m[k2]))
                return dict(stuff)
            else:
                current = current[p]
        return current

    def series(self):
        return pd.Series(self.data())

    def df(self):
        d = self.data()
        if isinstance(list(d.values())[0], dict):
            return pd.DataFrame(d)
        else:
            return pd.DataFrame({self.path[-1]: d})

    def level_is_at_least(self, level):
        return self.log.level <= level

    def concat(self, f, axis=1):
        k = self.data().keys()
        return pd.concat([f(x) for x in iter(self)], axis=axis, keys=k)

class Log(object):
    def __init__(self, name=None):
        self.data = {}
        self.level = logging.INFO
        self.listeners = []

    def select(self, name):
        return LogSelector([name], self)

    def info(self, k, v):
        return LogSelector([], self).info(k,v)

    def debug(self, k, v):
        return LogSelector([], self).info(k,v)

    def df(self):
        return LogSelector([], self).df()

    def series(self):
        return LogSelector([], self).series()

    def __getattr__(self, name):
        return LogSelector([name], self)

    def __repr__(self):
        return repr(self.data.keys())

    def __iter__(self):
        return iter([LogSelector([key], self) for key in self.data])

    def concat(self, f, axis=1):
        return LogSelector([], self).concat(f, axis)

    def notify(self, level, path, k, v):
        if level >= logging.INFO:
            current = self.data
            for p in path:
                if p not in current:
                    current[p] = {}
                current = current[p]
            if k in current:
                raise Exception("%s is overriding log entry!" % (self.path + [k]))
            current[k] = v
        for listener in self.listeners:
            listener.notify(level, path, k, v)
