
import ctypes as c
def TOPOINTER(c_type):
    tp = c.POINTER(c_type)
    def get(self, item):
        return self[0].__getattribute__(item)
    def set(self, key, value):
        self[0].__setattr__(key, value)
    def rep(self):
        return self[0].__repr__()
    tp.__getattribute__ = get
    tp.__setattr__ = set
    tp.__repr__ = rep
    return tp

