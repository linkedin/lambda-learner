from typing import Dict


class Singleton(type):
    """A Metaclass for building singleton classes.

    Usage:
    class MySingletonClass(metaclass=Singleton):
        pass
    """

    _instances: Dict[object, object] = {}

    def __call__(klass, *args, **kwargs):
        if klass not in klass._instances:
            klass._instances[klass] = super(Singleton, klass).__call__(*args, **kwargs)
        return klass._instances[klass]
