class cached:
    """
    Decorator that caches the returned value of a propetry and updates the cache
    and when a new value is set to this property.

    This decorator internally creates a private attribute `_decorated_cache`, which
    is used as a cache, represented by a dictionnay.
    """

    def __init__(self, func):
        self.func = func
        self.local_func_cache_name = f"_{func.__name__}_cache"
        self.global_object_cache_name = "_decorated_cache"

    def __get__(self, instance, owner):
        """Called when a property is read."""
        if instance is None:
            return self
        self.create_cache_if_needed(instance)

        # If in cache, we return the cached value
        if self.is_in_cache(instance):
            return self.read_cache(instance)

        # Else, we compute the function and cache it
        value = self.func(instance)
        self.set_cache(instance, value)

        return value

    def __set__(self, instance, value):
        """Called when a property is updated via a setter."""
        self.create_cache_if_needed(instance)

        # Call a specific setter method if it exists, otherwise directly update
        setter_method_name = f"set_{self.func.__name__}"
        if hasattr(instance, setter_method_name):
            setter = getattr(instance, setter_method_name)
            setter(value)
        else:
            self.set_cache(instance, value)

    def read_cache(self, instance):
        cache = getattr(instance, self.global_object_cache_name)
        return cache[self.local_func_cache_name]

    def set_cache(self, instance, value):
        cache = getattr(instance, self.global_object_cache_name)
        cache[self.local_func_cache_name] = value

    def is_in_cache(self, instance):
        cache_exists = hasattr(instance, self.global_object_cache_name)
        is_in_cache = self.local_func_cache_name in getattr(
            instance, self.global_object_cache_name
        )
        return cache_exists and is_in_cache

    def create_cache_if_needed(self, instance):
        cache_exists = hasattr(instance, self.global_object_cache_name)
        if not cache_exists:
            setattr(instance, self.global_object_cache_name, {})
