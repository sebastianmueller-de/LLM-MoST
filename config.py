class SessionConfig:
    USE_REPAIR_MODULE = True

    @classmethod
    def set_use_repair_module(cls, value: bool):
        cls.USE_SPECIAL_MODULE = value

    @classmethod
    def get_use_repair_module(cls) -> bool:
        return cls.USE_SPECIAL_MODULE
