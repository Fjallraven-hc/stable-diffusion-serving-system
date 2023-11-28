class CustomError(Exception):
    def __init__(self, message="This is a custom error"):
        self.message = message
        super().__init__(self.message)