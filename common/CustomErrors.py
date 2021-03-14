class TrainError(Exception):
    # store image id that caused the error
    def __init__(self, image_ids, message):
        self.image_ids = image_ids
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'Error caused in training loop by image_ids {self.image_ids}\n' \
               f'Stack Trace:\n{self.message}'


class ValidationError(Exception):
    # store image id that caused the error
    def __init__(self, image_ids, message):
        self.image_ids = image_ids
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'Error caused in validation loop by image_ids {self.image_ids}\n' \
               f'Stack Trace:\n{self.message}'

