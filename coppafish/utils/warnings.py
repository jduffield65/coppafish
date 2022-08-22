class NotebookPageWarning(Warning):
    def __init__(self, page):
        """

        :param page: string e.g. 'find_spots'
        """
        message = "Notebook already contains page: "
        self.message = message + page

    def __str__(self):
        return self.message
