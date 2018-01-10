class WSFileError(Exception):
    """Summary

    Attributes:
        ID (TYPE): Description
        message (str): Description
        offender (TYPE): Description
    """

    def __init__(self, ID, offender, expected='', extra=''):
        """Summary

        Args:
            ID (TYPE): Description
            offender (TYPE): Description
            expected (str, optional): Description
            extra (str, optional): Description
        """
        self.ID = ID
        self.offender = offender
        self.message = ''
        if ID == 'ext':
            self.message = 'File extention not recognized: {}'.format(offender)
            if expected:
                self.message += '\n Expected: {}'.format(expected)
        elif ID == 'int':
            self.message = 'Error while reading file: \n {}'.format(offender)
        elif ID == 'fnf':
            self.message = 'File not found: {}'.format(offender)
        elif ID == 'fmt':
            self.message = 'Format not recognized: {}'.format(offender)
            if expected:
                self.message += '\n Expected: {}'.format(expected)
        if extra:
            self.message += '\n {}'.format(extra)

        super(WSFileError, self).__init__(self.message)
