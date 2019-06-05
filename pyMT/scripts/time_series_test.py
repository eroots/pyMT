import os
import numpy as np
import struct


MTU5A_DEFINITIONS = {'.TS2': 2400, '.TS3': 2400,
                     '.TS4': 150, '.TS5': 15}
# Some definitions for precision types:
# uint8 is a single byte ('H')
# uint16 is a short ('Q')


def read_24bit_data(bites):
    return bites[0] | (bites[1] << 8) | (bites[2] << 16)


class TimeSeries(object):
    def __init__(self, **kwargs):
        self.instrument_type = ''  # Type of instrument
        self.instrument_number = 0  # Instrument serial number
        self.sample_frequency = 0  # Sample rate of file
        self.start_time = 0  # Time of first sample
        self.end_time = 0  # Time of last sample
        self.start_date = 0  # Date of first sample
        self.end_date = 0  # Date of last sample
        self.time_stamps = []  # Time stamps of each record
        self.date_stamps = []  # Date stamps of each record
        self.saturation_flags = []  # Saturation flags for each record
        self.num_chan = 0  # Number of channels for instrument
        self.tag_code = 0  # Tag code
        self.status_code = 0  # Status code
        self.format_type = 0  # Format type
        self.unit_type = 0
        self.clock_status = 0
        self.clock_error = 0
        self.extra_bytes = {str(ii): [] for ii in range(26, 32)}
        self.data = []  # Actual count data
        self.record_samples = []  # Length of each record (samples)
        self.total_length = 0  # Total length of record (seconds)
        self.file_name = ''
        for key in kwargs:
            setattr(self, key, kwargs[key])


def nextpow2(n):
    return (1<<(int(n - 1))).bit_length()


def read_phx(in_file, instrument='MTU5A', hours=1, NFFT=0, notch_freq=60):
    # Set up all the parameters
    instrument = instrument.lower()
    file, ext = os.path.splitext(in_file)
    if not NFFT:
        NFFT = nextpow2(hours * 3600 * 15 / 72)  # Not sure how this got defined...
    if instrument == 'mtu5a':
        # num_channel = 5
        tag_length = 32
        try:
            frequency = MTU5A_DEFINITIONS[ext]
        except KeyError:
            print('File extention {} not recognized'.format(ext))
            print('Exiting...')
            return
    else:
        print('Instrument {} not supported yet'.format(instrument))
        print('Exiting...')
        return
    record_length = hours * 3600
    num_samples = record_length * frequency
    # offset = NFFT / (4 * num_samples)
    # Start reading stuff
    headers = ('UTC1', 'UTC2', 'UTC3', 'UTC4', 'UTC5', 'UTC6', 'UTC7', 'UTC8',
               'instrument_num',
               'sample_freq',
               'num_chan',
               'tag_code',
               'status_code',
               'saturation_flag',
               'format_type',
               'bytes_per_sample',
               'samples_per_time_unit',
               'unit_type',
               'clock_status',
               'clock_error',
               'byte_26',
               'byte_27',
               'byte_28',
               'byte_29',
               'byte_30',
               'byte_31')
    with open(in_file, 'rb+') as f:
        counter = 0
        # f.seek((29 + 3 * (2400 * 5 + 1)) * 2500, 0)
        while True:
            if counter % 100 == 0:
                print('Reading record {}'.format(counter + 1))
            bites = f.read(29)
            if not bites:
                break
            values = struct.unpack('8B2H6BH9B', bites)
            header_dict = dict(zip(headers, values))
            # UTC = np.fromfile(f, dtype=np.uint8, count=8)  # 8 bytes
            # instrument_num = np.fromfile(f, dtype=np.uint16, count=1)[0]  # 2 bytes
            # sample_freq = np.fromfile(f, dtype=np.uint16, count=1)[0]  # 2 bytes
            start_time = str(header_dict['UTC3']) + ':' + \
                         str(header_dict['UTC2']) + ':' + \
                         str(header_dict['UTC1'])
            start_date = str(header_dict['UTC4']) + ':' + \
                         str(header_dict['UTC5']) + ':' + \
                         str(header_dict['UTC6'])
            # num_chan = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            # tag_code = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            # status_code = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            # saturation_flag = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            # if tag_length == 32:
            #     format_type = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     bytes_per_sample = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     samples_per_time_unit = np.fromfile(f, dtype=np.uint16, count=1)[0]  # 2 bytes
            #     unit_type = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     clock_status = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     clock_error = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_26 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_27 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_28 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_29 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_30 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            #     byte_31 = np.fromfile(f, dtype=np.uint8, count=1)[0]  # 1 bytes
            # data = [read_24bit_data(struct.unpack('BBB', x)) for x in f.read(num_chan * sample_freq * 24)]
            data = []
            for ii in range(header_dict['num_chan'] * header_dict['sample_freq'] + 1):
                chunk = f.read(3)
                if ii:
                    data.append(struct.unpack('<i', bytes(1) + chunk)[0] >> 8)
            data = np.reshape(np.array(data), [header_dict['sample_freq'], header_dict['num_chan']])
            # bites = f.read(3 * (header_dict['num_chan'] * header_dict['sample_freq'] + 1))
            # data = array.frombytes()
            # data = list(struct.iter_unpack('<i', bytes(1), + ))
            # data = f.read(num_chan * sample_freq * 24)
            # data = np.fromfile(f, dtype=np.int32, count=num_chan * sample_freq)
            if counter == 0:
                record_length = header_dict['sample_freq'] / header_dict['samples_per_time_unit']
                time_series = TimeSeries(**{'instrument_type ': instrument,
                                            'instrument_number': header_dict['instrument_num'],
                                            'sample_frequency': header_dict['sample_freq'],
                                            'start_time': start_time,
                                            'end_time': [],
                                            'start_date': start_date,
                                            'end_date': [],
                                            'time_stamps ': [],
                                            'saturation_flags ': [],
                                            'num_chan': header_dict['num_chan'],
                                            'tag_code': header_dict['tag_code'],
                                            'status_code': header_dict['status_code'],
                                            'format_type': header_dict['format_type'],
                                            'unit_type': header_dict['unit_type'],
                                            'clock_status': [],
                                            'clock_error': [],
                                            'data ': [],
                                            'file_name': in_file,
                                            'record_length': []})
            time_series.time_stamps.append(start_time)
            time_series.date_stamps.append(start_date)
            time_series.record_length.append(len(data))
            time_series.saturation_flags.append(header_dict['saturation_flag'])
            time_series.clock_status.append(header_dict['clock_status'])
            time_series.clock_error.append(header_dict['clock_error'])
            time_series.extra_bytes['26'].append(header_dict['byte_26'])
            time_series.extra_bytes['27'].append(header_dict['byte_27'])
            time_series.extra_bytes['28'].append(header_dict['byte_28'])
            time_series.extra_bytes['29'].append(header_dict['byte_29'])
            time_series.extra_bytes['30'].append(header_dict['byte_30'])
            time_series.extra_bytes['31'].append(header_dict['byte_31'])
            # time_series.data = data
            time_series.data.append(data)
            counter += 1
    time_series.num_records = counter
    time_series.total_length = counter * header_dict['sample_freq'] / header_dict['samples_per_time_unit']
    return time_series, header_dict
