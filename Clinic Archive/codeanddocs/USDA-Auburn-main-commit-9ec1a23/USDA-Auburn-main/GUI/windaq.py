'''
Created on Febuary, 2019

@author: samper

Windaq class object to work directly with .wdq files
Python 3 Version

Source: https://github.com/sdp8483/windaq3
'''

#!/usr/bin/python
import struct
import datetime
import numpy

class windaq(object):
    '''
    Read windaq files (.wdq extension) without having to convert them to .csv or other human readable text

    Code based on http://www.dataq.com/resources/pdfs/misc/ff.pdf provided by Dataq, code and comments will refer to conventions from this file
    and python library https://www.socsci.ru.nl/wilberth/python/wdq.py that does not appear to support the .wdq files created by WINDAQ/PRO+
    '''

    def __init__(self, filename):
        ''' Define data types based off convention used in documentation from Dataq '''
        UI = "<H" # unsigned integer, little endian
        I  = "<h" # integer, little endian
        B  = "B"  # unsigned byte, kind of redundant but lets keep consistent with the documentation
        UL = "<L" # unsigned long, little endian
        D  = "<d" # double, little endian
        L  = "<l" # long, little endian
        F  = "<f" # float, little endian

        ''' Open file as binary '''
        with open(filename, 'rb') as self._file:
            self._fcontents = self._file.read()

        ''' Read Header Info '''
        if (struct.unpack_from(B, self._fcontents, 1)[0]):                                              # max channels >= 144
            self.nChannels      = (struct.unpack_from(B, self._fcontents, 0)[0])                        # number of channels is element 1
        else:
            self.nChannels      = (struct.unpack_from(B, self._fcontents, 0)[0]) & 31                   # number of channels is element 1 mask bit 5

        self._hChannels     = struct.unpack_from(B,  self._fcontents, 4)[0]                             # offset in bytes from BOF to header channel info tables
        self._hChannelSize  = struct.unpack_from(B,  self._fcontents, 5)[0]                             # number of bytes in each channel info entry
        self._headSize      = struct.unpack_from(I,  self._fcontents, 6)[0]                             # number of bytes in data file header
        self._dataSize      = struct.unpack_from(UL, self._fcontents, 8)[0]                             # number of ADC data bytes in file excluding header
        self.nSample        = (self._dataSize/(2*self.nChannels))                                       # number of samples per channel
        self._trailerSize   = struct.unpack_from(UL, self._fcontents,12)[0]                             # total number of event marker, time and date stamp, and event marker comment pointer bytes in trailer
        self._annoSize      = struct.unpack_from(UI, self._fcontents, 16)[0]                            # total number of usr annotation bytes including 1 null per channel
        self.timeStep       = struct.unpack_from(D,  self._fcontents, 28)[0]                            # time between channel samples: 1/(sample rate throughput / total number of acquired channels)
        e14                 = struct.unpack_from(L,  self._fcontents, 36)[0]                            # time file was opened by acquisition: total number of seconds since jan 1 1970
        e15                 = struct.unpack_from(L,  self._fcontents, 40)[0]                            # time file was written by acquisition: total number of seconds since jan 1 1970
        self.fileCreated    = datetime.datetime.fromtimestamp(e14).strftime('%Y-%m-%d %H:%M:%S')        # datetime format of time file was opened by acquisition
        self.fileWritten    = datetime.datetime.fromtimestamp(e15).strftime('%Y-%m-%d %H:%M:%S')        # datetime format of time file was written by acquisition
        self._packed        = ((struct.unpack_from(UI, self._fcontents, 100)[0]) & 16384) >> 14         # bit 14 of element 27 indicates packed file. bitwise & e27 with 16384 to mask all bits but 14 and then shift to 0 bit place
        self._HiRes         = ((struct.unpack_from(UI, self._fcontents, 100)[0]) & 2)                   # bit 1 of element 27 indicates a HiRes file with 16-bit data

        ''' read channel info '''
        self.scalingSlope       = []
        self.scalingIntercept   = []
        self.calScaling         = []
        self.calIntercept       = []
        self.engUnits           = []
        self.sampleRateDivisor  = []
        self.phyChannel         = []

        for channel in range(0,self.nChannels):
            channelOffset = self._hChannels + (self._hChannelSize * channel)                                        # calculate channel header offset from beginning of file, each channel header size is defined in _hChannelSize
            self.scalingSlope.append(struct.unpack_from(F, self._fcontents, channelOffset)[0])                      # scaling slope (m) applied to the waveform to scale it within the display window
            self.scalingIntercept.append(struct.unpack_from(F,self._fcontents, channelOffset + 4)[0])               # scaling intercept (b) applied to the waveform to scale it withing the display window
            self.calScaling.append(struct.unpack_from(D, self._fcontents, channelOffset + 4 + 4)[0])                # calibration scaling factor (m) for waveform value display
            self.calIntercept.append(struct.unpack_from(D, self._fcontents, channelOffset + 4 + 4 + 8)[0])          # calibration intercept factor (b) for waveform value display
            self.engUnits.append(struct.unpack_from("cccccc", self._fcontents, channelOffset + 4 + 4 + 8 + 8))      # engineering units tag for calibrated waveform, only 4 bits are used last two are null

            if self._packed:                                                                                        #  if file is packed then item 7 is the sample rate divisor
                self.sampleRateDivisor.append(struct.unpack_from(B, self._fcontents, channelOffset + 4 + 4 + 8 + 8 + 6 + 1)[0])
            else:
                self.sampleRateDivisor.append(1)
            self.phyChannel.append(struct.unpack_from(B, self._fcontents, channelOffset + 4 + 4 + 8 + 8 + 6 + 1 + 1)[0])        # describes the physical channel number

        ''' read user annotations '''
        aOffset = self._headSize + self._dataSize + self._trailerSize
        aTemp = ''
        for i in range(0, self._annoSize):
            aTemp += struct.unpack_from('c', self._fcontents, aOffset + i)[0].decode("utf-8")
        self._annotations = aTemp.split('\x00')
        #create a numpy view into the data for efficient reading
        dt = numpy.dtype(numpy.int16)
        dt = dt.newbyteorder('<')
        self.npdata =  numpy.frombuffer(self._fcontents, dtype=dt,count = int(self.nSample*self.nChannels),offset = self._headSize)

        # ------------

        # element 1 is self.nChannels
        # element 5 is self._headSize
        # element 6 is self._dataSize
        # element 7 is self._trailerSize

        # list of event marker dicts
        self.eventmarkers = []

        mOffset = self._headSize + self._dataSize

        # TODO: Not sure if this is the proper way to detect whether the file contains event markers
        # may not need to check: "However, an event marker pointer (at least one will always exist)"
        if self._trailerSize > 0:

            #while mOffset < aOffset:
            while mOffset <= aOffset - 4:

                # TODO: add info to marker, then append it to self.eventmarkers at the end of the loop
                marker = {}

                #print('------------')
                #print(f'mOffset: 0x{mOffset:08x}')

                # each value is a signed long
                event_marker_pointer = struct.unpack_from(L, self._fcontents, offset=mOffset)[0]
                mOffset += 4
                #print(f'event marker pointer: {event_marker_pointer}')

                # TODO
                # *For HiRes files, in Equations 1, 2, and 3 above, "(number of channels acquired)" should be omitted
                # from the equation since WinDaq HiRes acquisition does this multiplication already. Advanced CODAS
                # peak detect adds the channel number minus 1 to the product to indicate the channel number for peaks
                # and valleys, and the stored values for the valleys are negated.

                # Equation 3* IF Event Marker Pointer < 0 THEN Event Marker Pointer = Event Marker Pointer × -1
                # OFFSET = (Event Marker Pointer × (2(number of channels acquired))) + Element 5
                event_marker_pointer_abs = abs(event_marker_pointer)
                event_marker_offset = (event_marker_pointer_abs * 2 * self.nChannels) + self._headSize
                #print(f'event marker offset: {event_marker_offset}')
                # TODO: maybe use this to find index into the numpy arr?

                # TODO: after handling hires, divide by self.nChannels to get this
                event_marker_index = event_marker_pointer_abs
                #print(f'event marker index: {event_marker_index}')

                marker['index'] = event_marker_index

                # 00 The default state of all channels except lowest-numbered acquired channel
                # 01 The default state of lowest-numbered acquired channel**
                # 10 Displays a negative-going marker on any channel's waveform*
                # 11 Displays a positive-going marker on any channel's waveform*
                marker_flag = self.npdata[event_marker_index*self.nChannels] & 0b11
                #print(f'marker flag: 0b{marker_flag:02b}')

                # TODO: Parse this sentence, do we need to do something about it?
                # Event markers enabled during data acquisition are flagged along with the data sample associated with
                # the lowest-numbered acquired channel through use of the two least significant data bits of the waveform
                # data word. Refer to the ADC DATA FORMAT section above.

                # TODO: use event marker pointer to get x-value (time)

                # IF Event Marker Pointer < 0 THEN a Time and Date Stamp DOES NOT exist
                # IF Event Marker Pointer is greater than or equal to 0 THEN a Time and Date Stamp exists
                if event_marker_pointer >= 0:
                    rel_timestamp = struct.unpack_from(L, self._fcontents, offset=mOffset)[0]
                    mOffset += 4
                    abs_timestamp = rel_timestamp + e14
                    marker_timestamp = datetime.datetime.fromtimestamp(abs_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    #print(f'rel_timestamp: {rel_timestamp}')
                    #print(f'marker_timestamp: {marker_timestamp}')

                    marker['timestamp'] = marker_timestamp

                # TODO: Ensure we have enough buffer available to unpack...
                #print(f'BUFFER: mOffset: {mOffset}, aOffset: {aOffset}, buflen: {len(self._fcontents)}')

                # manually check the loop condition again so we don't read too far (jank, i know)
                if mOffset <= aOffset - 4:

                    next_read = struct.unpack_from(L, self._fcontents, offset=mOffset)[0]
                    # Equation 1* IF Long > -1 × (Element 6 ÷ (2(number of channels acquired))) THEN the Long is another Event Marker Pointer
                    # Equation 2* IF Long is less than or equal to -1 × (Element 6 ÷ (2(number of channels acquired))) THEN the Long is an Event Marker Comment Pointer
                    if next_read <= -1 * (self._dataSize / (2 * self.nChannels)): # event marker comment pointer

                        # only advance moffset if it's an event marker comment pointer (inside this if statement)
                        # otherwise, don't advance mOffset so the next iteration reads this value into event_marker_pointer
                        mOffset += 4

                        event_marker_comment_pointer = next_read
                        #print(f'event marker comment pointer: {event_marker_comment_pointer}')

                        # Event Marker Comment Pointer = Event Marker Comment Pointer AND 7FFFFFFFH
                        # OFFSET = Event Marker Comment Pointer + Element 5 + Element 6 + Element 7
                        # "Note that each event marker comment is terminated by a null (00) character."

                        # this just sets the msb to zero, for a signed long (4 bytes) that adds 2**31
                        # could use np.int32 to force a 4-byte int if the python variable int size causes issues
                        event_marker_comment_pointer_masked = event_marker_comment_pointer & 0x7FFFFFFF
                        assert event_marker_comment_pointer_masked == event_marker_comment_pointer + (2**31 if event_marker_comment_pointer < 0 else 0)
                        #print(f'event marker comment pointer masked: {event_marker_comment_pointer_masked}')

                        event_marker_comment_offset = event_marker_comment_pointer_masked + self._headSize + self._dataSize + self._trailerSize
                        #print(f'event marker comment offset: {event_marker_comment_offset}')

                        comment = self._fcontents[event_marker_comment_offset:].split(b'\x00')[0].decode()
                        #print(f'comment: {comment}')

                        marker['comment'] = comment

                self.eventmarkers.append(marker)

    def data(self, channelNumber):
        ''' return the data for the channel requested
            data format is saved CH1tonChannels one sample at a time.
            each sample is read as a 16bit word and then shifted to a 14bit value
        '''
        data = self.npdata[(channelNumber-1)::self.nChannels]
        if self._HiRes:
            temp = data * 0.25            # multiply by 0.25 for HiRes data
        else:
            temp = numpy.floor(data*0.25)              # bit shift by two for normal data

        temp2 = self.calScaling[channelNumber-1]*temp + self.calIntercept[channelNumber-1]

        return temp2

    def time(self):
        ''' return time '''
        
        return numpy.arange(0,int(self.nSample))*self.timeStep

    def unit(self, channelNumber):
        ''' return unit of requested channel '''
        unit = ''
        for b in self.engUnits[channelNumber-1]:
            unit += b.decode('utf-8')

        ''' Was getting \x00 in the unit string after decoding, lets remove that and whitespace '''
        unit.replace('\x00', '').strip()
        return unit

    def chAnnotation(self, channelNumber):
        ''' return user annotation of requested channel '''
        return self._annotations[channelNumber-1]
