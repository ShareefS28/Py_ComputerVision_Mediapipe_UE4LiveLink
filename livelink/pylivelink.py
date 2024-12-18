from __future__ import annotations
from collections import deque
from statistics import mean
from enum import Enum
import struct
from typing import Tuple
import datetime
import uuid
from timecode import Timecode

class BlendShape(Enum):
    EyeBlinkLeft = 0        
    EyeLookDownLeft = 1
    ShoulderLeftAngle_z = 2             # EyeLookInLeft = 2         
    ShoulderLeftAngle_x = 3             # EyeLookOutLeft = 3
    EyeLookUpLeft = 4
    EyeSquintLeft = 5
    EyeWideLeft = 6
    EyeBlinkRight = 7
    EyeLookDownRight = 8
    EyeLookInRight = 9
    EyeLookOutRight = 10
    ElbowLeftAngle_z = 11               # EyeLookUpRight = 11
    ElbowLeftAngle_x = 12               # EyeSquintRight = 12
    EyeWideRight = 13
    ShoulderRightAngle_z = 14           # JawForward = 14
    ShoulderRightAngle_x = 15           # JawLeft = 15
    JawRight = 16
    JawOpen = 17
    MouthClose = 18
    MouthFunnel = 19
    MouthPucker = 20
    ElbowRightAngle_z = 21              # MouthLeft = 21
    ElbowRightAngle_x = 22              # MouthRight = 22
    MouthSmileLeft = 23
    ThighLeftAngle_z = 24               # MouthSmileRight = 24
    ThighLeftAngle_x = 25               # MouthFrownLeft = 25
    ThighRightAngle_z = 26              # MouthFrownRight = 26
    ThighRightAngle_x = 27              # MouthDimpleLeft = 27
    ShinLeftAngle_z = 28                # MouthDimpleRight = 28
    ShinLeftAngle_x = 29                # MouthStretchLeft = 29
    ShinRightAngle_z= 30                # MouthStretchRight = 30
    ShinRightAngle_x = 31               # MouthRollLower = 31
    FootLeftAngle_x = 32                # MouthRollUpper = 32
    FootLeftAngle_y = 33                # MouthShrugLower = 33
    FootRightAngle_x = 34               # MouthShrugUpper = 34
    FootRightAngle_y = 35               # MouthPressLeft = 35
    BodyShoulderAngle_y = 36            # MouthPressRight = 36
    BodyShoulderAngle_x = 37            # MouthLowerDownLeft = 37 
    BodyAnglePitch_z = 38               # MouthLowerDownRight = 38
    BodyHipAngle_y = 39                 # MouthUpperUpLeft = 39
    BodyHipAngle_x = 40                 # MouthUpperUpRight = 40
    BrowDownLeft = 41             
    BrowDownRight = 42
    BrowInnerUp = 43
    Walk_x = 44                         # BrowOuterUpLeft = 44
    Walk_y = 45                         # BrowOuterUpRight = 45
    Walk_z = 46                         # CheekPuff = 46
    CheekSquintLeft = 47
    CheekSquintRight = 48
    NoseSneerLeft = 49
    NoseSneerRight = 50
    TongueOut = 51
    HeadYaw = 52
    HeadPitch = 53
    HeadRoll = 54
    # LeftEyeYaw = 55
    # LeftEyePitch = 56
    # LeftEyeRoll = 57
    # RightEyeYaw = 58
    # RightEyePitch = 59
    # RightEyeRoll = 60

    # Lef Hand
    WristLeftThumb_x = 55
    WristLeftThumb_y = 56
    ThumbCMCLeftAngle_x = 57
    ThumbCMCLeftAngle_y = 58
    ThumbMCPLeftAngle_x = 59
    ThumbMCPLeftAngle_y = 60
    ThumbIPLeftAngle_x = 61
    ThumbIPLeftAngle_y = 62
    WristLeftIndex_x = 63
    WristLeftIndex_y = 64
    IndexFingerMCPLeftAngle_x = 65
    IndexFingerMCPLeftAngle_y = 66
    IndexFingerPIPLeftAngle_x = 67
    IndexFingerPIPLeftAngle_y = 68
    IndexFingerDIPLeftAngle_x = 69
    IndexFingerDIPLeftAngle_y = 70
    WristLeftMiddle_x = 71
    WristLeftMiddle_y = 72
    MiddleFingerMCPLeftAngle_x = 73
    MiddleFingerMCPLeftAngle_y = 74
    MiddleFingerPIPLeftAngle_x = 75
    MiddleFingerPIPLeftAngle_y = 76
    MiddleFingerDIPLeftAngle_x = 77
    MiddleFingerDIPLeftAngle_y = 78
    WristLeftRing_x = 78
    WristLeftRing_y = 79
    RingFingerMCPLeftAngle_x = 81
    RingFingerMCPLeftAngle_y = 82
    RingFingerPIPLeftAngle_x = 83
    RingFingerPIPLeftAngle_y = 84
    RingFingerDIPLeftAngle_x = 85
    RingFingerDIPLeftAngle_y = 86
    WristLeftPinky_x = 87
    WristLeftPinky_y = 88
    PinkyMCPLeftAngle_x = 89
    PinkyMCPLeftAngle_y = 90
    PinkyPIPLeftAngle_x = 91
    PinkyPIPLeftAngle_y = 92
    PinkyDIPLeftAngle_x = 93
    PinkyDIPLeftAngle_y = 94
    # Right Hand
    WristRightThumb_x = 95
    WristRightThumb_y = 96
    ThumbCMCRightAngle_x = 97
    ThumbCMCRightAngle_y = 98
    ThumbMCPRightAngle_x = 99
    ThumbMCPRightAngle_y = 100
    ThumbIPRightAngle_x = 101
    ThumbIPRightAngle_y = 102
    WristRightIndex_x = 103
    WristRightIndex_y = 104
    IndexFingerMCPRightAngle_x = 105
    IndexFingerMCPRightAngle_y = 106
    IndexFingerPIPRightAngle_x = 107
    IndexFingerPIPRightAngle_y = 108
    IndexFingerDIPRightAngle_x = 109
    IndexFingerDIPRightAngle_y = 110
    WristRightMiddle_x = 111
    WristRightMiddle_y = 112
    MiddleFingerMCPRightAngle_x = 113
    MiddleFingerMCPRightAngle_y = 114
    MiddleFingerPIPRightAngle_x = 115
    MiddleFingerPIPRightAngle_y = 116
    MiddleFingerDIPRightAngle_x = 117
    MiddleFingerDIPRightAngle_y = 118
    WristRightRing_x = 119
    WristRightRing_y = 120
    RingFingerMCPRightAngle_x = 121
    RingFingerMCPRightAngle_y = 122
    RingFingerPIPRightAngle_x = 123
    RingFingerPIPRightAngle_y = 124
    RingFingerDIPRightAngle_x = 125
    RingFingerDIPRightAngle_y = 126
    WristRightPinky_x = 127
    WristRightPinky_y = 128
    PinkyMCPRightAngle_x = 129
    PinkyMCPRightAngle_y = 130
    PinkyPIPRightAngle_x = 131
    PinkyPIPRightAngle_y = 132
    PinkyDIPRightAngle_x = 133
    PinkyDIPRightAngle_y = 134


class PyLiveLink:
    """PyLiveLinkFace class

    Can be used to receive PyLiveLinkFace from the PyLiveLinkFace IPhone app or
    other PyLiveLinkFace compatible programs like this library.
    """

    def __init__(self, name: str = "Python_LiveLinkPose", 
                        uuid: str = str(uuid.uuid1()), fps=60, 
                        filter_size: int = 5) -> None:

        # properties
        self.uuid = uuid
        self.name = name
        self.fps = fps
        self._filter_size = filter_size

        self._version = 6
        now = datetime.datetime.now()
        timcode = Timecode(
            self._fps, f'{now.hour}:{now.minute}:{now.second}:{now.microsecond * 0.001}')
        self._frames = timcode.frames
        self._sub_frame = 1056060032                # I don't know how to calculate this [default(1056060032)]
        self._denominator = int(self._fps / 60)     # 1 most of the time
        self._blend_shapes = [0.000] * len(BlendShape.__members__)          # BlendShape size of value in enum+1
        self._old_blend_shapes = []                 # used for filtering
        for i in range(len(BlendShape.__members__)):  # BlendShape size of value in enum+1
            self._old_blend_shapes.append(deque([0.0], maxlen = self._filter_size))

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, value: str) -> None:
        # uuid needs to start with a $, if it doesn't add it
        if not value.startswith("$"):
            self._uuid = '$' + value
        else:
            self._uuid = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        if value < 1:
            raise ValueError("Only fps values greater than 1 are allowed.")
        self._fps = value

    def encode(self) -> bytes:
        """ Encodes the PyLiveLinkFace object into a bytes object so it can be 
        send over a network. """              
        
        version_packed = struct.pack('<I', self._version)
        uuiid_packed = bytes(self._uuid, 'utf-8')
        name_lenght_packed = struct.pack('!i', len(self._name))
        name_packed = bytes(self._name, 'utf-8')

        now = datetime.datetime.now()
        timcode = Timecode(
            self._fps, f'{now.hour}:{now.minute}:{now.second}:{now.microsecond * 0.001}')
        frames_packed = struct.pack("!II", timcode.frames, self._sub_frame)  
        frame_rate_packed = struct.pack("!II", self._fps, self._denominator)
        data_packed = struct.pack(f'!B{len(BlendShape.__members__)}f', len(BlendShape.__members__), *self._blend_shapes)  # 61 is size of value in enum+1 !B61f too 
        
        return version_packed + uuiid_packed + name_lenght_packed + name_packed + \
            frames_packed + frame_rate_packed + data_packed

    def get_blendshape(self, index: BlendShape) -> float:
        """ Get the current value of the blend shape. 

        Parameters
        ----------
        index : FaceBlendShape
            Index of the BlendShape to get the value from.

        Returns
        -------
        float
            The value of the BlendShape.
        """        
        return self._blend_shapes[index.value]

    def set_blendshape(self, index: BlendShape, value: float, 
                        no_filter: bool = False) -> None:
        """ Sets the value of the blendshape. 
        
        The function will use mean to filter between the old and the new 
        values, unless `no_filter` is set to True.

        Parameters
        ----------
        index : FaceBlendShape
            Index of the BlendShape to get the value from.
        value: float
            Value to set the BlendShape to, should be in the range of 0 - 1 for 
            the blendshapes and between -1 and 1 for the head rotation 
            (yaw, pitch, roll).
        no_filter: bool
            If set to True, the blendshape will be set to the value without 
            filtering.
        
        Returns
        ----------
        None
        """

        if no_filter:
            self._blend_shapes[index.value] = value
        else:
            self._old_blend_shapes[index.value].append(value)
            filterd_value = mean(self._old_blend_shapes[index.value])
            self._blend_shapes[index.value] = filterd_value

    @staticmethod
    def decode(bytes_data: bytes) -> Tuple[bool, PyLiveLink]:
        """ Decodes the given bytes (send from an PyLiveLinkFace App or from 
        this library) and creates a new PyLiveLinkFace object.
        Returns True and the generated object if a face was found in the data, 
        False an a new empty PyLiveLinkFace otherwise. 

        Parameters
        ----------
        bytes_data : bytes
            Bytes input to create the PyLiveLinkFace object from.

        Returns
        -------
        bool
            True if the bytes data contained a face, False if not.        
        PyLiveLinkFace
            The PyLiveLinkFace object.

        """
        version = struct.unpack('<i', bytes_data[0:4])[0]
        uuid = bytes_data[4:41].decode("utf-8")
        name_length = struct.unpack('!i', bytes_data[41:45])[0]
        name_end_pos = 45 + name_length
        name = bytes_data[45:name_end_pos].decode("utf-8")
        if len(bytes_data) > name_end_pos + 16:

            #FFrameTime, FFrameRate and data length
            frame_number, sub_frame, fps, denominator, data_length = struct.unpack(
                "!if2ib", bytes_data[name_end_pos:name_end_pos + 17])

            if data_length != 61:
                raise ValueError(
                    f'Blend shape length is {data_length} but should be 61, something is wrong with the data.')

            data = struct.unpack(
                "!61f", bytes_data[name_end_pos + 17:])

            live_link_face = PyLiveLink(name, uuid, fps)
            live_link_face._version = version
            live_link_face._frames = frame_number
            live_link_face._sub_frame = sub_frame
            live_link_face._denominator = denominator
            live_link_face._blend_shapes = data

            return True, live_link_face
        else:
            #print("Data does not contain a face, returning default empty face.")
            return False, PyLiveLink()
