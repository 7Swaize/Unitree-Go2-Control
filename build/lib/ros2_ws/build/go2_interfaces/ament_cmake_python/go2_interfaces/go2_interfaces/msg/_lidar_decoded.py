# generated from rosidl_generator_py/resource/_idl.py.em
# with input from go2_interfaces:msg/LidarDecoded.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'xyz_shape'
# Member 'xyz_data'
# Member 'intensity_shape'
# Member 'intensity_data'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LidarDecoded(type):
    """Metaclass of message 'LidarDecoded'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('go2_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'go2_interfaces.msg.LidarDecoded')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__lidar_decoded
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__lidar_decoded
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__lidar_decoded
            cls._TYPE_SUPPORT = module.type_support_msg__msg__lidar_decoded
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__lidar_decoded

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LidarDecoded(metaclass=Metaclass_LidarDecoded):
    """Message class 'LidarDecoded'."""

    __slots__ = [
        '_header',
        '_xyz_shape',
        '_xyz_dtype',
        '_xyz_data',
        '_has_intensity',
        '_intensity_shape',
        '_intensity_dtype',
        '_intensity_data',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'xyz_shape': 'sequence<uint32>',
        'xyz_dtype': 'uint8',
        'xyz_data': 'sequence<uint8>',
        'has_intensity': 'boolean',
        'intensity_shape': 'sequence<uint32>',
        'intensity_dtype': 'uint8',
        'intensity_data': 'sequence<uint8>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint32')),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint32')),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.xyz_shape = array.array('I', kwargs.get('xyz_shape', []))
        self.xyz_dtype = kwargs.get('xyz_dtype', int())
        self.xyz_data = array.array('B', kwargs.get('xyz_data', []))
        self.has_intensity = kwargs.get('has_intensity', bool())
        self.intensity_shape = array.array('I', kwargs.get('intensity_shape', []))
        self.intensity_dtype = kwargs.get('intensity_dtype', int())
        self.intensity_data = array.array('B', kwargs.get('intensity_data', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.xyz_shape != other.xyz_shape:
            return False
        if self.xyz_dtype != other.xyz_dtype:
            return False
        if self.xyz_data != other.xyz_data:
            return False
        if self.has_intensity != other.has_intensity:
            return False
        if self.intensity_shape != other.intensity_shape:
            return False
        if self.intensity_dtype != other.intensity_dtype:
            return False
        if self.intensity_data != other.intensity_data:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def xyz_shape(self):
        """Message field 'xyz_shape'."""
        return self._xyz_shape

    @xyz_shape.setter
    def xyz_shape(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'I', \
                "The 'xyz_shape' array.array() must have the type code of 'I'"
            self._xyz_shape = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 4294967296 for val in value)), \
                "The 'xyz_shape' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 4294967295]"
        self._xyz_shape = array.array('I', value)

    @builtins.property
    def xyz_dtype(self):
        """Message field 'xyz_dtype'."""
        return self._xyz_dtype

    @xyz_dtype.setter
    def xyz_dtype(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'xyz_dtype' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'xyz_dtype' field must be an unsigned integer in [0, 255]"
        self._xyz_dtype = value

    @builtins.property
    def xyz_data(self):
        """Message field 'xyz_data'."""
        return self._xyz_data

    @xyz_data.setter
    def xyz_data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'xyz_data' array.array() must have the type code of 'B'"
            self._xyz_data = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'xyz_data' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._xyz_data = array.array('B', value)

    @builtins.property
    def has_intensity(self):
        """Message field 'has_intensity'."""
        return self._has_intensity

    @has_intensity.setter
    def has_intensity(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'has_intensity' field must be of type 'bool'"
        self._has_intensity = value

    @builtins.property
    def intensity_shape(self):
        """Message field 'intensity_shape'."""
        return self._intensity_shape

    @intensity_shape.setter
    def intensity_shape(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'I', \
                "The 'intensity_shape' array.array() must have the type code of 'I'"
            self._intensity_shape = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 4294967296 for val in value)), \
                "The 'intensity_shape' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 4294967295]"
        self._intensity_shape = array.array('I', value)

    @builtins.property
    def intensity_dtype(self):
        """Message field 'intensity_dtype'."""
        return self._intensity_dtype

    @intensity_dtype.setter
    def intensity_dtype(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'intensity_dtype' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'intensity_dtype' field must be an unsigned integer in [0, 255]"
        self._intensity_dtype = value

    @builtins.property
    def intensity_data(self):
        """Message field 'intensity_data'."""
        return self._intensity_data

    @intensity_data.setter
    def intensity_data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'intensity_data' array.array() must have the type code of 'B'"
            self._intensity_data = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'intensity_data' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._intensity_data = array.array('B', value)
