# Project Structure

```text
├── c_extensions
│   └── src
│       ├── pc_utils
│       └── utils
├── docs
│   ├── internal
│   │   ├── api
│   │   ├── dev
│   │   └── _static
│   └── public
│       └── install
│           └── dependencies
├── examples
│   ├── aruko_markers
│   └── modules
│       ├── movement
│       └── video
├── go2
│   ├── communication
│   ├── core
│   ├── hardware
│   │   ├── native
│   │   └── virtual
│   ├── modules
│   │   ├── audio
│   │   ├── input
│   │   ├── lidar
│   │   │   └── utils
│   │   ├── movement
│   │   ├── ocr
│   │   └── video
│   │       ├── sources
│   │       └── streaming
│   └── states
├── requirements
├── ros2_ws
│   └── src
│       ├── bringup
│       │   ├── config
│       │   └── launch
│       ├── go2_interfaces
│       │   └── msg
│       └── lidar_processor
│           ├── lidar_processor
│           │   └── utils
│           ├── resource
│           └── test
│               ├── integration
│               ├── profiling
│               └── unit
└── scripts
    ├── unix
    └── windows
```

`c_extensions/` - Lidar filtering and decoding utilities written in C.

`docs/internal/` - Markdown files for building API documentation. \
`docs/internal/api/` - Markdown files for building student-facing API documentation. \
`docs/internal/dev/` - Markdown files for building developer-facing API documentation. 

`docs/public/` - Formatted documentation for students regarding the SDK itself.

`examples/` - SDK usage examples.

`go2/` - Python package containing all SDK functionalities. \
`go2/communication/` - CycloneDDS configurations. \
`go2/core/` - Core SDK functionalities. \
`go2/hardware/` - Wrappers for accessing the underlying robot (Native Hardware) or simulator (Virtual Hardware) communications. \
`go2/modules/` - Implementation code for each module (robot functionality) supported by the SDK. \
`go2/states/` - Code for supporting a state-machine architecture on the robot. 

`requirements/` - Python dependency information needed for package installation and documentation builds.

`ros2_ws/` - Workspace for communicating with the robot via ROS2. \
`ros2_ws/src/bringup/` - Files for creating ROS2 node launch scripts. \
`ros2_ws/src/go2_interfaces/` - ROS2 message created for inter-node communication. \
`ros2_ws/src/lidar_processor/` - ROS2 nodes for decoding and filtering Lidar information. 

`scripts/` - .sh or .bat files for miscellaneous use.
