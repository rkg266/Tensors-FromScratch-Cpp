#ifndef GLOBALS_HPP
#define GLOBALS_HPP

enum class Device {
    cpu,
    gpu
};

extern Device UseDevice;
using dataType = float;

#endif // GLOBALS_HPP