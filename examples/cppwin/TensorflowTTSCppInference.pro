TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt
TARGET = TFTTSCppInfer

HEADERS += \
    TensorflowTTSCppInference/EnglishPhoneticProcessor.h \
    TensorflowTTSCppInference/FastSpeech2.h \
    TensorflowTTSCppInference/MultiBandMelGAN.h \
    TensorflowTTSCppInference/TextTokenizer.h \
    TensorflowTTSCppInference/Voice.h \
    TensorflowTTSCppInference/VoxCommon.hpp \
    TensorflowTTSCppInference/ext/AudioFile.hpp \
    TensorflowTTSCppInference/ext/CppFlow/include/Model.h \
    TensorflowTTSCppInference/ext/CppFlow/include/Tensor.h \
    TensorflowTTSCppInference/ext/ZCharScanner.h

SOURCES += \
    TensorflowTTSCppInference/EnglishPhoneticProcessor.cpp \
    TensorflowTTSCppInference/FastSpeech2.cpp \
    TensorflowTTSCppInference/MultiBandMelGAN.cpp \
    TensorflowTTSCppInference/TensorflowTTSCppInference.cpp \
    TensorflowTTSCppInference/TextTokenizer.cpp \
    TensorflowTTSCppInference/Voice.cpp \
    TensorflowTTSCppInference/VoxCommon.cpp \
    TensorflowTTSCppInference/ext/CppFlow/src/Model.cpp \
    TensorflowTTSCppInference/ext/CppFlow/src/Tensor.cpp \
    TensorflowTTSCppInference/ext/ZCharScanner.cpp

INCLUDEPATH += $$PWD/deps/include
LIBS += -L$$PWD/deps/lib -lfst -lfstfar -lfstngram -ltensorflow -lphonetisaurus

# GCC shits itself on memcp in AudioFile.hpp (l-1186) unless we add this
QMAKE_CXXFLAGS += -fpermissive

# Stop ld from whining about LoadClusters
QMAKE_CXXFLAGS += -Wl,-b,svr4,-z,multidefs
