#!/bin/bash
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
g++ -std=c++11 -stdlib=libc++ -Ofast third_party/glad/glad.cpp kohonen_rgb_random.c -o kohonen_rgb_random -isysroot ${SDKROOT} -Wl,-search_paths_first -Wl,-headerpad_max_install_names -framework OpenGL -framework Cocoa -Wno-deprecated -I/usr/include -I/usr/local/Cellar/glfw/3.3/include/ -I. -Isrc/ -Ithird_party -L/usr/local/Cellar/glfw/3.3/lib/ -lGLFW

