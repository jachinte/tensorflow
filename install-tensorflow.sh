#!/bin/sh
# install-tensorflow.sh
# Usage: sh install-tensorflow.sh
#
TF_TYPE="cpu" # Change to "gpu" for GPU support
OS="$(uname -s  | awk '{print tolower($0)}')"
TARGET_DIRECTORY="/usr/local"

# Download and expand tensorflow lib
sudo echo "Downloading the tensorflow library:"
curl -L \
  "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.1.0.tar.gz" | \
  sudo tar -C $TARGET_DIRECTORY -xz

case "$(uname -s)" in
   Darwin)
    echo "\nYou're ready to use tensorflow!"
    echo "To compile your C programs, use the following command:"
    echo "\n\tgcc myprogram.c -ltensorflow\n"
    echo "Where myprogram.c is your program. Happy coding!"
    ;;
   Linux)
    sudo ldconfig
    echo "\nYou're ready to use tensorflow!"
    ;;
   CYGWIN*|MINGW32*|MSYS*)
     echo "I'm sorry mate, Windows is not supported"
     exit 1
     ;;
   *)
     echo "I'm sorry mate, this operating system is not supported"
     exit 1
     ;;
esac
