#!/bin/sh
# install-tensorflow.sh
# Usage: sh install-tensorflow.sh
#
TF_TYPE="cpu" # Change to "gpu" for GPU support
OS="$(uname -s  | awk '{print tolower($0)}')"
TARGET_DIRECTORY="/usr/local"
URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.1.0.tar.gz"

# Download and expand tensorflow lib
case "$(uname -s)" in
   Darwin)
    sudo echo "Downloading the tensorflow library:"
    curl -L "$URL" | sudo tar -C $TARGET_DIRECTORY -xz
    ;;
   Linux)
    sudo echo "Downloading the tensorflow library:"
    wget -O - -o /dev/null "$URL" | sudo tar -C $TARGET_DIRECTORY -xz
    sudo ldconfig
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

echo "\nYou're ready to use tensorflow!"
echo "To compile your C programs, use the following command:"
echo "\n\tgcc myprogram.c -ltensorflow\n"
echo "Where myprogram.c is your program. Happy coding!"
