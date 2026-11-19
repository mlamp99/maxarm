#!/usr/bin/env bash
# Build the CH340/CH341 USB serial driver (ch341.ko) on an FRDM-IMX95 board, for the kernel it runs.
# NXP's images leave it out, so the arm's ESP32 shows up in lsusb but gets no /dev/ttyUSB0.
#
# Only the module is built, not the kernel: NXP's source at the commit in uname -r, the running kernel's
# /proc/config.gz, and the image's own gcc. The kernel checks a version code (CRC) for every symbol a module uses
# and the image has no Module.symvers, so the directories exporting what ch341 uses are compiled to get theirs.
# Everything is in ~/kbuild; nothing else is touched. The result is ~/kbuild/ch341/ch341.ko, to be copied to
# files/ch341/<uname -r>/ for scripts/install-ch341.sh.
set -e

# The kernel's build scripts need bison and flex, which the image does not have. They go in ~/kbuild/tools.
build_tool() {
  local tarball=$(basename $1)
  curl -fsSLo ${HOME}/kbuild/dl/$tarball $1
  tar xf ${HOME}/kbuild/dl/$tarball
  cd ${tarball%.tar.*}
  ./configure -q --prefix=${HOME}/kbuild/tools
  make -s -j6
  make -s install
  cd ..
}

# A release like 6.18.37-2.1.0-g6e694eda40aa: the kernel's version, NXP's suffix, and the commit it was built from.
RELEASE=$(uname -r)

mkdir -p ${HOME}/kbuild/dl
cd ${HOME}/kbuild
export PATH=${HOME}/kbuild/tools/bin:$PATH
build_tool https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.xz > bison.log 2>&1
build_tool https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz > flex.log 2>&1

curl -fsSLo dl/linux-imx.tar.gz https://github.com/nxp-imx/linux-imx/archive/${RELEASE##*-g}.tar.gz
rm -rf linux
mkdir linux
tar xzf dl/linux-imx.tar.gz -C linux --strip-components=1
cd linux
zcat /proc/config.gz > .config
# The module's vermagic must read exactly as uname -r; Yocto and git made the part after the version.
echo "-${RELEASE#*-}" > localversion
make olddefconfig
make -j6 modules_prepare

# Each object records the CRCs of what it exports as "#SYMVER <symbol> <crc>" lines in its .cmd file.
make -j6 kernel/ mm/ drivers/base/ drivers/tty/ drivers/usb/core/ drivers/usb/serial/
mkdir -p ${HOME}/kbuild/ch341
find . -name "*.o.cmd" | xargs grep -h "^#SYMVER" \
  | awk '{ printf "%s\t%s\tvmlinux\tEXPORT_SYMBOL\t\n", $3, $2 }' > ${HOME}/kbuild/ch341/vmlinux.symvers

cd ${HOME}/kbuild/ch341
cp ../linux/drivers/usb/serial/ch341.c .
echo "obj-m := ch341.o" > Makefile
make -C ../linux M=$PWD KBUILD_EXTRA_SYMBOLS=$PWD/vmlinux.symvers modules

modinfo -F vermagic ch341.ko
echo "Built ${HOME}/kbuild/ch341/ch341.ko. Try it with: insmod ${HOME}/kbuild/ch341/ch341.ko"
