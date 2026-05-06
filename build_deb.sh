#!/bin/bash
# Build roboto-inference Debian package
# Requires ROS 2 (rclcpp, sensor_msgs, geometry_msgs, std_srvs)
# Requires roboto-motors/imu/bms installed to /opt/roboparty
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PACKAGE="roboto-inference"
VERSION="1.1.1"
ARCH="$(dpkg --print-architecture)"
PREFIX="/opt/roboparty"
PKG_NAME="roboto-inference"
DEB_DIR="${PACKAGE}_${VERSION}_${ARCH}"

# Thirdparty dependencies (auto-extract if archives exist)
# ONNX Runtime
ONNXRT_X64="thirdparty/onnxruntime-linux-x64-1.21.0"
ONNXRT_ARM64="thirdparty/onnxruntime-linux-aarch64-1.21.0"
# wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz -O thirdparty/onnxruntime-linux-x64-1.21.0.tgz
# wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-aarch64-1.21.0.tgz -O thirdparty/onnxruntime-linux-aarch64-1.21.0.tgz
if [ ! -d "$ONNXRT_X64" ] && [ -f "thirdparty/onnxruntime-linux-x64-1.21.0.tgz" ]; then
    echo ">>> Extracting onnxruntime x64..."
    mkdir -p "$ONNXRT_X64" && tar xzf thirdparty/onnxruntime-linux-x64-1.21.0.tgz -C "$ONNXRT_X64" --strip-components=1
fi
if [ ! -d "$ONNXRT_ARM64" ] && [ -f "thirdparty/onnxruntime-linux-aarch64-1.21.0.tgz" ]; then
    echo ">>> Extracting onnxruntime aarch64..."
    mkdir -p "$ONNXRT_ARM64" && tar xzf thirdparty/onnxruntime-linux-aarch64-1.21.0.tgz -C "$ONNXRT_ARM64" --strip-components=1
fi

# yaml-cpp
# wget -q https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz -O thirdparty/yaml-cpp-0.9.0.tar.gz
if [ ! -d "thirdparty/yaml-cpp-0.9.0" ] && [ -f "thirdparty/yaml-cpp-0.9.0.tar.gz" ]; then
    echo ">>> Extracting yaml-cpp..."
    mkdir -p thirdparty/yaml-cpp-0.9.0 && tar xzf thirdparty/yaml-cpp-0.9.0.tar.gz -C thirdparty/yaml-cpp-0.9.0 --strip-components=1
fi

# Source ROS 2 (Always required for inference)
ROS_SOURCED=0
for distro in jazzy iron humble rolling; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        echo ">>> Sourcing ROS 2 ${distro}"
        source "/opt/ros/${distro}/setup.bash"
        ROS_SOURCED=1
        break
    fi
done

if [ "$ROS_SOURCED" = "0" ]; then
    echo "ERROR: ROS 2 is required to build ${PACKAGE}."
    exit 1
fi

echo ">>> Starting compilation..."
rm -rf build && mkdir -p build
pushd build > /dev/null
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
DESTDIR="${SCRIPT_DIR}/build/destdir" cmake --install .
popd > /dev/null

echo ">>> Preparing Debian package structure..."
rm -rf "${DEB_DIR}" "${DEB_DIR}.deb"
mkdir -p "${DEB_DIR}/DEBIAN"

# Copy cmake installed files
if [ -d "build/destdir${PREFIX}" ]; then
    mkdir -p "${DEB_DIR}${PREFIX}"
    cp -a "build/destdir${PREFIX}/." "${DEB_DIR}${PREFIX}/"
    # Remove cnpy helper scripts
    rm -f "${DEB_DIR}${PREFIX}/bin/mat2npz" "${DEB_DIR}${PREFIX}/bin/npy2mat" "${DEB_DIR}${PREFIX}/bin/npz2mat"
fi

# Fix .so permissions and symlinks
LIB_DIR="${DEB_DIR}${PREFIX}/lib"
if [ -d "$LIB_DIR" ]; then
    chmod +x "$LIB_DIR"/*.so* 2>/dev/null || true
    # Fix onnxruntime symlinks
    if [ -f "$LIB_DIR/libonnxruntime.so.1.21.0" ]; then
        ln -sf libonnxruntime.so.1.21.0 "$LIB_DIR/libonnxruntime.so.1"
        ln -sf libonnxruntime.so.1.21.0 "$LIB_DIR/libonnxruntime.so"
    fi
fi

# Reorganize config: inference*.yaml -> config/inference/, robot*.yaml -> config/robot/
CONFIG_DIR="${DEB_DIR}${PREFIX}/share/${PKG_NAME}/config"
if [ -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR/inference" "$CONFIG_DIR/robot"
    mv "$CONFIG_DIR"/inference*.yaml "$CONFIG_DIR/inference/" 2>/dev/null || true
    mv "$CONFIG_DIR"/robot*.yaml "$CONFIG_DIR/robot/" 2>/dev/null || true
fi

# Install models and motions (not handled by cmake install)
if [ -d "models" ]; then
    mkdir -p "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/models"
    cp models/*.onnx "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/models/" 2>/dev/null || true
    chmod 644 "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/models/"*.onnx 2>/dev/null || true
fi

if [ -d "motions" ]; then
    mkdir -p "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/motions"
    cp motions/*.npz "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/motions/" 2>/dev/null || true
    chmod 644 "${DEB_DIR}${PREFIX}/share/${PKG_NAME}/motions/"*.npz 2>/dev/null || true
fi

# Copy DEBIAN maintainer scripts
cp debian/control  "${DEB_DIR}/DEBIAN/"
cp debian/postinst "${DEB_DIR}/DEBIAN/"
cp debian/postrm   "${DEB_DIR}/DEBIAN/"
[ -f debian/conffiles ] && cp debian/conffiles "${DEB_DIR}/DEBIAN/"
chmod 755 "${DEB_DIR}/DEBIAN/postinst" "${DEB_DIR}/DEBIAN/postrm"

# Generate Control file (Replace placeholders)
sed -i "s/ARCH_PLACEHOLDER/${ARCH}/" "${DEB_DIR}/DEBIAN/control"
sed -i "s/VERSION_PLACEHOLDER/${VERSION}/" "${DEB_DIR}/DEBIAN/control"

echo ">>> Executing dpkg-deb build..."
dpkg-deb --root-owner-group --build "${DEB_DIR}"

echo ">>> Success! Generated ${DEB_DIR}.deb"
