#!/bin/bash
# Build roboto-inference Debian package
# Requires ROS 2 (rclcpp, sensor_msgs, geometry_msgs, std_srvs)
# Requires roboto-motors and roboto-imu installed to /opt/roboparty
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${WORKSPACE}/build_deb/roboto-inference"
OUTPUT_DIR="${WORKSPACE}/deb_output"

PACKAGE="roboto-inference"
VERSION="1.0.0"
ARCH="$(dpkg --print-architecture)"
PREFIX="/opt/roboparty"

# Source ROS 2 (required)
ROS_SOURCED=false
for distro in jazzy iron humble rolling; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        echo ">>> Sourcing ROS 2 ${distro}"
        source "/opt/ros/${distro}/setup.bash"
        ROS_SOURCED=true
        break
    fi
done

if ! $ROS_SOURCED; then
    echo "ERROR: ROS 2 is required to build ${PACKAGE}."
    exit 1
fi

echo ">>> Building ${PACKAGE} ${VERSION}"
mkdir -p "${OUTPUT_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# cmake build + DESTDIR install
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}/cmake" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}/cmake" -j"$(nproc)"
DESTDIR="${BUILD_DIR}/destdir" cmake --install "${BUILD_DIR}/cmake"

# Stage deb
PKG_STAGE="${BUILD_DIR}/${PACKAGE}_${VERSION}_${ARCH}"
mkdir -p "${PKG_STAGE}/DEBIAN"

if [ -d "${BUILD_DIR}/destdir${PREFIX}" ]; then
    mkdir -p "${PKG_STAGE}${PREFIX}"
    cp -a "${BUILD_DIR}/destdir${PREFIX}/." "${PKG_STAGE}${PREFIX}/"
fi

# Install models and motions (not handled by cmake install)
if [ -d "${SCRIPT_DIR}/models" ]; then
    mkdir -p "${PKG_STAGE}${PREFIX}/share/inference/models"
    cp "${SCRIPT_DIR}/models/"*.onnx "${PKG_STAGE}${PREFIX}/share/inference/models/" 2>/dev/null || true
fi

if [ -d "${SCRIPT_DIR}/motions" ]; then
    mkdir -p "${PKG_STAGE}${PREFIX}/share/inference/motions"
    cp "${SCRIPT_DIR}/motions/"*.npz "${PKG_STAGE}${PREFIX}/share/inference/motions/" 2>/dev/null || true
fi

cp "${SCRIPT_DIR}/debian/control"  "${PKG_STAGE}/DEBIAN/"
cp "${SCRIPT_DIR}/debian/postinst" "${PKG_STAGE}/DEBIAN/"
cp "${SCRIPT_DIR}/debian/postrm"   "${PKG_STAGE}/DEBIAN/"
[ -f "${SCRIPT_DIR}/debian/conffiles" ] && cp "${SCRIPT_DIR}/debian/conffiles" "${PKG_STAGE}/DEBIAN/"
sed -i "s/ARCH_PLACEHOLDER/${ARCH}/" "${PKG_STAGE}/DEBIAN/control"
chmod 755 "${PKG_STAGE}/DEBIAN/postinst" "${PKG_STAGE}/DEBIAN/postrm"

dpkg-deb --root-owner-group --build "${PKG_STAGE}" "${OUTPUT_DIR}/"
echo ">>> Done: ${OUTPUT_DIR}/${PACKAGE}_${VERSION}_${ARCH}.deb"
