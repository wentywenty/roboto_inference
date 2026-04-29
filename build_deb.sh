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
DEB_DIR="${PACKAGE}_${VERSION}_${ARCH}"

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
fi

# Install models and motions (not handled by cmake install)
if [ -d "models" ]; then
    mkdir -p "${DEB_DIR}${PREFIX}/share/inference/models"
    cp models/*.onnx "${DEB_DIR}${PREFIX}/share/inference/models/" 2>/dev/null || true
fi

if [ -d "motions" ]; then
    mkdir -p "${DEB_DIR}${PREFIX}/share/inference/motions"
    cp motions/*.npz "${DEB_DIR}${PREFIX}/share/inference/motions/" 2>/dev/null || true
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
