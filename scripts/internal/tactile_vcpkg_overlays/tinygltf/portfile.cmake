# Fetch the same source by immutable Git object instead of a regenerated tag archive.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/syoyo/tinygltf.git
    REF 26422192e2908a562b641175dde18489824e609e
)
vcpkg_replace_string(
    "${SOURCE_PATH}/tiny_gltf.h"
    "#include \"json.hpp\""
    "#include <nlohmann/json.hpp>"
)
file(INSTALL "${SOURCE_PATH}/tiny_gltf.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
