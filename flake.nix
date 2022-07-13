{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            # CMake et al
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config

            # FFmpeg
            pkgs.ffmpeg
            pkgs.nv-codec-headers-11

            # OpenCL
            pkgs.ocl-icd
            pkgs.opencl-clhpp
            pkgs.opencl-headers

            # Vulkan
            pkgs.vulkan-headers
            pkgs.vulkan-loader

            # Getting a window from X
            pkgs.xorg.libXau
            pkgs.xorg.libXdmcp
            pkgs.xorg.libxcb
          ];
        };
      });
}
