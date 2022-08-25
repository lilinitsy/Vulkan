{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg:
            builtins.elem (nixpkgs.lib.getName pkg) [
              "tools" # Android SDK tools
            ];
        };
        androidComposition = (pkgs.androidenv.override {
          licenseAccepted = true;
        }).composeAndroidPackages {
          abiVersions = [ "arm64-v8a" ];
          buildToolsVersions = [ "32.0.0" ];
          cmakeVersions = [ "3.22.1" ];
          includeNDK = true;
          ndkVersion = "23.0.7344513-rc4";
          platformVersions = [ "26" ];
          platformToolsVersion = "33.0.1";
          toolsVersion = "26.1.1";
        };
      in rec {
        packages = { inherit (androidComposition) androidsdk; };
        devShells.default = pkgs.mkShell {
          ANDROID_SDK_ROOT =
            "${androidComposition.androidsdk}/libexec/android-sdk";
          # ANDROID_NDK_ROOT = "${androidComposition.androidsdk}/libexec/android-sdk/ndk-bundle";
          GRADLE_OPTS =
            "-Dorg.gradle.project.android.aapt2FromMavenOverride=${androidComposition.androidsdk}/libexec/android-sdk/build-tools/32.0.0/aapt2";

          shellHook = ''
            export PATH=$ANDROID_SDK_ROOT/cmake/3.22.1/bin:$PATH
          '';

          nativeBuildInputs = [
            # android setup stuff
            pkgs.jdk11
            pkgs.android-tools
            packages.androidsdk

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
