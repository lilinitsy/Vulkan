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

        ndk = "${androidComposition.androidsdk}/libexec/android-sdk/ndk-bundle";
        sysroot = "${ndk}/toolchains/llvm/prebuilt/linux-x86_64/sysroot";

        android-ffmpeg = pkgs.stdenvNoCC.mkDerivation {
          pname = "android-ffmpeg";
          version = "5.1.1";
          src = pkgs.fetchzip {
            url = "https://ffmpeg.org/releases/ffmpeg-5.1.1.tar.xz";
            hash = "sha256-IQelw+Bv8Dy6oTdhByveaij0CRaO5CKVON4RmaAx9iY=";
          };
          configureFlags = [
            "--ar=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
            "--cc=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang"
            "--cxx=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++"
            "--ld=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang"
            "--nm=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm"
            "--ranlib=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ranlib"
            "--strip=${ndk}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
            "--enable-cross-compile"
            "--enable-static"
            "--sysroot=${sysroot}"
            "--arch=aarch64"
            "--target-os=android"
          ];
        };
      in rec {
        packages = {
          inherit android-ffmpeg;
          inherit (androidComposition) androidsdk;
        };
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
