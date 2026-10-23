{
  description = "Gufo Toolbx: one pinned ROCm stack with the matching tuning tools";

  inputs = {
    gufo.url = "github:gufo-org/gufo/7c8196527ad02ac4c585a32b1131fab7e70509d0";
    nixpkgs.follows = "gufo/nixpkgs";
  };

  outputs = { self, gufo, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      revision = builtins.replaceStrings [ "\n" "\r" ] [ "" "" ]
        (builtins.readFile ./gufo-revision);
      base = gufo.packages.${system}.default.override { inherit revision; };
      runtime = base.rocmRuntime;
      withTools = base.overrideAttrs (old: {
        pname = "gufo-toolbox";
        # The upstream production source filter intentionally excludes tools/.
        src = gufo.outPath;
        buildInputs = (old.buildInputs or [ ]) ++ [
          base.rocmPackages.rocprofiler-sdk
        ];
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [
          "-DGUFO_BUILD_TOOLS=ON"
          "-DCMAKE_IGNORE_PREFIX_PATH=/opt/rocm"
          # ROCm's unwrapped Clang contains its GNU sysroot but does not select
          # it automatically on an FHS host/container. Keep HIP's CRT objects
          # and headers in the same Nix toolchain during CMake's probe and build.
          "-DCMAKE_HIP_FLAGS=--sysroot=${base.rocmPackages.llvm.clang}"
        ];
        # Runtime selection is process-local. Inherited host/container ROCm
        # paths must not override the pinned HIP/BLAS dependency closure.
        postFixup = ''
          for program in gufo gufo-kernel-bench tune_hipblaslt; do
            wrapProgram "$out/bin/$program" \
              --unset LD_LIBRARY_PATH --unset LD_PRELOAD \
              --prefix LD_LIBRARY_PATH : "${runtime}/lib"
          done
        '';
        # Upstream's production-only check asserts the tools are absent.
        installCheckPhase = ''
          runHook preInstallCheck
          test "$($out/bin/gufo --version)" = "gufo version development (${revision})"
          $out/bin/gufo --help >/dev/null
          $out/bin/gufo-server --version >/dev/null
          $out/bin/gufo-kernel-bench --help >/dev/null
          $out/bin/tune_hipblaslt --help >/dev/null
          test -f $out/share/licenses/gufo/LICENSE
          runHook postInstallCheck
        '';
      });
      # Retain upstream's complete compiler, profiling, model-download and
      # reference-evaluation environment, including its patched ROCr paths,
      # TORCH_HOME and ccache hook. Only add our matching installed tools.
      toolsShell = gufo.devShells.${system}.default.overrideAttrs (old: {
        nativeBuildInputs = [ withTools pkgs.bashInteractive ]
          ++ (old.nativeBuildInputs or [ ]);
        shellHook = (old.shellHook or "") + ''
          export GUFO_SOURCE_REVISION=${revision}
          export GUFO_SOURCE_ROOT=${gufo.outPath}
        '';
      });
    in
    assert gufo.rev == revision;
    {
      packages.${system} = {
        default = withTools;
        gufo = withTools;
        bash-interactive = pkgs.bashInteractive;
        inherit (gufo.packages.${system})
          ds4-reference llama-cpp-reference llama-cpp-mtp-reference;
      };
      devShells.${system}.default = toolsShell;
      lib = {
        inherit revision;
        gufoSource = gufo.outPath;
        nixpkgsSource = nixpkgs.outPath;
        rocmVersion = base.toolchain.rocmVersion;
      };
    };
}
