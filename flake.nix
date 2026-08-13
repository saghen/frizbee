{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      fenix,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages =
            let
              fx = fenix.packages.${pkgs.stdenv.hostPlatform.system};
              toolchain = fx.combine [
                fx.stable.rustc
                fx.stable.cargo
                fx.stable.clippy
                fx.stable.rustfmt
                fx.stable.rust-src
                fx.rust-analyzer
                fx.targets.wasm32-unknown-unknown.stable.rust-std
                fx.targets.wasm32-wasip1.stable.rust-std
              ];
            in
            [
              toolchain
              pkgs.just
              pkgs.git-cliff

              # bindings
              pkgs.rust-cbindgen
              pkgs.maturin
              pkgs.python3
              pkgs.nodejs
              pkgs.wasm-pack
              pkgs.wasmtime
              # keep the wasm-bindgen version in bindings/frizbee-wasm updated with this
              pkgs.wasm-bindgen-cli
              pkgs.binaryen
            ];
        };
      });
    };
}
