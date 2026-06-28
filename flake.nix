{
  inputs = {
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, fenix, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        fenixPkgs = fenix.packages.${system};

        rustToolchain = fenixPkgs.minimal.toolchain;

        # TODO: pick based on $system
        x86_64_target = "x86_64-unknown-linux-gnu";
        aarch64_target = "aarch64-unknown-linux-gnu";

        toolchainWithTargets = fenixPkgs.combine [
          rustToolchain
          fenixPkgs.targets.${x86_64_target}.latest.rust-std
          fenixPkgs.targets.${aarch64_target}.latest.rust-std
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ toolchainWithTargets ];
        };
      }
    );
}
