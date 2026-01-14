{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, utils }: utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3
          python313Packages.matplotlib
          hwloc
          libgcc
          gnumake
          pkg-config
          openblas
          R
          rPackages.RColorBrewer
          rPackages.FactoMineR
          rPackages.caret
          rPackages.ROCR
          rPackages.klaR
          rPackages.gridExtra
          rstudio

        ];
      };
    }
  );
}
