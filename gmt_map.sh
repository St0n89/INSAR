#!/bin/bash
# Vedurstofa-style LOS deformation map from the insar_sat.jl xyz exports.
# Needs: GMT 6 classic binaries on PATH (or set GMT=/path/to/gmt) + ghostscript for ps2raster.
# Usage: bash gmt_map.sh <outdir> <label> [title]
#   <outdir> directory containing disp_<label>.xyz and dem_<label>.xyz
#   <label>  pair label used by raw ifg --out (e.g. raw_b8)
# Writes: map_<label>.png
set -e
OUT=${1:-data/ifg_out/sq}
LABEL=${2:-raw_b8}
TITLE=${3:-"LOS deformation (20210307-20210319)"}
GMT=${GMT:-gmt}
cd "$OUT"
# Region/incidence: 0.22 deg lon x 0.118 deg lat, 201x201 nodes (~55x65 m)
R="-R-22.39/-22.17/63.826/63.944"
I="-I0.0011/0.00059"
J="-JM14c"

# Minimum-curvature splines for DEM and displacement
$GMT surface dem_$LABEL.xyz $R $I -Gdem_$LABEL.nc -T0.35 -C0.01 -V0
$GMT surface disp_$LABEL.xyz $R $I -Gdisp_$LABEL.nc -T0.35 -C0.01 -V0

# Background: smoothed DEM + hillshade, ocean/land gray-scale cpt
$GMT grdfilter dem_$LABEL.nc -Dp -Fg5 -Gdems_$LABEL.nc
$GMT grdgradient dems_$LABEL.nc -Ne0.6 -A315 -Gshade_$LABEL.nc
cat > bg.cpt << 'CPTEOF'
0   208 224 238   0.1 208 224 238
0.1 232 232 228   380  250 248 244
CPTEOF

# Displacement: heavy Gaussian smoothing (g9 ~ 500 m), then mask to the
# coherent data support (nearneighbor radius 0.006 deg ~ 660 m) so the
# ocean and never-covered areas stay transparent, then fill small holes.
$GMT makecpt -Cjet -T-8/8/1 -Z > disp.cpt
$GMT grdfilter disp_$LABEL.nc -Dp -Fg9 -Gdispf_$LABEL.nc
$GMT nearneighbor disp_$LABEL.xyz $R $I -S0.006 -Gnear_$LABEL.nc
$GMT grdmath near_$LABEL.nc ISNAN 1 SUB -1 MUL 0 NAN = mask_$LABEL.nc
$GMT grdmath dispf_$LABEL.nc mask_$LABEL.nc MUL = dispm_$LABEL.nc
$GMT grdfill dispm_$LABEL.nc -An10 -Gdispmf_$LABEL.nc

# Compose: DEM hillshade -> ocean fill -> displacement (-Q NaN transparent) -> coastline
$GMT grdimage dems_$LABEL.nc -Cbg.cpt -Ishade_$LABEL.nc $R $J -K -P > map_$LABEL.ps
$GMT pscoast $R $J -S208,224,238 -Df -K -O >> map_$LABEL.ps
$GMT grdimage dispmf_$LABEL.nc -Cdisp.cpt $R $J -Q -K -O >> map_$LABEL.ps
$GMT pscoast $R $J -W0.75p -Df -K -O >> map_$LABEL.ps

# Vent marker + label, colorbar, frame/title
echo "-22.272 63.890" | $GMT psxy $R $J -Sa0.45c -Gblack -W0.5p,white -K -O >> map_$LABEL.ps
echo "-22.272 63.898 9 0 0 LM Geldingadalur" | $GMT pstext $R $J -K -O >> map_$LABEL.ps
$GMT psscale $R $J -Cdisp.cpt -DjCB+w12c+h+o0/0.8c -Bx2+l"LOS displacement [cm]" -K -O >> map_$LABEL.ps
$GMT psbasemap $R $J -Baf -B+t"$TITLE" -O >> map_$LABEL.ps

$GMT ps2raster map_$LABEL.ps -Tg -A -Z
echo "wrote $OUT/map_$LABEL.png"
