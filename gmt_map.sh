#!/bin/bash
# Vedurstofa-style LOS deformation map from the insar_sat.jl xyz exports.
# Needs: GMT 6 classic binaries on PATH (or set GMT=/path/to/gmt) + ghostscript for ps2raster.
# Usage: bash gmt_map.sh <outdir> <label> [title] [vent_lon] [vent_lat]
#   <outdir>  directory containing disp_<label>.xyz and dem_<label>.xyz
#   <label>   pair label used by `raw pair --out` (e.g. raw_b8)
#   [title]   map title (default: LOS deformation (<label dates>))
#   [vent]    optional marker position (default: Geldingadalur -22.272 63.890)
# Region and grid spacing are derived automatically from dem_<label>.xyz,
# so the same script works for any crop size (small block or 1100x3900).
# Writes: map_<label>.png
set -e
OUT=${1:-data/ifg_out/sq}
LABEL=${2:-raw_b8}
TITLE=${3:-"LOS deformation (20210307-20210319)"}
VLON=${4:--22.272}
VLAT=${5:-63.890}
VNAME=${6:-Geldingadalur}
GMT=${GMT:-gmt}
cd "$OUT"

# Region = dem extents, grid = 201x201 nodes -> exact multiples for nearneighbor
read W E S N I < <(awk -v n=200 '
    NR==1 {w=$1; e=$1; s=$2; nn=$2}
    {if($1<w)w=$1; if($1>e)e=$1; if($2<s)s=$2; if($2>nn)nn=$2}
    END {
        w=sprintf("%.4f",w); e=sprintf("%.4f",e); s=sprintf("%.4f",s); nn=sprintf("%.4f",nn);
        ix=(e-w)/n; iy=(nn-s)/n;
        printf "%s %s %s %s %.10g/%.10g\n", w, e, s, nn, ix, iy
    }' dem_$LABEL.xyz)
R="-R$W/$E/$S/$N"
I="-I$I"
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

# Displacement: heavy Gaussian smoothing (~500 m), then mask to the
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
echo "$VLON $VLAT" | $GMT psxy $R $J -Sa0.45c -Gblack -W0.5p,white -K -O >> map_$LABEL.ps
echo "$VLON $(awk -v la=$VLAT 'BEGIN{printf "%.4f", la+0.008}') 9 0 0 LM $VNAME" \
    | $GMT pstext $R $J -K -O >> map_$LABEL.ps
$GMT psscale $R $J -Cdisp.cpt -DjCB+w12c+h+o0/0.8c -Bx2+l"LOS displacement [cm]" -K -O >> map_$LABEL.ps
$GMT psbasemap $R $J -Baf -B+t"$TITLE" -O >> map_$LABEL.ps

$GMT ps2raster map_$LABEL.ps -Tg -A -Z
echo "wrote $OUT/map_$LABEL.png"
