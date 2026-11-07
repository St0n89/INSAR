# gmt_map.jl — Vedurstofa-style LOS deformation map via GMT.jl.
#
# This is the same pipeline as gmt_map.sh (validated classic-GMT reference),
# wrapped for users who installed GMT via Julia (Pkg.add("GMT")).
# Untested on Windows: if a gmt(...) call complains about redirection,
# fall back to gmt_map.sh with the GMT binaries on PATH.
#
# Usage: julia gmt_map.jl [outdir] [label] [title] [vent_lon] [vent_lat] [vent_name]
#   outdir   directory containing disp_<label>.xyz and dem_<label>.xyz
#   label    pair label used by `raw pair --out` (e.g. raw_b8)
#   title    map title
#   vent     optional marker position (default: Geldingadalur -22.272 63.890)
# Region and grid spacing are derived automatically from dem_<label>.xyz,
# so the same script works for any crop size (small block or 1100x3900).
using GMT
using Printf

out    = length(ARGS) >= 1 ? ARGS[1] : "data/ifg_out/sq"
label  = length(ARGS) >= 2 ? ARGS[2] : "raw_b8"
title  = length(ARGS) >= 3 ? ARGS[3] : "LOS deformation (20210307-20210319)"
vlon   = length(ARGS) >= 4 ? ARGS[4] : "-22.272"
vlat   = length(ARGS) >= 5 ? ARGS[5] : "63.890"
vname  = length(ARGS) >= 6 ? ARGS[6] : "Geldingadalur"

# Region = dem extents, grid = 201x201 nodes (exact multiples for nearneighbor)
w, e, s, n = Inf, -Inf, Inf, -Inf
demfile = joinpath(out, "dem_$(label).xyz")
for line in eachline(demfile)
    c = split(line)
    length(c) >= 2 || continue
    lo = parse(Float64, c[1]); la = parse(Float64, c[2])
    lo < w && (w = lo); lo > e && (e = lo)
    la < s && (s = la); la > n && (n = la)
end
w = round(w, digits = 4); e = round(e, digits = 4)
s = round(s, digits = 4); n = round(n, digits = 4)
ix = (e - w) / 200; iy = (n - s) / 200
R = @sprintf("-R%.4f/%.4f/%.4f/%.4f", w, e, s, n)
I = @sprintf("-I%.10g/%.10g", ix, iy)
J = "-JM14c"
@printf("region %s  grid %s\n", R, I)

cd(out) do
    # Minimum-curvature splines for DEM and displacement
    gmt("surface dem_$(label).xyz $R $I -Gdem_$(label).nc -T0.35 -C0.01 -V0")
    gmt("surface disp_$(label).xyz $R $I -Gdisp_$(label).nc -T0.35 -C0.01 -V0")

    # Background: smoothed DEM + hillshade
    gmt("grdfilter dem_$(label).nc -Dp -Fg5 -Gdems_$(label).nc")
    gmt("grdgradient dems_$(label).nc -Ne0.6 -A315 -Gshade_$(label).nc")
    open("bg.cpt", "w") do io
        write(io, "0   208 224 238   0.1 208 224 238\n" *
                  "0.1 232 232 228   380  250 248 244\n")
    end

    # Displacement: heavy smoothing, mask to coherent data support, fill small holes
    gmt("makecpt -Cjet -T-8/8/1 -Z > disp.cpt")
    gmt("grdfilter disp_$(label).nc -Dp -Fg9 -Gdispf_$(label).nc")
    gmt("nearneighbor disp_$(label).xyz $R $I -S0.006 -Gnear_$(label).nc")
    gmt("grdmath near_$(label).nc ISNAN 1 SUB -1 MUL 0 NAN = mask_$(label).nc")
    gmt("grdmath dispf_$(label).nc mask_$(label).nc MUL = dispm_$(label).nc")
    gmt("grdfill dispm_$(label).nc -An10 -Gdispmf_$(label).nc")

    # Compose
    gmt("grdimage dems_$(label).nc -Cbg.cpt -Ishade_$(label).nc $R $J -K -P > map_$(label).ps")
    gmt("pscoast $R $J -S208,224,238 -Df -K -O >> map_$(label).ps")
    gmt("grdimage dispmf_$(label).nc -Cdisp.cpt $R $J -Q -K -O >> map_$(label).ps")
    gmt("pscoast $R $J -W0.75p -Df -K -O >> map_$(label).ps")

    # Vent marker + label, colorbar, frame
    open("star_$(label).txt", "w") do io
        write(io, "$(vlon) $(vlat)\n")
    end
    open("lab_$(label).txt", "w") do io
        @printf(io, "%s %.4f 9 0 0 LM %s\n", vlon, parse(Float64, vlat) + 0.008, vname)
    end
    gmt("psxy star_$(label).txt $R $J -Sa0.45c -Gblack -W0.5p,white -K -O >> map_$(label).ps")
    gmt("pstext lab_$(label).txt $R $J -K -O >> map_$(label).ps")
    gmt("psscale $R $J -Cdisp.cpt -DjCB+w12c+h+o0/0.8c -Bx2+l\"LOS displacement [cm]\" -K -O >> map_$(label).ps")
    gmt("psbasemap $R $J -Baf -B+t\"$title\" -O >> map_$(label).ps")

    gmt("ps2raster map_$(label).ps -Tg -A")
    @info "wrote $(joinpath(out, "map_$(label).png"))"
end
