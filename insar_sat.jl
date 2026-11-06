#!/usr/bin/env julia
# ============================================================================
#
#   insar_sat.jl -- satellite imagery & InSAR fringe simulator
#   ---------------------------------------------------------
#   Pure Julia standard library (Printf, Random, Downloads). No packages.
#
#   * Renders natural-colour "satellite pictures" from synthetic fractal
#     terrain, or from REAL elevation data (Open-Meteo elevation API,
#     free, no key: Etna, Fuji, Matterhorn, Reykjanes, Grand Canyon...).
#   * Simulates SAR amplitude images (land-cover backscatter + Rayleigh
#     speckle with N-look averaging).
#   * Simulates SAR interferograms (InSAR): deformation sources (Mogi,
#     dyke, sill, strike-slip), topographic phase from a perpendicular
#     baseline, atmospheric delay screens, DEM-error fringes, and
#     coherence-dependent decorrelation noise. Wrapped phase is colored
#     with the classic cyclic InSAR rainbow (1 fringe = lambda/2 of LOS
#     displacement). Coherence map + unwrapped model field + SAR
#     amplitude are written alongside.
#   * All images are hand-rolled 24-bit BMP (no image package needed).
#
#   Usage:
#     julia insar_sat.jl                  # demo: synthetic volcano island
#     julia insar_sat.jl etna             # real DEM + summit uplift
#     julia insar_sat.jl terrain --place fuji --noclouds
#     julia insar_sat.jl insar --source slip --cm 20
#     julia insar_sat.jl places           # list real-data places
#     julia insar_sat.jl help             # full option list
#
#   Tips: run with `julia -t auto insar_sat.jl ...` for threads.
#   Real DEMs are cached in the output dir; fetches are resumable --
#   if a fetch is cut short (rate limit, --maxsec budget, Ctrl-C + rerun),
#   progress is kept in dem_<place>.txt.part and the same command resumes.
#   Phase convention: 1 fringe = lambda/2 (2.77 cm for C-band Sentinel-1).
#
#   The deformation models are deliberately simple (Mogi point sources,
#   a screw-dislocation approximation for strike-slip). This is a fringe
#   picture generator, not a geophysical inversion engine.
#
# ============================================================================

using Printf
using Random
using Downloads

# ----------------------------------------------------------------------------
# 1. Small helpers
# ----------------------------------------------------------------------------

clamp01(x) = clamp(x, 0.0, 1.0)
lerp(a, b, t) = a + (b - a) * t
smooth01(t) = (t = clamp01(t); t * t * (3.0 - 2.0 * t))

@inline function lerp3(c1::NTuple{3,Float64}, c2::NTuple{3,Float64}, t::Float64)
    (lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t), lerp(c1[3], c2[3], t))
end

to8(x) = UInt8(clamp(round(x), 0.0, 255.0))

# ----------------------------------------------------------------------------
# 2. BMP writer (24-bit, BGR, bottom-up, 54-byte header)
# ----------------------------------------------------------------------------

function write_bmp(path::String, img::Array{UInt8,3})
    H, W = size(img, 1), size(img, 2)
    rowsz = 4 * cld(3W, 4)
    datasz = rowsz * H
    row = Vector{UInt8}(undef, rowsz)
    open(path, "w") do io
        write(io, UInt8('B'), UInt8('M'))
        write(io, UInt32(54 + datasz), UInt32(0), UInt32(54))     # size, resv, off
        write(io, UInt32(40), Int32(W), Int32(H))                 # DIB hdr, dims
        write(io, UInt16(1), UInt16(24))                          # planes, bpp
        write(io, UInt32(0), UInt32(datasz))                      # compr, img size
        write(io, Int32(2835), Int32(2835))                       # ppm
        write(io, UInt32(0), UInt32(0))                           # colors used/imp
        for y in H:-1:1
            fill!(row, 0x00)
            k = 1
            @inbounds for x in 1:W
                row[k]     = img[y,x,3]
                row[k + 1] = img[y,x,2]
                row[k + 2] = img[y,x,1]
                k += 3
            end
            write(io, row)
        end
    end
    println("  wrote ", path, "  (", W, "x", H, ", ",
            round(Int, stat(path).size / 1024), " KiB)")
end

# ----------------------------------------------------------------------------
# 3. Hash-based value noise / fBm / ridged noise (deterministic, seeded)
# ----------------------------------------------------------------------------

struct NG
    seed::UInt64
end

@inline function lhash(g::NG, xi::Int, yi::Int)::Float64
    xu = reinterpret(UInt64, Int64(xi))
    yu = reinterpret(UInt64, Int64(yi))
    u = (xu * 0xD6E8FEB86659FD93) ⊻ (yu * 0xA24BAED4963EE407) ⊻ g.seed
    u = (u ⊻ (u >> 30)) * 0xBF58476D1CE4E5B9
    u = (u ⊻ (u >> 27)) * 0x94D049BB133111EB
    u = u ⊻ (u >> 31)
    return (u % UInt32) * (1.0 / 4294967295.0)
end

@inline function vnoise(g::NG, x::Float64, y::Float64)::Float64
    x0 = floor(Int, x); y0 = floor(Int, y)
    fx = x - x0;       fy = y - y0
    sx = fx * fx * (3.0 - 2.0fx)
    sy = fy * fy * (3.0 - 2.0fy)
    v00 = lhash(g, x0, y0);     v10 = lhash(g, x0 + 1, y0)
    v01 = lhash(g, x0, y0 + 1); v11 = lhash(g, x0 + 1, y0 + 1)
    a = v00 + sx * (v10 - v00)
    b = v01 + sx * (v11 - v01)
    return a + sy * (b - a)
end

function fbm(g::NG, x::Float64, y::Float64; oct::Int = 6, lac::Float64 = 2.03,
             gain::Float64 = 0.5)
    s = 0.0; amp = 1.0; tot = 0.0; f = 1.0
    for _ in 1:oct
        s += amp * vnoise(g, x * f, y * f)
        tot += amp; amp *= gain; f *= lac
    end
    return s / tot
end

function ridged(g::NG, x::Float64, y::Float64; oct::Int = 6,
                lac::Float64 = 2.1, gain::Float64 = 0.55)
    s = 0.0; amp = 1.0; tot = 0.0; f = 1.0; prev = 1.0
    for _ in 1:oct
        n = vnoise(g, x * f, y * f)
        r = 1.0 - abs(2.0n - 1.0)
        r *= r
        s += amp * r * prev
        prev = r
        tot += amp; amp *= gain; f *= lac
    end
    return s / tot
end

# ----------------------------------------------------------------------------
# 4. Colormaps
# ----------------------------------------------------------------------------

# Cyclic "classic InSAR rainbow" from anchor stops (blue-cyan-green-yellow-red-magenta)
const INSAR_STOPS = [
    (0.00, (0.16, 0.13, 0.55)),
    (0.14, (0.00, 0.50, 0.85)),
    (0.30, (0.00, 0.62, 0.42)),
    (0.50, (0.32, 0.76, 0.00)),
    (0.66, (0.95, 0.85, 0.05)),
    (0.80, (0.92, 0.30, 0.05)),
    (0.92, (0.72, 0.08, 0.45)),
    (1.00, (0.16, 0.13, 0.55)),
]

const COH_STOPS = [
    (0.00, (0.06, 0.01, 0.02)),
    (0.30, (0.48, 0.05, 0.05)),
    (0.60, (0.95, 0.80, 0.15)),
    (1.00, (1.00, 1.00, 1.00)),
]

function sample_stops(stops, t::Float64)
    t = t - floor(t)
    for k in 2:length(stops)
        if t <= stops[k][1] + 1e-12
            t0, c0 = stops[k-1]; t1, c1 = stops[k]
            f = (t - t0) / max(t1 - t0, 1e-12)
            return lerp3(c0, c1, clamp01(f))
        end
    end
    return stops[end][2]
end

insar_cmap(t::Float64) = sample_stops(INSAR_STOPS, t) .* 255.0
coh_cmap(t::Float64)   = sample_stops(COH_STOPS, clamp01(t)) .* 255.0

# ----------------------------------------------------------------------------
# 5. Hillshade & slope (x: i -> east, y: row j -> south; north = j-1)
# ----------------------------------------------------------------------------

function hillshade(dem::Matrix{Float64}, px::Float64; az::Float64 = 315.0,
                   el::Float64 = 42.0)
    H, W = size(dem)
    a = deg2rad(az); e = deg2rad(el)
    lx = cos(e) * sin(a); ly = cos(e) * cos(a); lz = sin(e)
    sh = Matrix{Float64}(undef, H, W)
    sl = Matrix{Float64}(undef, H, W)
    for j in 1:H, i in 1:W
        hx = (i < W ? dem[j,i+1] : dem[j,i]) - (i > 1 ? dem[j,i-1] : dem[j,i])
        hn = (j > 1 ? dem[j-1,i] : dem[j,i]) - (j < H ? dem[j+1,i] : dem[j,i])
        gx = hx / (2px); gy = hn / (2px)
        d = (-gx * lx - gy * ly + lz) / sqrt(gx^2 + gy^2 + 1.0)
        sh[j,i] = clamp(d, 0.0, 1.0)
        sl[j,i] = hypot(gx, gy)
    end
    return sh, sl
end

# ----------------------------------------------------------------------------
# 6. Synthetic terrain (volcanic island)
# ----------------------------------------------------------------------------

function synthetic_dem(W::Int, H::Int; seed::Int = 1, relief::Float64 = 1900.0,
                       waterfrac::Float64 = 0.30)
    gw = NG(UInt64(seed));        g1 = NG(UInt64(seed + 101))
    g2 = NG(UInt64(seed + 202));  g3 = NG(UInt64(seed + 303))
    g4 = NG(UInt64(seed + 404))
    dem = Matrix{Float64}(undef, H, W)
    Threads.@threads for j in 1:H
        for i in 1:W
            u = (i - 1) / W; v = (j - 1) / H
            wx = u + 0.22 * fbm(g1, 2.5u + 11.3, 2.5v + 7.7; oct = 4) - 0.11
            wy = v + 0.22 * fbm(g2, 2.5u + 4.2, 2.5v + 9.1; oct = 4) - 0.11
            b    = fbm(gw, 3.1wx, 3.1wy; oct = 7)
            rid  = ridged(g3, 2.2wx + 3.0, 2.2wy + 8.0; oct = 6)
            r    = sqrt((u - 0.5)^2 + (v - 0.5)^2) / 0.5
            coast = fbm(g4, 5.0u, 5.0v; oct = 5)
            mask  = clamp01(1.15 - 1.25r + 0.22 * (coast - 0.5))
            cone  = max(0.0, 1.0 - 1.55r)^1.7
            h = relief * (0.78 * b^2.1 * (0.45 + 0.85rid) * mask +
                          0.85 * cone * (0.85 + 0.45rid)) - 0.22relief
            dem[j,i] = h
        end
    end
    s = sort(vec(dem))
    wl = s[max(1, round(Int, clamp01(waterfrac) * length(s)))]
    return dem .- wl
end

# ----------------------------------------------------------------------------
# 7. Real DEMs (Open-Meteo elevation API, free, no key) + places table
# ----------------------------------------------------------------------------

struct TerrainStyle
    treeline::Float64
    snowline::Float64
    sea::Bool
    farmland::Bool
    farmmax::Float64
    rock::NTuple{3,Float64}
    low::NTuple{3,Float64}
    mid::NTuple{3,Float64}
end

struct PlaceInfo
    lat1::Float64; lat2::Float64; lon1::Float64; lon2::Float64
    summit::Tuple{Float64,Float64}
    depth::Float64          # default Mogi depth [m]
    style::TerrainStyle
    desc::String
end

const PLACES = Dict{String,PlaceInfo}(
    "etna" => PlaceInfo(37.62, 37.88, 14.72, 15.12, (37.751, 14.993), 5000.0,
        TerrainStyle(1500.0, 2850.0, true, true, 420.0,
            (96.0, 88.0, 80.0), (92.0, 112.0, 58.0), (124.0, 112.0, 72.0)),
        "Mt Etna, Sicily (stratovolcano)"),
    "fuji" => PlaceInfo(35.25, 35.47, 138.55, 138.95, (35.361, 138.728), 9000.0,
        TerrainStyle(1700.0, 2900.0, false, true, 350.0,
            (128.0, 116.0, 105.0), (72.0, 108.0, 54.0), (112.0, 98.0, 62.0)),
        "Mt Fuji, Japan"),
    "matterhorn" => PlaceInfo(45.87, 46.05, 7.60, 7.85, (45.976, 7.658), 8000.0,
        TerrainStyle(2300.0, 3100.0, false, false, 0.0,
            (108.0, 100.0, 92.0), (88.0, 116.0, 60.0), (124.0, 112.0, 78.0)),
        "Zermatt / Matterhorn, Alps"),
    "reykjanes" => PlaceInfo(63.83, 64.02, -22.60, -22.00, (63.890, -22.270), 3000.0,
        TerrainStyle(150.0, 99999.0, true, false, 0.0,
            (74.0, 72.0, 68.0), (86.0, 110.0, 70.0), (76.0, 96.0, 64.0)),
        "Reykjanes peninsula, Iceland (dyke swarms)"),
    "grandcanyon" => PlaceInfo(36.02, 36.30, -112.30, -111.90, (36.100, -112.100), 6000.0,
        TerrainStyle(99999.0, 99999.0, false, false, 0.0,
            (150.0, 105.0, 75.0), (110.0, 105.0, 70.0), (140.0, 100.0, 70.0)),
        "Grand Canyon, Arizona"),
)

function parse_elevations(s::String)
    occursin("\"error\"", s) && error("elevation API error: ", s[1:min(end,200)])
    a = findfirst('[', s)
    b = findlast(']', s)
    (a === nothing || b === nothing) && error("unexpected API response")
    vals = Float64[]
    for tok in split(s[a+1:b-1], ',')
        t = strip(tok)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    return vals
end

# one API call with polite 429 handling (the anonymous elevation endpoint is
# minutely rate-limited; on 429 we pause a full minute and continue)
function fetch_chunk(url::String, outdir::String; attempts::Int = 6)
    tmp = joinpath(outdir, ".tmp_elev")
    last_err = ""
    for attempt in 1:attempts
        local s = ""
        try
            Downloads.download(url, tmp)
            s = read(tmp, String)
        catch err
            last_err = sprint(showerror, err)
            println("    download hiccup (attempt ", attempt, "/", attempts,
                    "), backing off...")
            sleep(attempt == 1 ? 10.0 : 40.0)
            continue
        end
        rm(tmp; force = true)
        if occursin("\"error\"", s)
            if occursin("try again", lowercase(s))
                println("    rate-limited by the API, pausing 65 s...")
                sleep(65.0)
                continue
            end
            error("elevation API error: ", s[1:min(end, 120)])
        end
        return parse_elevations(s)
    end
    error("download failed after ", attempts, " attempts: ", last_err)
end

function fetch_dem(pi::PlaceInfo, place::String, H::Int, outdir::String;
                   fast::Bool = false, maxsec::Float64 = 0.0)
    # aspect-correct width from latitude/longitude span
    dlat = pi.lat2 - pi.lat1
    dlon = (pi.lon2 - pi.lon1) * cosd(0.5 * (pi.lat1 + pi.lat2))
    W = clamp(round(Int, H * abs(dlon) / max(abs(dlat), 1e-9)), 32, 200)
    cache = joinpath(outdir, "dem_" * place * ".txt")
    if isfile(cache)
        lines = split(read(cache, String), '\n')
        hd = split(lines[1]); ch = parse(Int, hd[1]); cw = parse(Int, hd[2])
        if ch == H && cw == W
            dem = Matrix{Float64}(undef, H, W)
            ok = true
            for j in 1:H
                row = split(strip(lines[j+1]))
                length(row) == W || (ok = false; break)
                dem[j, :] .= parse.(Float64, row)
            end
            ok && (println("  cached DEM: ", place, " (", H, "x", W, ")");
                   return dem, W)
        end
    end
    println("  fetching elevation grid ", H, "x", W, " for ", pi.desc,
            " (Open-Meteo, ", cld(H * W, 100), " requests)...")
    lats = collect(range(pi.lat2, pi.lat1; length = H))   # row 1 = north
    lons = collect(range(pi.lon1, pi.lon2; length = W))
    n = H * W
    # ---- resumable partial progress ----------------------------------------
    part = cache * ".part"
    vals = Float64[]
    if isfile(part)
        pl = split(read(part, String), '\n')
        if length(pl) >= 2
            hd = split(pl[1]); ph = parse(Int, hd[1]); pw = parse(Int, hd[2])
            if ph == H && pw == W
                vals = [parse(Float64, x) for x in split(pl[2]) if !isempty(x)]
                println("  resuming at point ", length(vals) + 1, " / ", n)
            end
        end
    end
    t0 = time()
    gap = fast ? 0.6 : 5.0
    k = length(vals) + 1
    while k <= n
        if maxsec > 0 && time() - t0 > maxsec && k > length(vals) + 1
            println("  time budget reached (", round(Int, time() - t0),
                    " s) -- progress saved, rerun the same command to resume")
            return nothing, 0
        end
        ke = min(k + 99, n)
        ls = String[]; os = String[]
        for t in k:ke
            push!(ls, string(lats[div(t - 1, W) + 1]))
            push!(os, string(lons[(t - 1) % W + 1]))
        end
        url = "https://api.open-meteo.com/v1/elevation?latitude=" *
              join(ls, ",") * "&longitude=" * join(os, ",")
        append!(vals, fetch_chunk(url, outdir))
        open(part, "w") do io
            println(io, H, " ", W)
            println(io, join(vals, ' '))
        end
        k = ke + 1
        if k <= n
            @printf("    %3d %%\r", min(100, round(Int, 100ke / n)))
            flush(stdout)
            sleep(gap)
        end
    end
    println("    100 %")
    length(vals) == n || error("API returned ", length(vals), " of ", n, " values")
    dem = reshape(vals, W, H)'                       # row-major -> H x W
    rm(part; force = true)
    open(cache, "w") do io
        println(io, H, " ", W)
        for j in 1:H
            println(io, join(dem[j, :], ' '))
        end
    end
    println("  DEM cached as ", cache)
    return dem, W
end

# bilinear upsample (for rendering real grids at higher resolution)
function upsample(dem::Matrix{Float64}, H2::Int, W2::Int)
    H, W = size(dem)
    out = Matrix{Float64}(undef, H2, W2)
    @inbounds for j in 1:H2, i in 1:W2
        y = 1 + (H - 1) * (j - 1) / (H2 - 1)
        x = 1 + (W - 1) * (i - 1) / (W2 - 1)
        y0 = floor(Int, y); x0 = floor(Int, x)
        y1 = min(y0 + 1, H); x1 = min(x0 + 1, W)
        fy = y - y0; fx = x - x0
        a = dem[y0,x0] * (1-fx) + dem[y0,x1] * fx
        b = dem[y1,x0] * (1-fx) + dem[y1,x1] * fx
        out[j,i] = a * (1-fy) + b * fy
    end
    return out
end

# ----------------------------------------------------------------------------
# 8. Natural-colour satellite rendering
# ----------------------------------------------------------------------------

function render_satellite(dem::Matrix{Float64}, px::Float64, seed::Int,
                          style::TerrainStyle; clouds::Bool = true,
                          relief::Float64 = 1900.0)
    H, W = size(dem)
    sh, sl = hillshade(dem, px)
    g_m = NG(UInt64(seed) + 0x900); g_f = NG(UInt64(seed) + 0x950)
    g_c = NG(UInt64(seed) + 0x970); g_g = NG(UInt64(seed) + 0x990)
    img = Array{UInt8}(undef, H, W, 3)
    snowfrac = Ref(0.0); landfrac = Ref(0.0); npix = Ref(0)
    Threads.@threads for j in 1:H
        snow = 0; land = 0; cnt = 0
        for i in 1:W
            h = dem[j,i]; s = sl[j,i]; shd = sh[j,i]
            r = 0.0; g = 0.0; b = 0.0
            if h <= 0.0 && style.sea
                d = min(-h, 260.0)
                t = (d / 260.0)^0.6
                r = lerp(38.0, 9.0, t); g = lerp(118.0, 35.0, t); b = lerp(142.0, 66.0, t)
                if d < 30.0
                    gl = vnoise(g_g, 0.55i, 0.55j)^4 * 70.0
                    r += gl; g += gl; b += gl * 0.6
                end
            else
                land += 1
                moist = fbm(g_m, 0.013i, 0.013j; oct = 5)
                n2 = vnoise(g_f, 0.7i, 0.7j)
                treeline = style.treeline; snowline = style.snowline
                t = smooth01(h / max(treeline, 1.0))
                r, g, b = lerp3(style.low, style.mid, t)
                # farmland patchwork in the lowlands
                if style.farmland && h < style.farmmax && s < 0.35
                    p = vnoise(g_f, 0.35i + 17.0, 0.35j + 3.0)
                    if     p < 0.35; r, g, b = 124.0, 138.0, 72.0
                    elseif p < 0.55; r, g, b = 96.0, 130.0, 56.0
                    elseif p < 0.80; r, g, b = 150.0, 148.0, 88.0
                    end
                end
                # forests (moisture noise)
                if moist > 0.55 + 0.08n2 && h < treeline && s < 0.75
                    f = 0.55
                    r = lerp(r, 34.0, f); g = lerp(g, 74.0, f); b = lerp(b, 40.0, f)
                end
                # rock above treeline / on steep slopes (+ canyon strata banding)
                rockt = max(smooth01((h - treeline * (1.0 + 0.25(moist - 0.5))) / 600.0),
                            smooth01((s - 0.58 - 0.15n2) / 0.12))
                if treeline > 99000.0
                    rockt = smooth01((s - 0.30) / 0.15)      # bare-rock place
                end
                if rockt > 0.0
                    band = 0.5 + 0.5 * sin(h / 55.0)
                    rr = lerp3(style.rock, style.mid, 0.30 * band)
                    r = lerp(r, rr[1], rockt); g = lerp(g, rr[2], rockt); b = lerp(b, rr[3], rockt)
                end
                # beach
                if h < 10.0 + 12.0n2 && style.sea
                    r = lerp(r, 196.0, 0.8); g = lerp(g, 180.0, 0.8); b = lerp(b, 132.0, 0.8)
                end
                # snow
                if h > snowline * (1.0 + 0.07(n2 - 0.5)) ||
                   (snowline < 99000.0 && h > snowline * 0.94 && s < 0.35)
                    sf = clamp01((h - snowline * 0.92) / 200.0) * (1.0 - 0.35smooth01(s))
                    r = lerp(r, 247.0, sf); g = lerp(g, 250.0, sf); b = lerp(b, 254.0, sf)
                    sf > 0.3 && (snow += 1)
                end
                # hillshade + aerial haze
                f = 0.40 + 0.80shd
                r *= f; g *= f; b *= f
                hz = 0.10 * (1.0 - clamp01(h / max(relief, 1.0)))
                r = lerp(r, 188.0, hz); g = lerp(g, 204.0, hz); b = lerp(b, 224.0, hz)
            end
            # thin cirrus overlay
            if clouds
                c = fbm(g_c, 0.008i + 40.0, 0.008j + 40.0; oct = 5)
                if c > 0.60
                    a = min((c - 0.60) * 2.8, 0.75)
                    r = lerp(r, 252.0, a); g = lerp(g, 252.0, a); b = lerp(b, 255.0, a)
                end
            end
            img[j,i,1] = to8(r); img[j,i,2] = to8(g); img[j,i,3] = to8(b)
            cnt += 1
        end
        snowfrac[] += snow; landfrac[] += land; npix[] += cnt
    end
    stats = (snow = snowfrac[], land = landfrac[], npix = npix[])
    return img, stats
end

# ----------------------------------------------------------------------------
# 9. SAR amplitude simulation (land-cover backscatter + speckle)
# ----------------------------------------------------------------------------

function backscatter_class(h::Float64, s::Float64, style::TerrainStyle,
                           moist::Float64, n2::Float64)
    (h <= 0.0 && style.sea) && return 0.06
    h < 12.0 + 10.0n2 && style.sea && return 0.15
    if style.snowline < 99000.0 &&
       (h > style.snowline * (1.0 + 0.07(n2 - 0.5)))
        return 0.22
    end
    s > 0.58 + 0.15n2 && return 0.38                     # rock
    moist > 0.55 + 0.08n2 && h < style.treeline && return 0.50   # forest
    return 0.26                                           # grassland
end

function render_sar(dem::Matrix{Float64}, px::Float64, seed::Int,
                    style::TerrainStyle; looks::Int = 4)
    H, W = size(dem)
    sh, sl = hillshade(dem, px)
    g_m = NG(UInt64(seed) + 0x900); g_f = NG(UInt64(seed) + 0x950)
    g_t = NG(UInt64(seed) + 0x9A0)
    img = Array{UInt8}(undef, H, W, 3)
    Threads.@threads for j in 1:H
        for i in 1:W
            h = dem[j,i]; s = sl[j,i]
            moist = fbm(g_m, 0.013i, 0.013j; oct = 5)
            n2 = vnoise(g_f, 0.7i, 0.7j)
            base = backscatter_class(h, s, style, moist, n2)
            tex = 1.0 + 0.55(vnoise(g_t, 0.9i, 0.9j) - 0.5)
            speckle = 0.0
            for _ in 1:looks
                speckle += randexp()
            end
            speckle /= looks
            amp = base * tex * speckle * (0.55 + 0.45sh[j,i])
            v = 255.0 * clamp01(amp^0.42 / 1.15)
            img[j,i,1] = to8(v); img[j,i,2] = to8(v); img[j,i,3] = to8(v)
        end
    end
    return img
end

# ----------------------------------------------------------------------------
# 10. Deformation sources -> LOS displacement field [m]
#     LOS unit vector points from ground toward the satellite.
# ----------------------------------------------------------------------------

function los_vector(heading_deg::Float64, inc_deg::Float64)
    θ = deg2rad(inc_deg)
    ψ = deg2rad(heading_deg + 90.0)     # radar looks 90 deg right of flight
    return (sind(θ) * sind(ψ), sind(θ) * cosd(ψ), cosd(θ))
end

function source_field!(dlos::Matrix{Float64}, W::Int, H::Int, px::Float64,
                       source::String, xc::Float64, yc::Float64,
                       p::Dict{String,Float64}, l::NTuple{3,Float64})
    # xc, yc: source center in pixel coords (x: 1..W east, y: 1..H south)
    depth = get(p, "depth", 3500.0)
    l_e, l_n, l_u = l
    if source == "mogi"
        Threads.@threads for j in 1:H
            for i in 1:W
                dx = (i - xc) * px
                dy = (yc - j) * px              # north displacement
                r2 = dx*dx + dy*dy
                r = sqrt(r2)
                d3 = (r2 + depth^2)^1.5
                uz = depth^3 / d3               # unit u_z(0) scale
                ur = depth^2 * r / d3
                d = l_u * uz + (r > 1e-9 ? (l_e * ur * dx + l_n * ur * dy) / r : 0.0)
                dlos[j,i] = d
            end
        end
    elseif source == "dyke" || source == "sill"
        ns = source == "dyke" ? 12 : 16
        seg = source == "dyke" ? (1.0, 0.0) : (0.0, 1.0)
        pts = NTuple{3,Float64}[]
        for k in 1:ns
            t = (k - 0.5) / ns - 0.5
            push!(pts, (xc + seg[1] * t * (0.6W * px),
                        yc + seg[2] * t * (0.6H * px), depth))
        end
        Threads.@threads for j in 1:H
            for i in 1:W
                d = 0.0
                for (sx, sy, sd) in pts
                    dx = (i - sx) * px
                    dy = (sy - j) * px
                    r2 = dx*dx + dy*dy
                    r = sqrt(r2)
                    d3 = (r2 + sd^2)^1.5
                    uz = sd^3 / d3
                    ur = sd^2 * r / d3
                    d += l_u * uz + (r > 1e-9 ? (l_e * ur * dx + l_n * ur * dy) / r : 0.0)
                end
                dlos[j,i] = d / ns
            end
        end
    elseif source == "slip"
        # finite vertical strike-slip fault, screw-dislocation approximation
        strike = get(p, "strike", 35.0)
        L = get(p, "length", 8000.0)          # half-length [m]
        α = deg2rad(strike)
        ae = sind(α); an = cosd(α)            # along-strike unit (east, north)
        pe = cosd(α); pn = -sind(α)          # perpendicular unit
        Threads.@threads for j in 1:H
            for i in 1:W
                xm = (i - xc) * px
                ym = (yc - j) * px
                ξ = xm * ae + ym * an
                x = xm * pe + ym * pn
                ax = abs(x) + px
                taper = clamp((atan((ξ + L) / ax) - atan((ξ - L) / ax)) / π, 0.0, 1.0)
                u = atan(x / depth) / π        # surface profile of buried screw
                d = (l_e * ae + l_n * an) * u * taper
                dlos[j,i] = d
            end
        end
    elseif source == "none"
        fill!(dlos, 0.0)
    else
        error("unknown source '", source, "' (mogi | dyke | sill | slip | none)")
    end
    return dlos
end

# ----------------------------------------------------------------------------
# 11. Interferogram assembly & rendering
# ----------------------------------------------------------------------------

function render_insar(dem::Matrix{Float64}, px::Float64, seed::Int,
                      style::TerrainStyle, cfg::NamedTuple)
    H, W = size(dem)
    l = los_vector(cfg.heading, cfg.inc)
    lam = cfg.lam_cm / 100.0                       # wavelength [m]
    dlos = Matrix{Float64}(undef, H, W)
    source_field!(dlos, W, H, px, cfg.source, cfg.xc, cfg.yc, cfg.srcpars, l)
    # scale so max |LOS displacement| = cfg.cm target
    m = maximum(abs, dlos)
    if m > 0
        dlos .*= (cfg.cm / 100.0) / m
    end

    # --- phase model -------------------------------------------------------
    φ = (4π / lam) .* dlos                        # deformation phase [rad]
    k_topo = 4π / lam * cfg.bperp / (cfg.range * sind(cfg.inc))   # rad/m
    if cfg.bperp != 0.0
        φ .+= k_topo .* dem
        φ .+= k_topo * cfg.demerr .* dem          # DEM-error fringes
    end
    if cfg.atm > 0.0
        ga = NG(UInt64(seed) + 0xB00)
        ramp_a = cfg.atm * 0.9
        Threads.@threads for j in 1:H
            for i in 1:W
                scr = 2.6 * (fbm(ga, 0.006i, 0.006j; oct = 6) - 0.5) * 2.0
                strat = cfg.atm * 0.0065 * dem[j,i]           # stratified, elevation-correlated
                ramp = ramp_a * (2.0(i / W) + (j / H) - 1.0)  # orbit ramp
                φ[j,i] += cfg.atm * scr + strat + ramp
            end
        end
    end

    # --- coherence ----------------------------------------------------------
    coh = Matrix{Float64}(undef, H, W)
    Threads.@threads for j in 1:H
        for i in 1:W
            water = dem[j,i] <= 0.0 && style.sea
            g = 0.0
            if water
                g = 0.04
            else
                gph = φ[j,i]
                gnx = i < W ? φ[j,i+1] : gph; gpx = i > 1 ? φ[j,i-1] : gph
                gny = j > 1 ? φ[j-1,i] : gph; gsy = j < H ? φ[j+1,i] : gph
                grad = max(abs(gnx - gpx), abs(gny - gsy)) / 2
                g = cfg.coh * exp(-grad / 2.2)
                g = clamp(g, 0.05, 1.0)
            end
            coh[j,i] = g
        end
    end

    # --- noise, wrap, render ------------------------------------------------
    sh, _ = hillshade(dem, px)
    looks = 4
    img    = Array{UInt8}(undef, H, W, 3)
    img_c  = Array{UInt8}(undef, H, W, 3)
    img_u  = Array{UInt8}(undef, H, W, 3)
    Threads.@threads for j in 1:H
        for i in 1:W
            g = coh[j,i]
            σ = sqrt(max(1.0 - g^2, 0.0) / (2.0 * looks * max(g^2, 1e-6)))
            σ = clamp(σ, 0.02, 2.5)
            ph = φ[j,i] + σ * randn()
            t = mod(ph / (2π), 1.0)
            rgb = insar_cmap(t)
            if g < 0.25
                # incoherent zone: gray speckle salt-and-pepper
                v = 40.0 + 70.0 * rand()
                rgb = (v, v, v)
            else
                f = 0.45 + 0.55sh[j,i]
                rgb = (rgb[1] * f, rgb[2] * f, rgb[3] * f)
            end
            img[j,i,1] = to8(rgb[1]); img[j,i,2] = to8(rgb[2]); img[j,i,3] = to8(rgb[3])

            c = coh_cmap(g)
            img_c[j,i,1] = to8(c[1]); img_c[j,i,2] = to8(c[2]); img_c[j,i,3] = to8(c[3])

            v = 255.0 * clamp01((dlos[j,i] * 100.0 + cfg.cm) / (2cfg.cm))
            img_u[j,i,1] = to8(v); img_u[j,i,2] = to8(v); img_u[j,i,3] = to8(v)
        end
    end
    stats = (max_los_cm = maximum(abs, dlos) * 100.0,
             fringes = maximum(abs, dlos) / (lam / 2),
             mean_coh = sum(coh) / (H * W))
    return img, img_c, img_u, stats
end

# ----------------------------------------------------------------------------
# 12. CLI
# ----------------------------------------------------------------------------

function usage()
    println("""
    insar_sat.jl -- satellite imagery & InSAR fringe simulator (stdlib only)

    USAGE
      julia insar_sat.jl [command] [options]

    COMMANDS
      demo            synthetic volcanic island: satellite image, SAR
                      amplitude, interferogram, coherence, unwrapped field
      terrain         satellite picture only
      sar             SAR amplitude image only
      insar           interferogram suite only (satellite skipped)
      places          list real-elevation places
      help            this text
      <place>         shortcut, e.g. `etna` = real DEM + satellite + InSAR
                      (etna, fuji, matterhorn, reykjanes, grandcanyon)

    OPTIONS
      --size N        render size in px (default 900)
      --seed N        noise seed (default 1)
      --out DIR       output directory (default insar_out)
      --place NAME    real DEM instead of synthetic terrain
      --grid N        fetched DEM grid height in px (default 64)
      --maxsec T     stop the DEM fetch after T seconds; progress is saved
                      and the same command resumes on rerun (default: off)
      --fast          aggressive API pacing (only if you are not hitting the
                      rate limit; the tool self-throttles otherwise)
      --source S      mogi | dyke | sill | slip | none (default mogi)
      --depth M       source depth in m (default 3500 synthetic / per place)
      --cm X          max LOS displacement in cm (default 14)
      --x F --y F     source position, fraction of image (default 0.5 0.5;
                      for places the summit unless given)
      --strike D      fault strike in deg for slip (default 35)
      --length M      fault half-length in m for slip (default 8000)
      --heading D     satellite heading in deg (default 193, descending)
      --inc D         incidence angle in deg (default 35)
      --lambda CM     radar wavelength in cm (default 5.55, Sentinel-1 C)
      --bperp M       perpendicular baseline in m -> topographic fringes (0=off)
      --demerr F      DEM error fraction for bperp fringes (default 0)
      --atm F         atmospheric turbulence strength 0..1 (default 0.35)
      --coh F         baseline coherence 0..1 (default 0.9)
      --relief M      synthetic terrain relief in m (default 1900)
      --waterfrac F   synthetic sea fraction (default 0.30)
      --snowline M    override snowline elevation (per place default)
      --noclouds      disable the cirrus overlay
      --threads       note: run `julia -t auto` to use all cores""")
end

function parse_args(args)
    opts = Dict{String,String}()
    pos = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            body = a[3:end]
            if occursin('=', body)
                k, v = split(body, '='; limit = 2)
                opts[String(k)] = String(v)
            else
                k = body; v = ""
                if i < length(args) && !startswith(args[i+1], "--")
                    i += 1; v = args[i]
                end
                opts[k] = v
            end
        else
            push!(pos, a)
        end
        i += 1
    end
    return pos, opts
end

function num(opts::Dict{String,String}, key::String, dflt::Float64)
    v = get(opts, key, "")
    isempty(v) && return dflt
    try
        return parse(Float64, v)
    catch
        println("warning: bad value for --", key, ", using default")
        return dflt
    end
end

function main()
    pos, opts = parse_args(ARGS)
    cmd = isempty(pos) ? "demo" : pos[1]
    if cmd in ("help", "-h", "--help")
        usage(); return
    end
    if cmd in ("places", "list")
        println("Real-elevation places (Open-Meteo elevation API, cached after first fetch):")
        for (k, p) in sort(collect(PLACES))
            @printf("  %-12s %s\n", k, p.desc)
        end
        return
    end
    place = get(opts, "place", "")
    if haskey(PLACES, cmd)
        place = cmd
        cmd = "place"                     # terrain + insar for a real place
    elseif place != "" && !haskey(PLACES, place)
        println("unknown place '", place, "'. available: ")
        println("  ", join(sort(collect(keys(PLACES))), ", "))
        return
    end
    if cmd ∉ ("demo", "terrain", "sar", "insar", "place")
        println("unknown command '", cmd, "' -- try `help`")
        return
    end

    outdir = get(opts, "out", "insar_out")
    mkpath(outdir)
    seed = Int(num(opts, "seed", 1.0))
    clouds = !haskey(opts, "noclouds")
    println("insar_sat.jl -- mode: ", cmd, place == "" ? "" : " (" * place * ")")

    # ------------------------------------------------------------- terrain --
    local dem::Matrix{Float64}
    local style::TerrainStyle
    local px::Float64
    local prefix::String
    local pi_info::Union{PlaceInfo,Nothing} = nothing
    if place != ""
        pi_info = PLACES[place]
        style = pi_info.style
        prefix = place
        gH = clamp(Int(num(opts, "grid", 64.0)), 24, 140)
        r = fetch_dem(pi_info, place, gH, outdir;
                      fast = haskey(opts, "fast"),
                      maxsec = num(opts, "maxsec", 0.0))
        if r === nothing || r[1] === nothing
            return
        end
        dem, gW = r
        # render resolution: upscale so the larger side is ~ size (default 900)
        rsize = Int(num(opts, "size", 900.0))
        sc = max(1.0, rsize / max(gH, gW))
        rH = round(Int, gH * sc); rW = round(Int, gW * sc)
        dem = upsample(dem, rH, rW)
        px = 111320.0 * (pi_info.lat2 - pi_info.lat1) / (rH - 1)
        println("  real DEM upsampled to ", rW, "x", rH,
                "  (", round(px, digits = 1), " m/px)")
    else
        rsize = clamp(Int(num(opts, "size", 900.0)), 128, 4000)
        W = rsize; H = rsize
        relief = num(opts, "relief", 1900.0)
        waterfrac = num(opts, "waterfrac", 0.30)
        println("  generating synthetic terrain ", W, "x", H,
                " (seed ", seed, ", relief ", relief, " m)...")
        dem = synthetic_dem(W, H; seed = seed, relief = relief, waterfrac = waterfrac)
        px = 22.0
        style = TerrainStyle(1200.0, relief * 0.78, true, false, 0.0,
                             (110.0, 102.0, 92.0), (52.0, 92.0, 48.0),
                             (96.0, 84.0, 50.0))
        prefix = "synthetic"
    end
    snl = num(opts, "snowline", -1.0)
    if snl > 0
        style = TerrainStyle(style.treeline, snl, style.sea, style.farmland,
                             style.farmmax, style.rock, style.low, style.mid)
    end

    H, W = size(dem)
    hmax = maximum(dem)
    @printf("  elevation range: %.0f .. %.0f m\n", minimum(dem), hmax)

    # ------------------------------------------------------------ products --
    if cmd in ("demo", "terrain", "place")
        println("  rendering satellite image...")
        img, stats = render_satellite(dem, px, seed, style;
                    clouds = clouds, relief = hmax)
        write_bmp(joinpath(outdir, "satellite_" * prefix * ".bmp"), img)
        if stats.npix > 0
            @printf("  land %.1f %%  snow %.1f %%\n",
                    100 * stats.land / stats.npix, 100 * stats.snow / stats.npix)
        end
    end
    if cmd in ("demo", "sar")
        println("  rendering SAR amplitude...")
        img = render_sar(dem, px, seed, style)
        write_bmp(joinpath(outdir, "sar_" * prefix * ".bmp"), img)
    end
    if cmd in ("demo", "insar", "place")
        source = get(opts, "source", "mogi")
        srcpars = Dict{String,Float64}(
            "depth" => num(opts, "depth", place == "" ? 3500.0 : pi_info.depth),
            "strike" => num(opts, "strike", 35.0),
            "length" => num(opts, "length", 8000.0))
        xf = num(opts, "x", 0.5); yf = num(opts, "y", 0.5)
        local xc, yc
        if place != "" && !haskey(opts, "x") && !haskey(opts, "y")
            # default: source under the place's summit
            xc = 1 + (pi_info.summit[2] - pi_info.lon1) /
                 (pi_info.lon2 - pi_info.lon1) * (W - 1)
            yc = 1 + (pi_info.lat2 - pi_info.summit[1]) /
                 (pi_info.lat2 - pi_info.lat1) * (H - 1)
        else
            xc = 1 + xf * (W - 1); yc = 1 + yf * (H - 1)
        end
        cfg = (source = source, srcpars = srcpars, xc = xc, yc = yc,
               cm = num(opts, "cm", 14.0),
               heading = num(opts, "heading", 193.0),
               inc = num(opts, "inc", 35.0),
               lam_cm = num(opts, "lambda", 5.55),
               bperp = num(opts, "bperp", 0.0),
               demerr = num(opts, "demerr", 0.0),
               atm = num(opts, "atm", 0.35),
               coh = num(opts, "coh", 0.9),
               range = 800e3)
        println("  simulating interferogram: source=", source,
                " depth=", round(Int, srcpars["depth"]), " m",
                ", LOS max ", cfg.cm, " cm, lambda ", cfg.lam_cm, " cm,",
                " heading ", round(Int, cfg.heading), " deg, inc ",
                round(Int, cfg.inc), " deg")
        img, img_c, img_u, stats = render_insar(dem, px, seed, style, cfg)
        write_bmp(joinpath(outdir, "interferogram_" * prefix * ".bmp"), img)
        write_bmp(joinpath(outdir, "coherence_" * prefix * ".bmp"), img_c)
        write_bmp(joinpath(outdir, "unwrapped_" * prefix * ".bmp"), img_u)
        @printf("  deformation: %.2f cm max LOS -> %.1f fringes, mean coherence %.2f\n",
                stats.max_los_cm, stats.fringes, stats.mean_coh)
    end
    println("done.")
end

main()
