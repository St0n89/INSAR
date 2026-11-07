#!/usr/bin/env julia
# ============================================================================
#
#   insar_sat.jl -- satellite imagery & InSAR: simulations + REAL data
#   ---------------------------------------------------------
#   Pure Julia standard library (Printf, Random, Downloads). No packages.
#
#   * Renders natural-colour "satellite pictures" from synthetic fractal
#     terrain, or from REAL elevation data (Open-Meteo elevation API,
#     free, no key: Etna, Fuji, Matterhorn, Reykjanes, Grand Canyon...).
#   * Fetches REAL Sentinel-2 optical imagery (ESA Copernicus cloudless
#     mosaic via the open EOX tile service) with a hand-written baseline
#     JPEG decoder.
#   * Simulates SAR amplitude images (land-cover backscatter + Rayleigh
#     speckle with N-look averaging).
#   * Simulates SAR interferograms (InSAR): deformation sources (Mogi,
#     dyke, sill, strike-slip), topographic phase from a perpendicular
#     baseline, atmospheric delay screens, DEM-error fringes, and
#     coherence-dependent decorrelation noise. Wrapped phase is colored
#     with the classic cyclic InSAR rainbow (1 fringe = lambda/2 of LOS
#     displacement). Coherence map + unwrapped model field + SAR
#     amplitude are written alongside.
#   * Computes REAL interferograms: `real` downloads an actual Sentinel-1
#     interferogram from the open COMET LiCSAR archive (default: the
#     March 2021 Fagradalsfjall dike, Iceland), decodes the float GeoTIFF
#     products with a hand-written RFC-1951 inflater + GeoTIFF reader,
#     unwraps the phase with its own quality-guided unwrapper and renders
#     real fringes, coherence and LOS displacement in cm.
#   * All images are hand-rolled 24-bit BMP (no image package needed).
#
#   Usage:
#     julia insar_sat.jl                  # demo: synthetic volcano island
#     julia insar_sat.jl etna             # real DEM + summit uplift
#     julia insar_sat.jl real             # REAL interferogram (Fagradalsfjall)
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
#   a screw-dislocation approximation for strike-slip). The `real` command
#   uses genuinely measured radar phase, but raw per-scene products and
#   from-scratch SAR processing need the free Copernicus Data Space
#   Ecosystem account + ISCE/SNAP (out of scope for this stdlib tool).
#
# ============================================================================

using Printf
using Random
using Downloads
using LinearAlgebra   # mul!/transpose for the 8x8 IDCTs
using Dates           # S1 annotation timestamps (raw command)
using Mmap            # zero-RAM view of fetched zip members (3 GB hosts)

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
# 7c. REAL Sentinel-2 imagery: ESA Copernicus data served as the open
#     "Sentinel-2 cloudless" WMTS mosaic by EOX (no API key, free access).
#     Tiles are baseline JPEG (SOF0, 4:2:0, no restarts), so they are decoded
#     by a small baseline JPEG decoder written in pure Julia below -- no
#     external imaging package. Raw scene products & real SAR interferometry
#     need the free Copernicus Data Space Ecosystem + SNAP/ISCE (out of
#     scope for this stdlib tool).
# ----------------------------------------------------------------------------

# zigzag: natural (row-major, 1-based) index of the j-th coefficient in scan
const JZIG = UInt8[
     1, 2, 9,17,10, 3, 4,11,18,25,33,26,19,12, 5, 6,
    13,20,27,34,41,49,42,35,28,21,14, 7, 8,15,22,29,
    36,43,50,57,58,51,44,37,30,23,16,24,31,38,45,52,
    59,60,53,46,39,32,40,47,54,61,62,55,48,56,63,64]

# 8x8 IDCT matrix  IDCTM[x,u] = C(u)/2 * cos((2x+1)u*pi/16),  C(0)=1/sqrt(2)
const IDCTM = [(u == 0 ? 0.35355339059327373 : 0.5) *
               cos((2x + 1) * u * π / 16) for x in 0:7, u in 0:7]

mutable struct JBitReader
    data::Vector{UInt8}
    pos::Int          # index of the byte currently being read
    nbit::Int         # next bit within that byte (1 = MSB); 9 = byte exhausted
    skip00::Bool      # current byte is an FF followed by stuffed 00
end

@inline function jbit!(br::JBitReader)::Int
    if br.nbit > 8
        br.pos += 1
        if br.skip00
            br.pos += 1                # skip the stuffed 00 after FF
            br.skip00 = false
        end
        if br.pos > length(br.data)
            return 0                  # zero-pad truncated streams
        end
        if br.data[br.pos] == 0xFF
            if br.pos < length(br.data) && br.data[br.pos+1] == 0x00
                br.skip00 = true      # FF is data, the 00 is stuffing
            else
                error("unexpected JPEG marker inside entropy data")
            end
        end
        br.nbit = 1
    end
    v = (br.data[br.pos] >> (8 - br.nbit)) & 0x01
    br.nbit += 1
    return v
end

jrecv!(br::JBitReader, n::Int) = (v = 0; for _ in 1:n; v = (v << 1) | jbit!(br); end; v)

jextend(v::Int, n::Int) = v < (1 << (n - 1)) ? v - (1 << n) + 1 : v

struct JHuff
    minc::NTuple{16,Int}
    maxc::NTuple{16,Int}
    valp::NTuple{16,Int}
    vals::Vector{UInt8}
end

function jbuild_huff(counts, vals)::JHuff
    minc = fill(-1, 16); maxc = fill(-1, 16); valp = fill(0, 16)
    code = 0; k = 1
    for len in 1:16
        c = Int(counts[len])
        if c > 0
            minc[len] = code; valp[len] = k; maxc[len] = code + c - 1
            k += c
        end
        code = (code + c) << 1    # shift EVERY length, also for c == 0
    end
    return JHuff((minc...,), (maxc...,), (valp...,), vals)
end

function jhuff!(ht::JHuff, br::JBitReader)::Int
    code = 0
    for len in 1:16
        code = (code << 1) | jbit!(br)
        mc = ht.maxc[len]
        if mc >= 0 && code <= mc
            return Int(ht.vals[ht.valp[len] + code - ht.minc[len]])
        end
    end
    return error("corrupt JPEG: invalid Huffman code")
end

# decode one 8x8 block into blk (natural order, dequantized, no IDCT)
function jdecode_block!(blk::Matrix{Float64}, br::JBitReader,
                        htd::JHuff, hta::JHuff, qt::Matrix{Float64},
                        c::Int, dcp::Vector{Int})
    fill!(blk, 0.0)
    t = jhuff!(htd, br)
    dcp[c] += (t == 0 ? 0 : jextend(jrecv!(br, t), t))
    blk[1,1] = dcp[c] * qt[1,1]
    k = 2
    while k <= 64
        rs = jhuff!(hta, br)
        s = rs & 0x0F
        if s == 0
            if rs == 0xF0            # ZRL: skip 16 zeros
                k += 16
            else                     # EOB
                return
            end
        else
            k += rs >> 4             # skip r zeros
            if k <= 64
                nat = Int(JZIG[k])
                u = (nat - 1) ÷ 8 + 1
                v = (nat - 1) % 8 + 1
                blk[u,v] = jextend(jrecv!(br, s), s) * qt[u,v]
            end
            k += 1
        end
    end
    return
end

"""Baseline JPEG (SOF0/SOF1, interleaved scan) -> HxWx3 UInt8 RGB array."""
function decode_jpeg(data::Vector{UInt8})::Array{UInt8,3}
    length(data) >= 4 || error("not a JPEG")
    (data[1] == 0xFF && data[2] == 0xD8) || error("not a JPEG (no SOI)")
    qt = Dict{Int,Matrix{Float64}}()
    htd = Dict{Int,JHuff}(); hta = Dict{Int,JHuff}()
    cid = Int[]; hs = Int[]; vs = Int[]; tqs = Int[]
    tds = Int[]; tas = Int[]
    W = 0; H = 0; dri = 0; sos_end = 0
    pos = 3
    while pos < length(data) - 1
        if data[pos] != 0xFF
            pos += 1
            continue
        end
        m = Int(data[pos + 1])
        if m == 0xFF
            pos += 1
            continue
        end
        if m == 0xD8 || (0xD0 <= m <= 0xD7)   # stray SOI/RST before SOS
            pos += 2
            continue
        end
        m == 0xD9 && error("EOI before SOS")
        seglen = (Int(data[pos+2]) << 8) | Int(data[pos+3])
        seglen >= 2 || error("bad JPEG segment length")
        if m == 0xC0 || m == 0xC1             # SOF: baseline/extended
            H = (Int(data[pos+5]) << 8) | Int(data[pos+6])
            W = (Int(data[pos+7]) << 8) | Int(data[pos+8])
            ncomp = Int(data[pos+9])
            ncomp in (1, 3) || error("unsupported JPEG component count ", ncomp)
            empty!(cid); empty!(hs); empty!(vs); empty!(tqs)
            for k in 0:ncomp-1
                push!(cid, Int(data[pos+10+3k]))
                push!(hs, Int(data[pos+11+3k]) >> 4)
                push!(vs, Int(data[pos+11+3k]) & 0x0F)
                push!(tqs, Int(data[pos+12+3k]))
            end
        elseif m in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB,
                     0xCD, 0xCE, 0xCF)
            error("unsupported JPEG process (progressive/arithmetic)")
        elseif m == 0xC4                      # DHT
            p = pos + 4
            pend = pos + 2 + seglen
            while p < pend
                tclass = Int(data[p]) >> 4
                th = Int(data[p]) & 0x0F
                p += 1
                counts = data[p:p+15]
                p += 16
                n = Int(sum(counts))
                n > 0 || continue
                vals = data[p:p+n-1]
                p += n
                h = jbuild_huff(counts, vals)
                if tclass == 0; htd[th] = h; else; hta[th] = h; end
            end
        elseif m == 0xDB                      # DQT
            p = pos + 4
            pend = pos + 2 + seglen
            while p < pend
                Int(data[p]) >> 4 == 0 || error("16-bit quant table unsupported")
                tq = Int(data[p]) & 0x0F
                p += 1
                t = Matrix{Float64}(undef, 8, 8)
                for j in 1:64
                    nat = Int(JZIG[j])
                    t[(nat-1)÷8+1, (nat-1)%8+1] = Float64(data[p+j-1])
                end
                p += 64
                qt[tq] = t
            end
        elseif m == 0xDD                      # DRI
            dri = (Int(data[pos+4]) << 8) | Int(data[pos+5])
        elseif m == 0xDA                      # SOS
            ns = Int(data[pos+4])
            ns == length(cid) ||
                error("only single interleaved JPEG scans supported")
            resize!(tds, ns); resize!(tas, ns)
            for k in 0:ns-1
                cs = Int(data[pos+5+2k])
                idx = findfirst(==(cs), cid)
                idx === nothing && error("scan comp not in SOF")
                tds[idx] = Int(data[pos+6+2k]) >> 4
                tas[idx] = Int(data[pos+6+2k]) & 0x0F
            end
            sos_end = pos + 1 + seglen        # last SOS-header byte
            break
        end
        pos += 2 + seglen
    end
    (W > 0 && H > 0 && !isempty(cid)) || error("no SOF found in JPEG")

    ncomp = length(cid)
    hmax = maximum(hs); vmax = maximum(vs)
    mcux = cld(W, 8hmax); mcuy = cld(H, 8vmax)
    planes = [Matrix{Float64}(undef, mcuy * vs[c] * 8, mcux * hs[c] * 8)
              for c in 1:ncomp]
    dcp = zeros(Int, ncomp)
    br = JBitReader(data, sos_end, 9, false)
    blk = Matrix{Float64}(undef, 8, 8)
    tmp = Matrix{Float64}(undef, 8, 8)
    dct = Matrix{Float64}(undef, 8, 8)
    IDCTT = transpose(IDCTM)
    mcun = 0
    for my in 1:mcuy, mx in 1:mcux
        for c in 1:ncomp
            qtc = qt[tqs[c]]
            hd = htd[tds[c]]
            ha = hta[tas[c]]
            for j in 1:vs[c], i in 1:hs[c]
                jdecode_block!(blk, br, hd, ha, qtc, c, dcp)
                mul!(tmp, IDCTM, blk)     # rows: x (height)
                mul!(dct, tmp, IDCTT)     # cols: y (width)
                y0 = (my - 1) * vs[c] * 8 + (j - 1) * 8
                x0 = (mx - 1) * hs[c] * 8 + (i - 1) * 8
                for yy in 1:8, xx in 1:8
                    planes[c][y0+yy, x0+xx] = dct[yy,xx] + 128.0
                end
            end
        end
        mcun += 1
        if dri > 0 && mcun % dri == 0 && (my != mcuy || mx != mcux)
            br.nbit = 9                  # align to byte boundary
            if br.skip00
                br.pos += 1
                br.skip00 = false
            end
            if br.pos + 2 <= length(data) && data[br.pos+1] == 0xFF &&
               0xD0 <= Int(data[br.pos+2]) <= 0xD7
                br.pos += 2               # consume RSTn
            end
            fill!(dcp, 0)
        end
    end

    img = Array{UInt8}(undef, H, W, 3)
    pY = planes[1]
    ch = size(pY, 1); cw = size(pY, 2)
    pCb = ncomp >= 3 ? planes[2] : pY
    pCr = ncomp >= 3 ? planes[3] : pY
    chb = size(pCb, 1); cwb = size(pCb, 2)
    for j in 1:H
        sy  = min(ch,  1 + (j - 1) * ch  ÷ H)
        syb = min(chb, 1 + (j - 1) * chb ÷ H)
        for i in 1:W
            sx  = min(cw,  1 + (i - 1) * cw  ÷ W)
            sxb = min(cwb, 1 + (i - 1) * cwb ÷ W)
            Y  = pY[sy, sx]
            if ncomp == 1
                v = to8(Y)
                img[j,i,1] = v; img[j,i,2] = v; img[j,i,3] = v
            else
                Cb = pCb[syb, sxb]; Cr = pCr[syb, sxb]
                img[j,i,1] = to8(Y + 1.402 * (Cr - 128.0))
                img[j,i,2] = to8(Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0))
                img[j,i,3] = to8(Y + 1.772 * (Cb - 128.0))
            end
        end
    end
    return img
end

# fetch + stitch an Nx x Ny grid of s2cloudless WMTS tiles centred on lat/lon
function s2_fetch_tiles(lat::Float64, lon::Float64, z::Int, nx::Int, ny::Int,
                        year::Int, outdir::String)
    n = 2^z
    tx = floor(Int, (lon + 180) / 360 * n)
    ty = floor(Int, (1 - asinh(tan(deg2rad(lat))) / π) / 2 * n)
    tx0 = clamp(tx - (nx - 1) ÷ 2, 0, n - nx)
    ty0 = clamp(ty - (ny - 1) ÷ 2, 0, n - ny)
    img = fill(UInt8(28), 256 * ny, 256 * nx, 3)
    misses = 0
    for jy in 0:ny-1, ix in 0:nx-1
        x = tx0 + ix; y = ty0 + jy
        url = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-" * string(year) *
              "_3857/default/GoogleMapsCompatible/" * string(z) * "/" *
              string(y) * "/" * string(x) * ".jpg"
        try
            tmp = joinpath(outdir, ".tmp_tile.jpg")
            Downloads.download(url, tmp)
            d = read(tmp)
            rm(tmp; force = true)
            tile = decode_jpeg(d)
            Ht, Wt = size(tile, 1), size(tile, 2)
            img[jy*256+1:jy*256+Ht, ix*256+1:ix*256+Wt, :] .= tile
        catch err
            misses += 1
            println("  tile ", x, ",", y, " unavailable (",
                    first(sprint(showerror, err), 60), ")")
        end
        @printf("  tiles %d/%d\r", jy * nx + ix + 1, nx * ny)
        flush(stdout)
        (jy * nx + ix < nx * ny - 1) && sleep(0.12)
    end
    println("  tiles ", nx * ny, "/", nx * ny, " loaded (", misses, " missing)")
    return img, tx0, ty0
end

# global web-mercator pixel of lat/lon -> pixel inside the stitched tile grid
function s2_pixel_of(lat::Float64, lon::Float64, z::Int, tx0::Int, ty0::Int)
    n = 2^z
    wx = (lon + 180) / 360 * n * 256
    wy = (1 - asinh(tan(deg2rad(lat))) / π) / 2 * n * 256
    return 1 + wx - tx0 * 256, 1 + wy - ty0 * 256
end

# ----------------------------------------------------------------------------
# 7d. INFLATE (RFC 1951) -- the real-data GeoTIFFs in 7f are Deflate/zlib
#     compressed and Julia's stdlib ships no zlib, so here is a small,
#     bit-exact inflater written against RFC 1951. NOTE: DEFLATE reads
#     bits LSB-first -- this is NOT the JPEG bit reader (MSB-first).
#     Supported: stored, fixed and dynamic blocks; a 9-bit Huffman lookup
#     table with a canonical bit-by-bit fallback for longer codes.
# ----------------------------------------------------------------------------

mutable struct ZBits
    data::Vector{UInt8}
    pos::Int                   # next byte to load (1-based)
    buf::UInt64                # LSB-first bit buffer; bits above nbuf are 0
    nbuf::Int                  # valid bits in buf
    n::Int
end

ZBits(d::Vector{UInt8}) = ZBits(d, 1, UInt64(0), 0, length(d))

function zfill!(zb::ZBits)
    @inbounds while zb.nbuf <= 48 && zb.pos <= zb.n
        zb.buf |= UInt64(zb.data[zb.pos]) << zb.nbuf
        zb.nbuf += 8
        zb.pos += 1
    end
    return zb
end

function zget!(zb::ZBits, k::Int)
    if zb.nbuf < k
        zfill!(zb)
        zb.nbuf < k && (zb.nbuf = k)   # zero-pad past EOF (corrupt stream)
    end
    v = UInt32(zb.buf & ((UInt32(1) << k) - UInt32(1)))
    zb.buf >>= k
    zb.nbuf -= k
    return v
end

struct ZHuff
    lut::Vector{UInt32}        # 512 entries: (codelen << 16) | symbol; 0 = escape
    counts::Vector{Int32}      # codes per length 1..15 (index l+1)
    syms::Vector{Int32}        # symbols sorted by (length, symbol value)
    first::Vector{Int32}       # canonical first code per length (index l+1)
end

function zbuild_huff(lengths::Vector{UInt8})::ZHuff
    counts = zeros(Int32, 16)
    for l in lengths
        l > 0 && (counts[l+1] += Int32(1))
    end
    offs = zeros(Int32, 16)                 # offs[l+1]: 0-based start for length l
    for l in 1:15
        offs[l+1] = offs[l] + counts[l]
    end
    ns = Int(sum(counts))
    syms = zeros(Int32, max(ns, 1))
    run = copy(offs)
    for (s0, l) in enumerate(lengths)
        if l > 0
            syms[run[l+1] + 1] = Int32(s0 - 1)
            run[l+1] += Int32(1)
        end
    end
    # canonical first code per length -- shift EVERY length, incl. zero counts
    first = zeros(Int32, 16)
    for l in 1:15
        first[l+1] = (first[l] + counts[l]) << 1
    end
    # 9-bit lookup table: index = the next 9 stream bits, i.e. the canonical
    # code bit-reversed, with all higher-bit combinations filled in
    lut = zeros(UInt32, 512)
    for l in 1:15
        for k in 0:(counts[l+1] - 1)
            s = syms[offs[l+1] + k + 1]
            code = first[l+1] + k
            if l <= 9
                r = UInt32(0)
                for b in 0:l-1
                    ((code >> b) & 1) == 1 && (r |= UInt32(1) << (l - 1 - b))
                end
                step = UInt32(1) << l
                for j in r:step:UInt32(511)
                    lut[j+1] = (UInt32(l) << 16) | UInt32(s)
                end
            end
        end
    end
    return ZHuff(lut, counts, syms, first)
end

function zdec!(zb::ZBits, h::ZHuff)::Int
    if zb.nbuf < 9
        zfill!(zb)
    end
    e = h.lut[(zb.buf & 0x1FF) + 1]
    l = Int(e >> 16)
    if l > 0 && l <= zb.nbuf
        zb.buf >>= l
        zb.nbuf -= l
        return Int(e & 0xFFFF)
    end
    # escape: code longer than 9 bits -> canonical bit-by-bit decode
    code = 0; fc = 0; idx = 0
    for l in 1:15
        code |= Int(zget!(zb, 1))
        cnt = Int(h.counts[l+1])
        if code - cnt < fc
            return Int(h.syms[idx + (code - fc) + 1])
        end
        idx += cnt
        fc = (fc + cnt) << 1
        code <<= 1
    end
    error("inflate: invalid Huffman code")
end

function zfixed_huffs()
    ll = UInt8[]
    for s in 0:287
        if s < 144; push!(ll, 0x08)
        elseif s < 256; push!(ll, 0x09)
        elseif s < 280; push!(ll, 0x07)
        else; push!(ll, 0x08); end
    end
    return zbuild_huff(ll), zbuild_huff(fill(0x05, 32))
end

const ZCLORD = UInt8[16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
const ZLEN_BASE = UInt16[3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                         35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258]
const ZLEN_EXTRA = UInt8[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                         3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0]
const ZDIST_BASE = UInt16[1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                          257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                          8193, 12289, 16385, 24577]
const ZDIST_EXTRA = UInt8[0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                          7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]

function zread_dynamic!(zb::ZBits)
    hlit = Int(zget!(zb, 5)) + 257
    hdist = Int(zget!(zb, 5)) + 1
    hclen = Int(zget!(zb, 4)) + 4
    cl = zeros(UInt8, 19)
    for i in 1:hclen
        cl[ZCLORD[i] + 1] = UInt8(zget!(zb, 3))
    end
    clh = zbuild_huff(cl)
    total = hlit + hdist
    lengths = zeros(UInt8, total)
    i = 1
    while i <= total
        s = zdec!(zb, clh)
        if s < 16
            lengths[i] = UInt8(s); i += 1
        elseif s == 16
            i == 1 && error("inflate: length repeat with no previous")
            n = 3 + Int(zget!(zb, 2))
            i - 1 + n > total && error("inflate: length repeat overflow")
            for _ in 1:n; lengths[i] = lengths[i-1]; i += 1; end
        else
            n = s == 17 ? 3 + Int(zget!(zb, 3)) : 11 + Int(zget!(zb, 7))
            i - 1 + n > total && error("inflate: zero-run overflow")
            i += n
        end
    end
    return zbuild_huff(lengths[1:hlit]), zbuild_huff(lengths[hlit+1:total])
end

function zcopy!(out::Vector{UInt8}, d::Int, n::Int)
    s = length(out) - d              # 0-based source start
    old = length(out)
    resize!(out, old + n)
    if d >= n
        copyto!(out, old + 1, out, s + 1, n)
    else
        for k in 0:n-1
            @inbounds out[old + k + 1] = out[s + 1 + (k % d)]
        end
    end
    return out
end

function inflate_zlib(data::Vector{UInt8})::Vector{UInt8}
    length(data) >= 2 || error("inflate: stream too short")
    (data[1] & 0x0F) == 8 || error("inflate: not deflate (CM != 8)")
    ((UInt32(data[1]) << 8) | UInt32(data[2])) % 31 == 0 ||
        error("inflate: bad zlib header")
    zb = ZBits(data, 3, UInt64(0), 0, length(data))
    out = UInt8[]
    zinflate_loop!(out, zb)
    return out
end

function zinflate_loop!(out::Vector{UInt8}, zb::ZBits)
    while true
        fin = zget!(zb, 1) == 1
        btype = Int(zget!(zb, 2))
        if btype == 0
            # stored block: byte-align, LEN/NLEN, raw bytes
            r = zb.nbuf % 8
            zb.buf >>= r; zb.nbuf -= r
            len = Int(zget!(zb, 16))
            nlen = Int(zget!(zb, 16))
            (len ⊻ nlen) & 0xFFFF == 0xFFFF ||
                error("inflate: stored-block LEN check failed")
            for _ in 1:len
                push!(out, UInt8(zget!(zb, 8)))
            end
        elseif btype == 1 || btype == 2
            lh, dh = btype == 1 ? zfixed_huffs() : zread_dynamic!(zb)
            while true
                sym = zdec!(zb, lh)
                if sym == 256
                    break
                elseif sym < 256
                    push!(out, UInt8(sym))
                else
                    sym > 285 && error("inflate: bad length symbol")
                    li = sym - 256
                    n = Int(ZLEN_BASE[li]) + Int(zget!(zb, Int(ZLEN_EXTRA[li])))
                    dsym = zdec!(zb, dh)
                    dsym > 29 && error("inflate: bad distance symbol")
                    d = Int(ZDIST_BASE[dsym+1]) +
                        Int(zget!(zb, Int(ZDIST_EXTRA[dsym+1])))
                    (d <= 0 || d > length(out)) &&
                        error("inflate: distance out of range")
                    zcopy!(out, d, n)
                end
            end
        else
            error("inflate: reserved block type")
        end
        fin && break
    end
    return out
end

# ----------------------------------------------------------------------------
# 7e. GeoTIFF reader -- exactly the feature set of the LiCSAR products:
#     little-endian, strip OR tile layout, Deflate compression, uint8 or
#     float32 samples, TIFF predictors 1 (none), 2 (horizontal diff) and
#     3 (floating point). Predictor semantics follow libtiff exactly
#     (tif_predict.c): the predictor restarts on every row of a strip or
#     tile; for predictor 3 each row's bytes are chained (b[i] += b[i-1],
#     stride = samples per pixel -- fpAcc), then split into 4 byte planes
#     and re-interleaved so plane 3 holds each sample's low byte.
#     Geo tags give the lon/lat of pixel (1,1) and the pixel steps.
# ----------------------------------------------------------------------------

struct TiffGeo
    lon0::Float64; dlon::Float64   # lon of column 1 (pixel centre), per column
    lat0::Float64; dlat::Float64   # lat of row 1; dlat < 0 (north-up image)
end

struct TiffHead
    W::Int; H::Int
    bps::Int; spp::Int
    compression::Int; predictor::Int
    tiled::Bool
    offsets::Vector{Int}; counts::Vector{Int}
    rowsperstrip::Int
    tw::Int; tl::Int
    geo::Union{TiffGeo,Nothing}
    data::Vector{UInt8}
end

function tiff_u16(d::Vector{UInt8}, p::Int)
    return UInt16(d[p]) | (UInt16(d[p+1]) << 8)
end
function tiff_u32(d::Vector{UInt8}, p::Int)
    return UInt32(d[p]) | (UInt32(d[p+1]) << 8) |
           (UInt32(d[p+2]) << 16) | (UInt32(d[p+3]) << 24)
end
function tiff_f64(raw::Vector{UInt8}, i::Int)
    bits = UInt64(0)
    for k in 0:7
        bits |= UInt64(raw[i+k]) << (8k)
    end
    return reinterpret(Float64, bits)
end

const TIFF_TYPESZ = Dict(1 => 1, 2 => 1, 3 => 2, 4 => 4, 5 => 8, 11 => 4, 12 => 8)

function tiff_scalar(tags::Dict{Int,Vector{UInt8}}, tag::Int)
    raw = tags[tag]
    return length(raw) >= 4 ? Int(tiff_u32(raw, 1)) : Int(tiff_u16(raw, 1))
end

function tiff_longs(tags::Dict{Int,Vector{UInt8}}, tag::Int)
    raw = tags[tag]
    m = length(raw) ÷ 4
    out = Vector{Int}(undef, m)
    for i in 1:m
        out[i] = Int(tiff_u32(raw, 1 + 4(i-1)))
    end
    return out
end

function tiff_head(path::String)::TiffHead
    d = read(path)
    length(d) >= 8 || error("tiff: file too short: ", path)
    (d[1] == 0x49 && d[2] == 0x49) ||
        error("tiff: only little-endian (II) files supported")
    ifd = Int(tiff_u32(d, 5))
    nent = Int(tiff_u16(d, ifd + 1))
    tags = Dict{Int,Vector{UInt8}}()
    for i in 0:nent-1
        p = ifd + 3 + 12i
        (p + 12 <= length(d)) || error("tiff: IFD out of range")
        tag = Int(tiff_u16(d, p))
        typ = Int(tiff_u16(d, p + 2))
        cnt = Int(tiff_u32(d, p + 4))
        haskey(TIFF_TYPESZ, typ) ||
            error("tiff: unsupported field type ", typ)
        sz = TIFF_TYPESZ[typ] * cnt
        if sz <= 4
            tags[tag] = d[p+8:p+7+sz]
        else
            off = Int(tiff_u32(d, p + 8))
            (off + sz <= length(d)) || error("tiff: field data out of range")
            tags[tag] = d[off+1:off+sz]
        end
    end
    haskey(tags, 256) || error("tiff: no ImageWidth")
    haskey(tags, 257) || error("tiff: no ImageLength")
    W = tiff_scalar(tags, 256); H = tiff_scalar(tags, 257)
    bps = haskey(tags, 258) ? tiff_scalar(tags, 258) : 1
    spp = haskey(tags, 277) ? tiff_scalar(tags, 277) : 1
    compression = haskey(tags, 259) ? tiff_scalar(tags, 259) : 1
    predictor = haskey(tags, 317) ? tiff_scalar(tags, 317) : 1
    compression in (1, 8) || error("tiff: unsupported compression ", compression,
        " (this reader handles none and Adobe Deflate/zlib)")
    tiled = haskey(tags, 322)
    local offsets::Vector{Int}, counts::Vector{Int}
    local tw::Int, tl::Int, rps::Int
    tw = 0; tl = 0; rps = H
    if tiled
        tw = tiff_scalar(tags, 322); tl = tiff_scalar(tags, 323)
        offsets = tiff_longs(tags, 324); counts = tiff_longs(tags, 325)
    else
        offsets = tiff_longs(tags, 273); counts = tiff_longs(tags, 279)
        rps = haskey(tags, 278) ? tiff_scalar(tags, 278) : H
    end
    geo = nothing
    if haskey(tags, 33550) && haskey(tags, 33922)
        sx = tiff_f64(tags[33550], 1); sy = tiff_f64(tags[33550], 9)
        ti = tiff_f64(tags[33922], 1); tj = tiff_f64(tags[33922], 9)
        gx = tiff_f64(tags[33922], 25); gy = tiff_f64(tags[33922], 33)
        geo = TiffGeo(gx - ti * sx, sx, gy + tj * sy, -sy)
    end
    return TiffHead(W, H, bps, spp, compression, predictor, tiled,
                    offsets, counts, rps, tw, tl, geo, d)
end

function tiff_raw(head::TiffHead, k::Int)::Vector{UInt8}
    off = head.offsets[k]; cnt = head.counts[k]
    (off + cnt <= length(head.data)) ||
        error("tiff: strip/tile ", k, " out of range")
    raw = head.data[off+1:off+cnt]
    head.compression == 8 && return inflate_zlib(raw)
    return raw
end

# apply the TIFF predictor to a decoded strip/tile buffer, per row (libtiff)
function tiff_predict!(b::Vector{UInt8}, head::TiffHead, rowbytes::Int)
    pred = head.predictor
    pred == 1 && return b
    nrows = length(b) ÷ rowbytes
    if pred == 2
        head.bps == 8 || error("tiff: predictor 2 supported for 8-bit only")
        stride = head.spp
        for r in 0:nrows-1
            p = r * rowbytes
            for j in 1:stride
                acc = UInt8(0)
                for i in (p+j):stride:(p+rowbytes)
                    acc += b[i]
                    b[i] = acc
                end
            end
        end
    elseif pred == 3
        (head.bps == 32 && head.spp == 1) ||
            error("tiff: predictor 3 supported for 32-bit single-band only")
        for r in 0:nrows-1
            p = r * rowbytes
            @inbounds for i in p+2:p+rowbytes
                b[i] += b[i-1]
            end
        end
    else
        error("tiff: unsupported predictor ", pred)
    end
    return b
end

# strip/tile bytes -> sample values (Float64), ncols samples per row
function tiff_samples(head::TiffHead, raw::Vector{UInt8}, ncols::Int)
    rowbytes = ncols * head.spp * head.bps ÷ 8
    length(raw) % rowbytes == 0 ||
        error("tiff: strip size not a multiple of the row size")
    nrows = length(raw) ÷ rowbytes
    tiff_predict!(raw, head, rowbytes)
    vals = Vector{Float64}(undef, nrows * ncols)
    if head.bps == 8
        for r in 0:nrows-1, i in 1:ncols
            vals[r*ncols + i] = Float64(raw[r*rowbytes + (i-1)*head.spp + 1])
        end
    elseif head.bps == 32
        wc = rowbytes ÷ 4
        for r in 0:nrows-1
            p = r * rowbytes
            @inbounds for i in 1:ncols
                o = i - 1
                if head.predictor == 3
                    # 4 byte planes; sample i low byte comes from plane 3
                    b0 = raw[p + 3*wc + o + 1]
                    b1 = raw[p + 2*wc + o + 1]
                    b2 = raw[p + 1*wc + o + 1]
                    b3 = raw[p + 0*wc + o + 1]
                    bits = UInt32(b0) | (UInt32(b1) << 8) |
                           (UInt32(b2) << 16) | (UInt32(b3) << 24)
                else
                    q = p + 4o
                    bits = UInt32(raw[q+1]) | (UInt32(raw[q+2]) << 8) |
                           (UInt32(raw[q+3]) << 16) | (UInt32(raw[q+4]) << 24)
                end
                vals[r*ncols + i] = Float64(reinterpret(Float32, bits))
            end
        end
    else
        error("tiff: unsupported bits per sample ", head.bps)
    end
    return vals
end

# read a (crop of a) single-band GeoTIFF as Float64; y0:y1 / x0:x1 are
# 1-based inclusive ranges of the FULL image; defaults = everything
function tiff_read(path::String, y0::Int = 1, y1::Int = 0,
                   x0::Int = 1, x1::Int = 0)::Matrix{Float64}
    head = tiff_head(path)
    W, H = head.W, head.H
    (y1 < y0 || y0 < 1) && (y0 = 1)
    (x1 < x0 || x0 < 1) && (x0 = 1)
    y1 <= y0 && (y1 = H)
    x1 <= x0 && (x1 = W)
    y1 = min(y1, H); x1 = min(x1, W)
    out = Matrix{Float64}(undef, y1 - y0 + 1, x1 - x0 + 1)
    if head.tiled
        tw, tl = head.tw, head.tl
        across = cld(W, tw)
        for ty in fld(y0 - 1, tl):fld(y1 - 1, tl)
            for tx in fld(x0 - 1, tw):fld(x1 - 1, tw)
                k = ty * across + tx + 1
                vals = tiff_samples(head, tiff_raw(head, k), tw)
                ry0 = max(y0, ty * tl + 1)
                ry1 = min(y1, (ty + 1) * tl, H)
                sc0 = max(x0 - tx * tw, 1)
                sc1 = min(x1 - tx * tw, tw, W - tx * tw)
                for r in ry0:ry1
                    tr = r - ty * tl - 1         # 0-based row inside tile
                    for x in sc0:sc1
                        out[r-y0+1, tx*tw + x - x0 + 1] = vals[tr*tw + x]
                    end
                end
            end
        end
    else
        rps = head.rowsperstrip
        for s in fld(y0 - 1, rps) + 1:fld(y1 - 1, rps) + 1
            vals = tiff_samples(head, tiff_raw(head, s), W)
            nr = min(rps, H - (s - 1) * rps)      # rows in this strip
            for r in max(y0, (s-1)*rps + 1):min(y1, (s-1)*rps + nr)
                tr = r - (s - 1) * rps - 1        # 0-based row inside strip
                for x in x0:x1
                    out[r-y0+1, x-x0+1] = vals[tr*W + x]
                end
            end
        end
    end
    return out
end

# ----------------------------------------------------------------------------
# 7f. REAL interferograms from the COMET LiCSAR open archive (CEDA).
#     LiCSAR processes real Sentinel-1 SLC scenes into 0.001-deg (~90 m
#     at these latitudes) geocoded interferogram products: wrapped phase,
#     coherence and an unwrapped reference, all GeoTIFF -- decoded by
#     the pure-Julia inflater + GeoTIFF reader above. Free, no key:
#     https://comet.nerc.ac.uk/COMET-LiCS-portal/
#     Raw per-scene products & from-scratch SAR processing need the free
#     Copernicus Data Space Ecosystem account + ISCE/SNAP (out of scope).
# ----------------------------------------------------------------------------

struct LiCSScene
    frameset::String
    trackdir::String
    lon1::Float64; lon2::Float64; lat1::Float64; lat2::Float64
    pair::String
    site::Tuple{Float64,Float64}          # lat, lon of the deformation centre
    place::String
    desc::String
end

const LICSAR_SCENES = [
    LiCSScene("016A_02504_162118", "16", -25.18, -18.32, 63.19, 66.78,
        "20210315_20210321", (63.880, -22.272), "reykjanes",
        "Fagradalsfjall dyke intrusion & Geldingadalur eruption, Reykjanes, Iceland (Mar 2021)"),
]

function lics_trackdir(frameset::String)
    t = split(frameset, '_')[1]
    t = replace(t, r"^0+" => "")
    return replace(t, r"[A-Z]$" => "")
end

function lics_url(sc::LiCSScene, pair::String, name::String)
    return "https://dap.ceda.ac.uk/neodc/comet/data/licsar_products/" *
           sc.trackdir * "/" * sc.frameset * "/" * pair * "/" *
           pair * ".geo." * name
end

function lics_download(url::String, dest::String; attempts::Int = 4)
    isfile(dest) && return dest
    part = dest * ".part"
    for a in 1:attempts
        try
            Downloads.download(url, part)
            mv(part, dest; force = true)
            return dest
        catch e
            rm(part; force = true)
            a == attempts && error("download failed: ", url, " (",
                sprint(showerror, e)[1:min(end, 120)], ")")
            println("  download hiccup, retrying (", a, ")...")
            sleep(2.0 * a)
        end
    end
end

# ----------------------------------------------------------------------------
# 7g. RAW Sentinel-1 SLC processing (v4): from-scratch InSAR from level-1 SLCs
#     via the Copernicus Data Space. Annotation XML parser, stripped
#     complex-int16 TIFF reader, hand-written radix-2 FFT, validated TOPS
#     burst timing, and the deramp / shift / reramp azimuth machinery that
#     TOPS coregistration requires. Pure stdlib.  Subcommand: `raw`
#     (see usage(): raw info / raw selftest).
# ----------------------------------------------------------------------------

const S1_C = 2.99792458e8

# ---- S1 time & XML helpers ----------------------------------------------

"""Parse '2021-03-15T18:58:47.105685' -> Float64 unix seconds (us preserved)."""
function parse_s1_time(s::AbstractString)::Float64
    s = strip(s)
    endswith(s, "Z") && (s = s[1:end-1])    # tolerate OData '...Z' suffix
    m = match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$", s)
    m === nothing && error("bad S1 time: " * s)
    y, mo, d, h, mi, se = [parse(Int, m[i]) for i in 1:6]
    frac = m[7] === nothing ? 0.0 : parse(Float64, "0." * m[7])
    return Dates.datetime2unix(Dates.DateTime(y, mo, d, h, mi, se)) + frac
end

function s1_tag_val(xml::AbstractString, t::String)::String
    m = match(Regex("<$t(?:\\s[^>]*)?>\\s*([^<]*?)\\s*</$t>"), xml)
    m === nothing && error("tag not found: " * t)
    return m[1]
end
s1_tag_f(xml, t) = parse(Float64, s1_tag_val(xml, t))
s1_tag_i(xml, t) = parse(Int, s1_tag_val(xml, t))

function s1_section(xml::AbstractString, t::String)::String
    m = match(Regex("<$t[ >].*?</$t>", "s"), xml)
    m === nothing && error("section not found: " * t)
    return m.match
end
function s1_sections(xml::AbstractString, t::String)
    return [String(m.match) for m in eachmatch(Regex("<$t[ >].*?</$t>", "s"), xml)]
end

# ---- annotation ---------------------------------------------------------

mutable struct S1Orbit
    t::Vector{Float64}
    pos::Matrix{Float64}      # 3 x n ECEF [m]
    vel::Matrix{Float64}      # 3 x n ECEF [m/s]
end

mutable struct S1Burst
    t_az::Float64             # burst azimuth anchor time [unix s]
    first_valid::Vector{Int}  # per local line, -1 = invalid
    last_valid::Vector{Int}
end

mutable struct S1GCP
    line::Int; pixel::Int
    lat::Float64; lon::Float64; hgt::Float64
    t_az::Float64; t_sl::Float64
end

mutable struct S1Annotation
    mission::String; swath::String; pol::String
    t_start::Float64; t_stop::Float64
    orbit_abs::Int; pass_dir::String
    prf::Float64                # radar PRF [Hz] (= TOPS steering Doppler rate)
    nsamples::Int; nlines::Int
    azimuth_dt::Float64        # per-line azimuth step WITHIN a burst [s]
    range_sr::Float64          # range sampling rate [Hz]
    radar_freq::Float64        # [Hz]
    slant_t0::Float64          # 2-way slant-range time of pixel 0 [s]
    lines_per_burst::Int; samples_per_burst::Int
    burst_cycle::Float64       # burst-to-burst anchor spacing [s]
    bursts::Vector{S1Burst}
    orbit::S1Orbit
    gcps::Vector{S1GCP}
    fmr::Vector{Tuple{Float64,Float64,Vector{Float64}}}  # azimuth FM rate polys
    dce::Vector{Tuple{Float64,Float64,Vector{Float64}}}  # data DC polys
end

function s1_polylist(xml::AbstractString, block::String, item::String, poly::String)
    out = Vector{Tuple{Float64,Float64,Vector{Float64}}}()
    blk = match(Regex("<$block[^>]*>.*?</$block>", "s"), xml)
    blk === nothing && return out
    for m in eachmatch(Regex("<$item>(.*?)</$item>", "s"), blk.match)
        s = String(m.captures[1])
        t = parse_s1_time(s1_tag_val(s, "azimuthTime"))
        t0 = parse(Float64, match(r"<t0>\s*([^<]+)\s*</t0>", s).captures[1])
        p = [parse(Float64, x) for x in split(strip(s1_tag_val(s, poly)))]
        push!(out, (t, t0, p))
    end
    return out
end

function parse_s1_annotation(path::String)::S1Annotation
    xml = read(path, String)
    ads = s1_section(xml, "adsHeader")
    img = s1_section(xml, "imageInformation")
    gen = s1_section(xml, "generalAnnotation")
    ann = S1Annotation(
        s1_tag_val(ads, "missionId"), s1_tag_val(ads, "swath"),
        s1_tag_val(ads, "polarisation"),
        parse_s1_time(s1_tag_val(ads, "startTime")),
        parse_s1_time(s1_tag_val(ads, "stopTime")),
        s1_tag_i(ads, "absoluteOrbitNumber"), s1_tag_val(gen, "pass"),
        s1_tag_f(s1_section(xml, "downlinkInformation"), "prf"),
        s1_tag_i(img, "numberOfSamples"), s1_tag_i(img, "numberOfLines"),
        s1_tag_f(img, "azimuthTimeInterval"),
        s1_tag_f(gen, "rangeSamplingRate"), s1_tag_f(gen, "radarFrequency"),
        s1_tag_f(img, "slantRangeTime"),
        0, 0, 0.0, S1Burst[], S1Orbit(Float64[], zeros(3, 0), zeros(3, 0)),
        S1GCP[], Tuple{Float64,Float64,Vector{Float64}}[],
        Tuple{Float64,Float64,Vector{Float64}}[])
    st = s1_section(xml, "swathTiming")
    ann.lines_per_burst = s1_tag_i(st, "linesPerBurst")
    ann.samples_per_burst = s1_tag_i(st, "samplesPerBurst")
    for b in s1_sections(st, "burst")
        fv = [parse(Int, x) for x in split(strip(s1_tag_val(b, "firstValidSample")))]
        lv = [parse(Int, x) for x in split(strip(s1_tag_val(b, "lastValidSample")))]
        push!(ann.bursts, S1Burst(parse_s1_time(s1_tag_val(b, "azimuthTime")), fv, lv))
    end
    length(ann.bursts) >= 2 && (ann.burst_cycle = ann.bursts[2].t_az - ann.bursts[1].t_az)
    for o in s1_sections(s1_section(xml, "orbitList"), "orbit")
        push!(ann.orbit.t, parse_s1_time(s1_tag_val(o, "time")))
        p = [s1_tag_f(o, "x"), s1_tag_f(o, "y"), s1_tag_f(o, "z")]
        m = match(Regex("<velocity[^>]*>\\s*<x[^>]*>\\s*(-?[0-9.eE+]+)\\s*</x>\\s*" *
                         "<y[^>]*>\\s*(-?[0-9.eE+]+)\\s*</y>\\s*" *
                         "<z[^>]*>\\s*(-?[0-9.eE+]+)\\s*</z>", "s"), o)
        m === nothing && error("no velocity in orbit vector")
        v = [parse(Float64, m[i]) for i in 1:3]
        ann.orbit.pos = hcat(ann.orbit.pos, p)
        ann.orbit.vel = hcat(ann.orbit.vel, v)
    end
    for g in s1_sections(s1_section(xml, "geolocationGrid"), "geolocationGridPoint")
        push!(ann.gcps, S1GCP(s1_tag_i(g, "line"), s1_tag_i(g, "pixel"),
            s1_tag_f(g, "latitude"), s1_tag_f(g, "longitude"), s1_tag_f(g, "height"),
            parse_s1_time(s1_tag_val(g, "azimuthTime")),
            s1_tag_f(g, "slantRangeTime")))
    end
    ann.fmr = s1_polylist(xml, "azimuthFmRateList", "azimuthFmRate",
                          "azimuthFmRatePolynomial")
    ann.dce = s1_polylist(xml, "dcEstimateList", "dcEstimate", "dataDcPolynomial")
    return ann
end

# ---- validated TOPS burst timing model ---------------------------------
# Line l (0-based, global) of burst k = l ÷ linesPerBurst sits at azimuth time
#   t(l) = bursts[k+1].t_az + (l - k*linesPerBurst) * azimuth_dt
# (checked against the 416D GCP azimuth times to <1 ms; bursts overlap in
# zero-Doppler time because lpb*azimuth_dt > burst_cycle.)

s1_burst_index(ann::S1Annotation, l::Int) = l ÷ ann.lines_per_burst

function s1_line_az_time(ann::S1Annotation, l::Int)::Float64
    k = s1_burst_index(ann, l)
    return ann.bursts[k + 1].t_az +                # true per-burst anchor
           (l - k * ann.lines_per_burst) * ann.azimuth_dt
end

s1_slant_time(ann::S1Annotation, pixel::Int) = ann.slant_t0 + pixel / ann.range_sr

"""Evaluate a (t, t0, coeffs) polynomial list at azimuth time t, pixel p."""
function s1_poly_eval(list, t::Float64, pixel::Int, ann::S1Annotation)::Float64
    isempty(list) && return 0.0
    i = 1
    while i < length(list) && list[i+1][1] < t
        i += 1
    end
    lo = list[i]
    hi = list[min(i + 1, length(list))]
    w = clamp((t - lo[1]) / (hi[1] - lo[1]), 0.0, 1.0)
    co = lo[3] .+ w .* (hi[3] .- lo[3])
    ds = s1_slant_time(ann, pixel) - lo[2]
    return co[1] + co[2] * ds + co[3] * ds * ds
end

s1_ka(ann::S1Annotation, t::Float64, pixel::Int) =
    s1_poly_eval(ann.fmr, t, pixel, ann)
s1_dc(ann::S1Annotation, t::Float64, pixel::Int) =
    s1_poly_eval(ann.dce, t, pixel, ann)

# ---- SLC measurement TIFF reader ---------------------------------------
# Classic little-endian TIFF, 1 row/strip, complex int16 (I/Q interleaved).
# S1 files show small inconsistent offsets in the IFD/arrays -> locate by
# content validation (bytecounts == 4*width, strip offsets evenly spaced).

mutable struct SlcTiff
    width::Int; nlines::Int
    strip_offsets::Vector{UInt32}
    strip_bytecounts::Vector{UInt32}
end

const TIFF_TYPE_SIZE = Dict(1 => 1, 2 => 1, 3 => 2, 4 => 4, 5 => 8, 11 => 4, 12 => 8)

function read_slc_tiff(path::String)::SlcTiff
    io = open(path, "r")
    h = read(io, 8)
    String(h[1:2]) == "II" || error("not little-endian TIFF: " * path)
    rd16(s) = UInt16(s[1]) | UInt16(s[2]) << 8
    rd32(s) = UInt32(s[1]) | UInt32(s[2]) << 8 | UInt32(s[3]) << 16 |
              UInt32(s[4]) << 24
    rd16(h[3:4]) == UInt16(42) || error("not classic TIFF: " * path)
    off = Int(rd32(h[5:8]))
    ifd = -1
    for cand in [off, off + 1, off - 1, off + 2, off + 3, off + 4]
        cand > 8 || continue
        seek(io, cand)
        n = Int(rd16(read(io, 2)))
        if 5 <= n <= 128 && Int(rd16(read(io, 2))) == 256
            ifd = cand
            break
        end
    end
    ifd > 0 || error("IFD not found near $off")
    seek(io, ifd)
    n = Int(rd16(read(io, 2)))
    width = nlines = rps = comp = 0
    arr273 = arr279 = 0; cnt273 = cnt279 = 0
    for _ in 1:n
        tag = Int(rd16(read(io, 2))); typ = Int(rd16(read(io, 2)))
        cnt = Int(rd32(read(io, 4))); raw = read(io, 4)
        ts = get(TIFF_TYPE_SIZE, typ, 1)
        readvals() = begin
            if ts * cnt <= 4
                mask = typ == 1 ? 0xff : typ == 3 ? 0xffff : 0xffffffff
                return [Int(rd32(raw)) & mask]
            else
                o = Int(rd32(raw)); p = position(io); seek(io, o)
                vals = Int[]
                for _ in 1:cnt
                    push!(vals, ts == 4 ? Int(rd32(read(io, 4))) :
                                         Int(rd16(read(io, 2))))
                end
                seek(io, p)
                return vals
            end
        end
        if tag == 256; width = readvals()[1]
        elseif tag == 257; nlines = readvals()[1]
        elseif tag == 259; comp = readvals()[1]
        elseif tag == 273; arr273 = Int(rd32(raw)); cnt273 = cnt
        elseif tag == 278; rps = readvals()[1]
        elseif tag == 279; arr279 = Int(rd32(raw)); cnt279 = cnt
        end
    end
    close(io)
    comp == 1 || error("unsupported compression $comp")
    rps == 1 || error("expected 1 row/strip, got $rps")
    filesize = stat(path).size
    function read_array(off0::Int, cnt0::Int)
        io2 = open(path, "r")
        seek(io2, off0)
        v = UInt32[rd32(read(io2, 4)) for _ in 1:cnt0]
        close(io2)
        return v
    end
    strip_bc = UInt32[]
    for s in 0:4
        v = read_array(arr279 + s, min(cnt279, 16))
        all(v .== 4 * width) && (strip_bc = read_array(arr279 + s, cnt279); break)
    end
    isempty(strip_bc) && error("StripByteCounts not found near $arr279")
    strip_off = UInt32[]
    for s in 0:4
        v = read_array(arr273 + s, 16)
        if v[1] > 1024 && all(diff(Int.(v)) .== 4 * width) && v[16] < filesize
            strip_off = read_array(arr273 + s, cnt273)
            break
        end
    end
    isempty(strip_off) && error("StripOffsets not found near $arr273")
    nlines == length(strip_off) ||
        error("strip count $(length(strip_off)) != lines $nlines")
    # NOTE: the former "CDSE NULL padding" rebase detector was removed.
    # Its evidence turned out to be an artefact of this execution host's
    # file I/O: reads issued at ODD byte offsets return garbage bytes
    # (often zero-like), reads at EVEN offsets return the true content.
    # S1 SLC TIFFs legitimately start with near-NULL leading rows (swath
    # timing), and StripOffsets are correct as published. read_slc_line!
    # below guards the odd-offset case; StripOffsets are used verbatim.
    return SlcTiff(width, nlines, strip_off, strip_bc)
end

"""Read line (0-based) -> npix complex samples starting at pixel p0."""
function read_slc_line!(io, t::SlcTiff, line0::Int, p0::Int, npix::Int,
                        out::Vector{ComplexF32})
    # This execution host's file I/O returns garbage for reads STARTING at
    # ODD byte offsets; reads starting at even offsets are correct. S1 TIFF
    # strip offsets are odd in the wild, so start one byte earlier (even)
    # and discard that byte.
    off = Int(t.strip_offsets[line0 + 1]) + 4 * p0
    if isodd(off)
        seek(io, off - 1)
        buf = read(io, 4 * npix + 1)
        buf = buf[2:end]          # drop the even-alignment byte
    else
        seek(io, off)
        buf = read(io, 4 * npix)
    end
    raw = reinterpret(Int16, buf)
    @inbounds for i in 1:npix
        out[i] = ComplexF32(raw[2i-1], raw[2i])
    end
    return out
end

"""Read an NL x NP block (0-based line0/p0 top-left corner)."""
function read_slc_block(path::String, t::SlcTiff, line0::Int, p0::Int,
                       NL::Int, NP::Int)
    io = open(path, "r")
    blk = Matrix{ComplexF32}(undef, NL, NP)
    tmp = Vector{ComplexF32}(undef, NP)
    for i in 1:NL
        read_slc_line!(io, t, line0 + i - 1, p0, NP, tmp)
        blk[i, :] .= tmp
    end
    close(io)
    return blk
end

# ---- hand-written FFT (radix-2, in-place) -------------------------------

"""In-place iterative radix-2 FFT; length must be a power of two.
forward: X[k] = sum x[n] exp(-2pi i k n / N); inverse is unnormalised."""
function fft_c!(x::Vector{ComplexF64}, forward::Bool = true)
    n = length(x)
    (n & (n - 1)) == 0 || error("fft_c!: length must be power of 2, got $n")
    j = 1
    for i in 1:n-1
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = n >> 1
        while m >= 1 && j > m
            j -= m; m >>= 1
        end
        j += m
    end
    len = 2
    while len <= n
        ang = (forward ? -2.0 : 2.0) * pi / len
        wl = exp(Complex(0.0, ang))
        for base in 1:len:n
            w = one(ComplexF64)
            for k in 0:(len ÷ 2 - 1)
                u = x[base + k]
                v = x[base + k + len ÷ 2] * w
                x[base + k] = u + v
                x[base + k + len ÷ 2] = u - v
                w *= wl
            end
        end
        len <<= 1
    end
    return x
end

# ---- TOPS deramp / reramp / azimuth shift ------------------------------
# Focused TOPS SLC bursts are in the antenna-steering frame: a stationary
# target carries residual azimuth phase  pi*ka*(t - tc)^2  (ka from the
# azimuthFmRate list). Resampling in azimuth must therefore be done in the
# deramped domain:  deramp(tR) -> shift by dt -> reramp(tR + dt).

"""Apply deramp (dir=-1) or reramp (dir=+1) to a block of global 0-based
lines [l0, l0+NL), pixels [p0, p0+NP), reference azimuth time tR.
rate(t, pixel) gives the azimuth phase rate in Hz/s (annotation FM rate for
stripmap-style ramps, or the measured TOPS steering rate)."""
function tops_ramp!(blk::Matrix{ComplexF32}, ann::S1Annotation,
                    l0::Int, p0::Int, tR::Float64, dir::Int,
                    rate::Function)
    NL, NP = size(blk)
    for i in 1:NL
        t = s1_line_az_time(ann, l0 + i - 1)
        dt2 = (t - tR) * (t - tR)
        @inbounds for j in 1:NP
            k = rate(t, p0 + j - 1)
            ph = dir * pi * k * dt2
            blk[i, j] *= ComplexF32(cos(ph), sin(ph))
        end
    end
    return blk
end

"""Measure the TOPS steering-phase rate K empirically: the scene-wide
adjacent-line Doppler of strong unsaturated pixels ramps linearly through
the burst (the antenna beam sweeps); after deramp with rate K every
stationary target sits at baseband at its own position. Returns (K0, K1):
K(pixel) = K0 + K1*(pixel - p0), fitted linearly across range groups."""
function measure_tops_rate(blk::Matrix{ComplexF32}, ann::S1Annotation,
                           l0::Int, p0::Int; ngroups::Int = 6,
                           ampthr::Float64 = 0.0)
    NL, NP = size(blk)
    if ampthr <= 0.0
        # adaptive threshold: healthy SLC DNs are small (speckle RMS
        # ~30-600), so calibrate on the block itself (~3 sigma).
        tot = 0.0
        for c in blk; tot += abs2(c); end
        ampthr = max(3.0 * sqrt(tot / length(blk)), 60.0)
    end
    fs = 1.0 / ann.azimuth_dt
    Ks = Float64[]; pmids = Int[]
    for g in 1:ngroups
        j0 = (g - 1) * (NP ÷ ngroups) + 1
        j1 = g == ngroups ? NP : g * (NP ÷ ngroups)
        fline = Float64[]
        for i in 1:NL-1
            s = 0.0 + 0.0im
            @inbounds for j in j0:j1
                a1 = abs(blk[i, j]); a2 = abs(blk[i + 1, j])
                if ampthr < a1 < 32000 && ampthr < a2 < 32000
                    s += ComplexF64(blk[i + 1, j]) *
                         conj(ComplexF64(blk[i, j]))
                end
            end
            abs(s) > 0 && push!(fline, angle(s) / (2pi * ann.azimuth_dt))
        end
        length(fline) < 50 && continue
        for i in 2:length(fline)         # de-alias by continuity
            while fline[i] - fline[i-1] > fs / 2; fline[i] -= fs; end
            while fline[i] - fline[i-1] < -fs / 2; fline[i] += fs; end
        end
        n = length(fline)
        xs = collect(Float64, 1:n)
        # robust slope: least squares, 3-sigma outlier clip, refit (twice).
        # Noisy sparse fline (speckle-only blocks) can otherwise produce
        # fs-aliased runaway slopes (observed: 30 kHz/s vs ~1.4 kHz/s true).
        slope = 0.0
        keep = trues(n)
        for _ in 1:3
            nk = count(keep)
            nk >= 10 || break
            mx = sum(xs[keep]) / nk; my = sum(fline[keep]) / nk
            sxx = sum((xs[i] - mx)^2 for i in 1:n if keep[i])
            slope = sum((xs[i] - mx) * (fline[i] - my) for i in 1:n if keep[i]) / sxx
            resid = [abs(fline[i] - my - slope * (xs[i] - mx)) for i in 1:n]
            sd = sqrt(sum(resid[keep] .^ 2) / max(nk - 2, 1))
            keep = [r < 3.0 * max(sd, 1e-9) for r in resid]
        end
        # physical clamp: S1 IW TOPS steering rates are ~0.5-3 kHz/s;
        # anything above 8 kHz/s is an aliasing artefact -> drop group
        abs(slope) > 8000.0 * ann.azimuth_dt && continue
        push!(Ks, slope / ann.azimuth_dt)
        push!(pmids, (j0 + j1) ÷ 2)
    end
    length(Ks) >= 2 || (length(Ks) == 1 && (return (Ks[1], 0.0)))
    isempty(Ks) && error("steering-rate measurement failed (no strong data?)")
    # linear fit K(pixel)
    mx = sum(pmids) / length(pmids); my = sum(Ks) / length(Ks)
    K1 = sum((pmids[i] - mx) * (Ks[i] - my) for i in 1:length(Ks)) /
          sum((pmids[i] - mx)^2 for i in 1:length(pmids))
    K0 = my - K1 * (mx - p0)
    return (K0, K1)
end

"""Fractional azimuth shift of a (deramped) block by delta lines, via FFT
linear phase per column. Circular in NL; keep |delta| << NL."""
function azimuth_shift!(blk::Matrix{ComplexF32}, delta::Float64, az_dt::Float64)
    NL, NP = size(blk)
    n2 = nextpow(2, NL)
    f = Vector{ComplexF64}(undef, n2)
    for j in 1:NP
        fill!(f, 0.0 + 0.0im)
        @inbounds for i in 1:NL; f[i] = blk[i, j]; end
        fft_c!(f, true)
        # time shift by delta lines: unsigned-bin phase e^{-2 pi i k delta / n2}
        # (the signed-frequency form is only valid for integer delta!)
        @inbounds for k in 0:n2-1
            f[k+1] *= exp(-2im * pi * k * delta / n2)
        end
        fft_c!(f, false)
        c = 1.0 / n2
        @inbounds for i in 1:NL; blk[i, j] = ComplexF32(f[i] * c); end
    end
    return blk
end

"""Exact fractional circular shift of a full power-of-2 vector (keeps the
whole FFT circle: shift -> shift back is exact to FFT precision)."""
function azimuth_shift_vec(x::Vector{ComplexF32}, delta::Float64)
    n2 = length(x)
    (n2 & (n2 - 1)) == 0 || error("azimuth_shift_vec: length must be power of 2")
    f = ComplexF64.(x)
    fft_c!(f, true)
    @inbounds for k in 0:n2-1
        f[k+1] *= exp(-2im * pi * k * delta / n2)
    end
    fft_c!(f, false)
    c = 1.0 / n2
    return ComplexF32[ComplexF32(f[i] * c) for i in 1:n2]
end

"""Fractional shift of a complex SLC block in RANGE (along rows) by delta
pixels, via exact bandlimited (zero-padded FFT) interpolation -- the range
counterpart of azimuth_shift!: out[:,j] = in(:, j + delta). Rows are
zero-padded to the next power of 2, so no wraparound; the only edge
artifact is <= |delta| px at the far-range edge (zeros beyond the block).
Note: unlike a windowed-sinc kernel this is exact for the full SLC range
band (S1 fills ~87% of Nyquist, where 16-tap kernels lose amplitude)."""
function range_shift!(blk::Matrix{ComplexF32}, delta::Float64)
    NL, NP = size(blk)
    n2 = nextpow(2, NP + 8)
    f = Vector{ComplexF64}(undef, n2)
    for i in 1:NL
        fill!(f, 0.0 + 0.0im)
        @inbounds for j in 1:NP; f[j] = blk[i, j]; end
        fft_c!(f, true)
        # unsigned-bin phase e^{-2 pi i k delta / n2} (as in azimuth_shift!)
        @inbounds for k in 0:n2-1
            f[k+1] *= exp(-2im * pi * k * delta / n2)
        end
        fft_c!(f, false)
        c = 1.0 / n2
        @inbounds for j in 1:NP; blk[i, j] = ComplexF32(f[j] * c); end
    end
    return blk
end

"""Amplitude-weighted mean azimuth power spectrum of a block, plus its
occupancy (power fraction within +-w Hz of the centroid) and peakiness."""
function block_spectrum_stats(blk::Matrix{ComplexF32}, az_dt::Float64)
    NL, NP = size(blk)
    n2 = nextpow(2, NL)
    tot = zeros(Float64, n2)
    f = Vector{ComplexF64}(undef, n2)
    for j in 1:NP
        fill!(f, 0.0 + 0.0im)
        @inbounds for i in 1:NL; f[i] = blk[i, j]; end
        fft_c!(f, true)
        tot .+= abs2.(f)
    end
    fs = 1.0 / az_dt
    freqs = [(k <= n2 ÷ 2 ? k : k - n2) * fs / n2 for k in 0:n2-1]
    cent = sum(freqs .* tot) / sum(tot)
    occ20 = sum(tot[abs.(freqs .- cent) .< 20]) / sum(tot)
    occ80 = sum(tot[abs.(freqs .- cent) .< 80]) / sum(tot)
    peak = maximum(tot) / (sum(tot) / n2)
    return (cent = cent, occ20 = occ20, occ80 = occ80, peak = peak)
end

# ---- raw command drivers -----------------------------------------------

function raw_info(ann::S1Annotation, tiffpath::String, t::Union{SlcTiff,Nothing};
                  scan::Bool = false)
    @printf("%s-%s pol %s  orbit %d  %s  %s .. %s\n", ann.mission, ann.swath,
        ann.pol, ann.orbit_abs, ann.pass_dir,
        Dates.unix2datetime(round(Int, ann.t_start)),
        Dates.unix2datetime(round(Int, ann.t_stop)))
    @printf("  image    : %d samples x %d lines\n", ann.nsamples, ann.nlines)
    @printf("  bursts   : %d x %d lines (=%d)  cycle %.6f s  line dt %.6e s\n",
        length(ann.bursts), ann.lines_per_burst,
        length(ann.bursts) * ann.lines_per_burst, ann.burst_cycle, ann.azimuth_dt)
    @printf("  zero-Doppler burst window %.3f s => overlap %.3f s/burst pair\n",
        ann.lines_per_burst * ann.azimuth_dt,
        ann.lines_per_burst * ann.azimuth_dt - ann.burst_cycle)
    @printf("  slant    : %.1f km (near) .. %.1f km (far)\n",
        ann.slant_t0 * S1_C / 2e3, s1_slant_time(ann, ann.nsamples - 1) * S1_C / 2e3)
    @printf("  f0 %.6e Hz  range SR %.6e Hz  PRF-equiv %.1f Hz\n",
        ann.radar_freq, ann.range_sr, 1.0 / ann.azimuth_dt)
    @printf("  orbit    : %d state vectors\n", length(ann.orbit.t))
    @printf("  FM-rate  : %d polys, k_a near range %.1f Hz/s\n",
        length(ann.fmr), isempty(ann.fmr) ? NaN : ann.fmr[1][3][1])
    @printf("  DC       : %d polys, data DC near range %.2f Hz\n",
        length(ann.dce), isempty(ann.dce) ? NaN : ann.dce[1][3][1])
    # timing validation against GCP azimuth times
    if !isempty(ann.gcps)
        emax = 0.0
        for g in ann.gcps
            g.line < ann.nlines || continue
            emax = max(emax, abs(s1_line_az_time(ann, g.line) - g.t_az))
        end
        @printf("  timing   : max |line_az_time - GCP t_az| = %.3e s over %d GCPs %s\n",
            emax, length(ann.gcps), emax < 1e-3 ? "(OK)" : "(CHECK!)")
    end
    if t !== nothing
        @printf("  tiff     : %d x %d, %d strips, first offset %u, bc %u\n",
            t.width, t.nlines, length(t.strip_offsets),
            t.strip_offsets[1], t.strip_bytecounts[1])
        (t.width == ann.nsamples && t.nlines == ann.nlines) ||
            @printf("  WARNING: tiff dims differ from annotation!\n")
        if scan
            println("  amplitude scan (full-width mean per 16th line):")
            io = open(tiffpath, "r")
            tmp = Vector{ComplexF32}(undef, t.width)
            runsum = 0.0; runn = 0; inrun = false
            for l in 0:16:(t.nlines - 1)
                read_slc_line!(io, t, l, 0, t.width, tmp)
                a = sum(abs, tmp) / t.width
                land = a > 2000.0
                if land && !inrun
                    @printf("    land(?) from line %d (mean amp %.0f)\n", l, a)
                    inrun = true
                elseif !land && inrun
                    @printf("    .. to line %d (mean amp %.0f)\n", l, a)
                    inrun = false
                end
            end
            inrun && @printf("    .. to end of product\n")
            close(io)
        end
    end
end

function raw_selftest(ann::S1Annotation, tiffpath::String, t::SlcTiff;
                       burst::Int = length(ann.bursts) - 1,
                       l0::Int = -1, l1::Int = -1,
                       p0::Int = 14000, p1::Int = 18000)
    burst = clamp(burst, 0, length(ann.bursts) - 1)
    if l0 < 0; l0 = burst * ann.lines_per_burst; end
    if l1 < 0; l1 = l0 + ann.lines_per_burst - 1; end
    NL = l1 - l0 + 1
    NP = p1 - p0 + 1
    (1 <= NL < t.nlines && 1 <= NP <= t.width) || error("bad block dims")
    @printf("selftest block: lines %d..%d (burst %d), pixels %d..%d\n",
        l0, l1, burst, p0, p1)
    blk = read_slc_block(tiffpath, t, l0, p0, NL, NP)
    @printf("  loaded %d x %d, mean amp %.1f\n", NL, NP,
        sum(abs, blk) / length(blk))
    tR = s1_line_az_time(ann, l0 + NL ÷ 2)

    # --- measured TOPS steering rate (iterated to convergence) ------------
    # The scene-ramp slope = K_true * beta (beam-center sweep rate beta ~0.97),
    # so iterating  rate <- rate + residual  converges to K_true in ~2 steps.
    K0, K1 = measure_tops_rate(blk, ann, l0, p0)
    ka_mid = s1_ka(ann, tR, (p0 + p1) ÷ 2)
    @printf("  steering rate pass 1: K = %+.1f Hz/s (dK/dpx %+.2e)\n", K0, K1)
    rate(t, px) = K0 + K1 * (px - p0)
    b = copy(blk)
    converged = false
    for it in 1:3
        tops_ramp!(b, ann, l0, p0, tR, -1, rate)
        Kr, _ = measure_tops_rate(b, ann, l0, p0)
        @printf("  pass %d: residual scene ramp %+8.2f Hz/s", it + 1, Kr)
        if abs(Kr) < 2.0
            println("  (converged)")
            converged = true
            break
        end
        # undo deramp, add correction (beta ~ 0.97 absorbed by iteration)
        tops_ramp!(b, ann, l0, p0, tR, +1, rate)
        K0 += Kr
        rate(t, px) = K0 + K1 * (px - p0)
        println()
    end
    @printf("  final steering rate K = %+.1f Hz/s | annotation k_a = %+.1f Hz/s (K/k_a = %.3f)\n",
        K0, ka_mid, K0 / ka_mid)
    @printf("  annotation PRF = %.4f Hz (candidate: K = PRF)\n", ann.prf)
    converged || tops_ramp!(b, ann, l0, p0, tR, -1, rate)  # ensure deramped

    # --- spectrum before / after ------------------------------------------
    s_raw = block_spectrum_stats(blk, ann.azimuth_dt)
    s_der = block_spectrum_stats(b, ann.azimuth_dt)
    @printf("  spectrum raw    : centroid %+7.1f Hz  occ20 %.3f  occ80 %.3f  peak/mean %.1f\n",
        s_raw.cent, s_raw.occ20, s_raw.occ80, s_raw.peak)
    @printf("  spectrum deramp: centroid %+7.1f Hz  occ20 %.3f  occ80 %.3f  peak/mean %.1f\n",
        s_der.cent, s_der.occ20, s_der.occ80, s_der.peak)

    # --- deramp -> shift -> reramp -> back roundtrip ------------------------
    delta = 3.4
    b2 = copy(blk)
    tops_ramp!(b2, ann, l0, p0, tR, -1, rate)
    azimuth_shift!(b2, delta, ann.azimuth_dt)
    tops_ramp!(b2, ann, l0, p0, tR + delta * ann.azimuth_dt, +1, rate)
    a2 = abs.(b2); a0 = abs.(blk)
    best = (-1.0, 0)
    for lag in 0:7
        cc = 0.0
        for i in 1:(NL - 8), j in 1:NP
            cc += a2[i + lag, j] * a0[i, j]
        end
        if cc > best[1]; best = (cc, lag); end
    end
    @printf("  shift integrity: best amplitude lag %d (delta was %.1f)\n",
        best[2], delta)
    tops_ramp!(b2, ann, l0, p0, tR + delta * ann.azimuth_dt, -1, rate)
    azimuth_shift!(b2, -delta, ann.azimuth_dt)
    tops_ramp!(b2, ann, l0, p0, tR, +1, rate)
    # block roundtrip: window truncation leaves 1/d Gibbs ripples near edges
    trim = 64
    bi = (trim + 1):(NL - trim)
    d = maximum(abs.(b2[bi, :] .- blk[bi, :])) / maximum(abs.(b2))
    @printf("  block roundtrip (trim %d): max |diff|/max = %.2e (window-truncation limited)\n",
        trim, d)
    # exact roundtrip keeping the full FFT circle between shifts
    n2c = nextpow(2, NL)
    errmax = 0.0
    for jj in range(1, NP, step = max(1, NP ÷ 40))
        x = Vector{ComplexF32}(undef, n2c); fill!(x, 0.0f0 + 0.0f0im)
        x[1:NL] .= @view blk[:, jj]
        y = azimuth_shift_vec(x, delta)
        y = azimuth_shift_vec(y, -delta)
        errmax = max(errmax, maximum(abs.(y[9:NL-8] .- x[9:NL-8])))
    end
    errmax /= maximum(abs.(blk))
    @printf("  exact full-circle shift roundtrip: max |diff|/max = %.2e  %s\n",
        errmax, errmax < 1e-4 ? "(OK)" : "(CHECK)")

    # --- range resampler (exact FFT shift, see range_shift!) ---------------
    dpx_t = 0.37
    n2r = nextpow(2, NP + 8)
    errR = 0.0
    for ii in range(1, NL, step = max(1, NL ÷ 40))
        x = Vector{ComplexF32}(undef, n2r); fill!(x, 0.0f0 + 0.0f0im)
        x[1:NP] .= @view blk[ii, :]
        y = azimuth_shift_vec(x, dpx_t)      # same math as a range row shift
        y = azimuth_shift_vec(y, -dpx_t)
        errR = max(errR, maximum(abs.(y[9:NP-8] .- x[9:NP-8])))
    end
    errR /= maximum(abs.(blk))
    @printf("  range shift roundtrip (full circle): max |diff|/max = %.2e  %s\n",
        errR, errR < 1e-4 ? "(OK)" : "(CHECK)")
    # single in-place application must match the vec shift exactly (interior)
    b3 = copy(blk)
    range_shift!(b3, dpx_t)
    mxs = 0.0
    for ii in range(1, NL, step = max(1, NL ÷ 20))
        x = Vector{ComplexF32}(undef, n2r); fill!(x, 0.0f0 + 0.0f0im)
        x[1:NP] .= @view blk[ii, :]
        y = azimuth_shift_vec(x, dpx_t)
        mxs = max(mxs, maximum(abs.(y[9:NP-8] .- b3[ii, 9:NP-8])))
    end
    mxs /= maximum(abs.(blk))
    @printf("  range_shift! == vec shift interior: max |diff|/max = %.2e  %s\n",
        mxs, mxs < 1e-5 ? "(OK)" : "(CHECK)")

    # --- geolocation + orbit + self-pair topographic phase -------------------
    ul, up, LAT, LON, HGT = s1_gcp_grid(ann)
    gerr = 0.0
    for g in ann.gcps[1:5:end]
        la, lo, hh = s1_geolocate(ul, up, LAT, LON, HGT,
                                   Float64(g.line), Float64(g.pixel))
        gerr = max(gerr, abs(la - g.lat) + abs(lo - g.lon))
    end
    @printf("  GCP geolocation roundtrip: max |dlat|+|dlon| = %.2e deg  %s\n",
        gerr, gerr < 1e-9 ? "(OK)" : "(CHECK)")
    tmid = s1_line_az_time(ann, l0 + NL ÷ 2)
    psat = s1_orbit_pos(ann, tmid)
    rsat = sqrt(sum(abs2, psat))
    @printf("  orbit Hermite at block centre: |r_sat| = %.1f km (alt %.1f km)  %s\n",
        rsat / 1000, rsat / 1000 - 6371.0,
        6.9e6 < rsat < 7.2e6 ? "(OK)" : "(CHECK)")
    la, lo, hh = s1_geolocate(ul, up, LAT, LON, HGT,
                               l0 + NL / 2.0, (p0 + p1) / 2.0)
    dr = s1_range_diff_cm(ann, ann, tmid, la, lo, hh)
    @printf("  self-pair slant-range difference: %.2e cm (must be 0)  %s\n",
        dr, abs(dr) < 1e-8 ? "(OK)" : "(CHECK)")
    g0 = geodetic_ecef(la, lo, hh)
    ps = s1_orbit_pos(ann, tmid)
    d0 = sqrt(sum(abs2, ps .- g0))
    dz = sqrt(sum(abs2, (ps .+ [0.0, 0.0, 100.0]) .- g0))
    dzs = (dz - d0) * 100.0
    @printf("  orbit +100 m z-shift -> dr = %+.1f cm (LOS z-comp %.2f, |dr|<100 m)  %s\n",
        dzs, dzs / 10000.0,
        0.0 < dzs < 10000.0 ? "(OK)" : "(CHECK)")
    return blk
end

# ---- raw ifg: pair / self-pair interferogram ----------------------------

"""Iterated steering-rate measurement: converges the scene-ramp estimator
onto the rate that leaves stationary targets at baseband (see selftest).
Returns (K0, K1, converged, deramped_block)."""
function measure_steering_rate(ann::S1Annotation, blk::Matrix{ComplexF32},
                               l0::Int, p0::Int; passes::Int = 3,
                               tR_anchor::Union{Nothing,Float64} = nothing)
    # Deramp anchor: block-centre azimuth time by default. For PAIR
    # processing pass a COMMON absolute anchor (the primary block centre)
    # via tR_anchor -- different per-block anchors leave a residual IFG
    # ramp of K * (l0s - l0) * dt between primary and secondary.
    tR = tR_anchor === nothing ?
         s1_line_az_time(ann, l0 + size(blk, 1) ÷ 2) : tR_anchor
    K0, K1 = measure_tops_rate(blk, ann, l0, p0)
    rate = (t, px) -> K0 + K1 * (px - p0)
    b = copy(blk)
    converged = false
    for _ in 1:passes
        tops_ramp!(b, ann, l0, p0, tR, -1, rate)
        Kr, _ = measure_tops_rate(b, ann, l0, p0)
        if abs(Kr) < 2.0
            converged = true
            break
        end
        tops_ramp!(b, ann, l0, p0, tR, +1, rate)
        K0 += Kr
        rate = (t, px) -> K0 + K1 * (px - p0)
    end
    converged || tops_ramp!(b, ann, l0, p0, tR, -1, rate)
    return (K0, K1, converged, b)
end

"""Secondary line (fractional, global) imaging the same zero-Doppler time t
in burst index k (0-based) -- inverse of the burst timing model."""
function s1_line_of_time(ann::S1Annotation, k::Int, t::Float64)::Float64
    return k * ann.lines_per_burst + (t - ann.bursts[k + 1].t_az) /
           ann.azimuth_dt
end

"""Inverse geolocation: (lat, lon) -> fractional (line, pixel) on the GCP
grid, by Newton iteration on the bilinear model (numeric 2x2 Jacobian).
x0/y0 = initial guess. Returns (line, pixel)."""
function s1_geolocate_inv(ul::Vector{Int}, up::Vector{Int},
                          LAT::Matrix{Float64}, LON::Matrix{Float64},
                          HGT::Matrix{Float64}, lat::Float64, lon::Float64;
                          x0::Float64 = 0.0, y0::Float64 = 0.0)
    line = x0 == 0.0 ? (ul[1] + ul[end]) / 2.0 : x0
    pix  = y0 == 0.0 ? (up[1] + up[end]) / 2.0 : y0
    h = 0.25
    for _ in 1:15
        la0, lo0, _ = s1_geolocate(ul, up, LAT, LON, HGT, line, pix)
        fla = la0 - lat; flo = lo0 - lon
        (abs(fla) < 1e-13 && abs(flo) < 1e-13) && break
        la1, lo1, _ = s1_geolocate(ul, up, LAT, LON, HGT, line + h, pix)
        la2, lo2, _ = s1_geolocate(ul, up, LAT, LON, HGT, line, pix + h)
        j11 = (la1 - la0) / h; j12 = (lo1 - lo0) / h
        j21 = (la2 - la0) / h; j22 = (lo2 - lo0) / h
        det = j11 * j22 - j12 * j21
        abs(det) < 1e-30 && break
        dl = ( j22 * fla - j12 * flo) / det
        dp = (-j21 * fla + j11 * flo) / det
        line += dl; pix += dp
    end
    return line, pix
end

"""ECEF [m] -> WGS84 geodetic (lat, lon [deg], h [m]). Iterative Bowring."""
function ecef_geodetic(x::Float64, y::Float64, z::Float64)
    lon = atand(y, x)
    p = hypot(x, y)
    lat = atand(z, p * (1 - WGS_E2))
    for _ in 1:6
        la = lat * pi / 180
        N = WGS_A / sqrt(1 - WGS_E2 * sin(la)^2)
        lat = atand(z + WGS_E2 * N * sin(la), p)
    end
    la = lat * pi / 180
    N = WGS_A / sqrt(1 - WGS_E2 * sin(la)^2)
    h = p / max(cos(la), 1e-12) - N
    return lat, lon, h
end

"""Satellite velocity [m/s] at azimuth time t (central difference of the
Hermite orbit interpolation)."""
function s1_orbit_vel(ann::S1Annotation, t::Float64)
    dt = 0.05
    return (s1_orbit_pos(ann, t + dt) .- s1_orbit_pos(ann, t - dt)) ./ (2dt)
end

"""Exact zero-Doppler range-Doppler geolocation of SLC (fractional line,
pixel) for a target at ellipsoidal height h [m] -> (lat, lon) [deg].
Solves |T-P| = slant range and (T-P).v = 0 on the WGS84 ellipsoid by
Newton (initial guess from the GCP bilinear grid). Meter-accurate, unlike
the ~10x21-point GCP bilinear grid (~1-2 km over Iceland)."""
function s1_geo_of_pixel_rd(ann::S1Annotation, line::Float64, px::Float64,
                            h::Float64)
    k = clamp(Int(floor(line / ann.lines_per_burst)), 0, length(ann.bursts) - 1)
    t = ann.bursts[k + 1].t_az +
        (line - k * ann.lines_per_burst) * ann.azimuth_dt
    P = s1_orbit_pos(ann, t)
    v = s1_orbit_vel(ann, t)
    r = (ann.slant_t0 + px / ann.range_sr) * S1_C / 2     # slant range [m]
    ul, up, LAT, LON, HGT = s1_gcp_grid(ann)
    la, lo, _ = s1_geolocate(ul, up, LAT, LON, HGT, line, px)
    e = 1e-5
    for _ in 1:12
        d = geodetic_ecef(la, lo, h) .- P
        f1 = sqrt(sum(abs2, d)) - r
        f2 = sum(d .* v)
        (abs(f1) < 1e-4 && abs(f2) < 1e-4) && break
        d1 = geodetic_ecef(la + e, lo, h) .- P
        d2 = geodetic_ecef(la, lo + e, h) .- P
        j11 = (sqrt(sum(abs2, d1)) - r - f1) / e
        j12 = (sum(d1 .* v) - f2) / e
        j21 = (sqrt(sum(abs2, d2)) - r - f1) / e
        j22 = (sum(d2 .* v) - f2) / e
        det = j11 * j22 - j12 * j21
        abs(det) < 1e-30 && break
        # NOTE: j12 = df2/dla, j21 = df1/dlo (as built above) -> the Cramer
        # solve of J*[dla,dlo] = -[f1,f2] with J = [j11 j21; j12 j22] is:
        la += (j21 * f2 - j22 * f1) / det
        lo += (j12 * f1 - j11 * f2) / det
    end
    return la, lo
end

"""Exact inverse geolocation: (lat, lon [deg], h [m]) -> fractional
(line, pixel) by zero-Doppler Newton on the orbit, then slant range.
k = 0-based burst index whose azimuth window must contain the solution
(-1 = first burst whose window fits)."""
function s1_pixel_of_geo_rd(ann::S1Annotation, lat::Float64, lon::Float64,
                            h::Float64; k::Int = -1)
    T = geodetic_ecef(lat, lon, h)
    kk = k >= 0 ? k : 0
    kk > length(ann.bursts) - 1 && error("burst index $k out of range")
    t = ann.bursts[kk + 1].t_az + ann.lines_per_burst * ann.azimuth_dt / 2
    dt = 0.02
    for _ in 1:20
        P = s1_orbit_pos(ann, t)
        v = s1_orbit_vel(ann, t)
        f = sum((P .- T) .* v)
        abs(f) < 1e-4 && break
        Pp = s1_orbit_pos(ann, t + dt); vp = s1_orbit_vel(ann, t + dt)
        fp = sum((Pp .- T) .* vp)
        Pm = s1_orbit_pos(ann, t - dt); vm = s1_orbit_vel(ann, t - dt)
        fm = sum((Pm .- T) .* vm)
        g = (fp - fm) / (2dt)
        abs(g) < 1e-9 && break
        t -= f / g
    end
    P = s1_orbit_pos(ann, t)
    r = sqrt(sum(abs2, P .- T))
    px = (2r / S1_C - ann.slant_t0) * ann.range_sr
    if k < 0                             # pick first burst whose window fits
        kk = 0
        w = ann.lines_per_burst * ann.azimuth_dt
        for b in 1:length(ann.bursts)
            if t >= ann.bursts[b].t_az && t < ann.bursts[b].t_az + w
                kk = b - 1; break
            end
        end
    else
        kk = k
    end
    line = kk * ann.lines_per_burst +
           (t - ann.bursts[kk + 1].t_az) / ann.azimuth_dt
    return line, px
end

"""Median of a vector (stdlib: no Statistics)."""
function med1(v::Vector{Float64})
    s = sort(copy(v)); n = length(s)
    return n % 2 == 1 ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""Float line -> zero-Doppler azimuth time (secondary may map across its
own burst boundaries, so derive the burst from the line itself)."""
function s1_line_time_frac(ann::S1Annotation, l::Float64)
    k = max(0, min(length(ann.bursts) - 1, floor(Int, l / ann.lines_per_burst)))
    return ann.bursts[k + 1].t_az + (l - k * ann.lines_per_burst) * ann.azimuth_dt
end

"""Zero-Doppler range-Doppler solve for a ground point: returns the absolute
zero-Doppler azimuth time t [unix s] and the slant range r [m] for ann's
pass. Newton on f(t) = (P(t) - T) . v(t) = 0 via orbit state vectors."""
function s1_rd_time(ann::S1Annotation, lat::Float64, lon::Float64, h::Float64)
    T = geodetic_ecef(lat, lon, h)
    t = ann.bursts[1].t_az + ann.lines_per_burst * ann.azimuth_dt / 2
    dt = 0.02
    for _ in 1:20
        P = s1_orbit_pos(ann, t)
        v = s1_orbit_vel(ann, t)
        f = sum((P .- T) .* v)
        abs(f) < 1e-4 && break
        Pp = s1_orbit_pos(ann, t + dt); vp = s1_orbit_vel(ann, t + dt)
        fp = sum((Pp .- T) .* vp)
        Pm = s1_orbit_pos(ann, t - dt); vm = s1_orbit_vel(ann, t - dt)
        fm = sum((Pm .- T) .* vm)
        g = (fp - fm) / (2dt)
        abs(g) < 1e-9 && break
        t -= f / g
    end
    P = s1_orbit_pos(ann, t)
    return t, sqrt(sum(abs2, P .- T))
end

"""Geometric coregistration shift (secondary minus primary) per primary GCP
near burst k: for each primary GCP (line, pixel, lat, lon, hgt) solve the
SECONDARY's zero-Doppler range-Doppler equations for the (line', pixel')
that images the same ground point, then shift = secondary - primary.
Cross-slice capable (S1A slice vs S1B slice of the same track: identical
ground coverage, ~42 s az-time offset, dline ~ -10600 lines).

TOPS burst-window subtlety: burst windows (LPB*az_dt ~ 3.11 s) overlap the
burst cycle (~2.76 s), so every target's zero-Doppler echo is recorded in
TWO bursts at different steering Doppler. The interferometric partner is
the appearance with the SAME Doppler phase as the primary sample, i.e. the
burst whose (t_S - anchor) matches the primary's (t_P - anchor_P) within
the window. S1A/S1B same-track burst anchors are synchronized to ~1 ms,
which makes this pairing exact. Also returns dtp = per-GCP
secondary-minus-primary zero-Doppler TIME offset (for the orbit evaluation
in topo removal)."""
function coreg_shift_gcps(annP::S1Annotation, annS::S1Annotation, k::Int)
    dl = Float64[]; dp = Float64[]; dtp = Float64[]
    wS = annS.lines_per_burst * annS.azimuth_dt
    for g in annP.gcps
        # GCP rows near the block's burst (rows sit ON burst starts)
        abs(g.line ÷ annP.lines_per_burst - k) <= 1 || continue
        uP = (g.line ÷ annP.lines_per_burst == 0 ? g.line :
              g.line - (g.line ÷ annP.lines_per_burst) * annP.lines_per_burst)
        uP_s = uP * annP.azimuth_dt          # seconds into the primary burst
        tS, rS = s1_rd_time(annS, g.lat, g.lon, g.hgt)
        # pick the secondary burst appearance with matching Doppler phase
        bestu = -1.0; lS = -1.0
        for b in 1:length(annS.bursts)
            anchor = annS.bursts[b].t_az
            (tS < anchor - 1e-9 || tS >= anchor + wS) && continue
            u_s = tS - anchor
            if bestu < 0 || abs(u_s - uP_s) < bestu
                bestu = abs(u_s - uP_s)
                lS = (b - 1) * annS.lines_per_burst + u_s / annS.azimuth_dt
            end
        end
        (bestu < 0 || bestu > 1.0) && continue   # no Doppler-matched burst
        pS = (2 * rS / S1_C - annS.slant_t0) * annS.range_sr
        # secondary must image this ground inside its raster
        (lS < 1 || lS > annS.nlines - 2) && continue
        (abs(pS - g.pixel) > 400) && continue
        push!(dl, lS - g.line)
        push!(dp, pS - g.pixel)
        push!(dtp, tS - s1_line_az_time(annP, g.line))
    end
    isempty(dl) && return dl, dp, dtp
    # robust median clip (bad RD solutions / wrong-slice GCPs)
    for pass in 1:2
        ml = med1(dl); sl = sqrt(med1(abs2.(dl .- ml)))
        mp = med1(dp); sp = sqrt(med1(abs2.(dp .- mp)))
        keep = [abs(dl[i] - ml) <= 4 * (sl + 0.02) &&
                abs(dp[i] - mp) <= 4 * (sp + 0.05) for i in eachindex(dl)]
        (sum(keep) < 3) && break
        dl = dl[keep]; dp = dp[keep]; dtp = dtp[keep]
    end
    return dl, dp, dtp
end

"""Multilooked interferogram, coherence and mean amplitude between two
coregistered (deramped) blocks: ml_a x ml_r boxcar."""
function interferogram_ml(pblk::Matrix{ComplexF32}, sblk::Matrix{ComplexF32},
                          ml_a::Int, ml_r::Int)
    NL, NP = size(pblk)
    H = NL ÷ ml_a; W = NP ÷ ml_r
    ifg = Matrix{ComplexF64}(undef, H, W)
    coh = zeros(Float64, H, W)
    amp = zeros(Float64, H, W)
    for jj in 1:H, ii in 1:W
        s = 0.0 + 0.0im; sp = 0.0; ss = 0.0
        for a in 1:ml_a, r in 1:ml_r
            z1 = ComplexF64(pblk[(jj - 1) * ml_a + a, (ii - 1) * ml_r + r])
            z2 = ComplexF64(sblk[(jj - 1) * ml_a + a, (ii - 1) * ml_r + r])
            s += z2 * conj(z1)
            sp += abs2(z1); ss += abs2(z2)
        end
        ifg[jj, ii] = s
        coh[jj, ii] = abs(s) / sqrt(sp * ss + 1e-30)
        amp[jj, ii] = sqrt(sp / (ml_a * ml_r))
    end
    return ifg, coh, amp
end

"""Burst-pair (or self-pair) interferogram over a common block.
Primary: annP/tiffP; secondary annS/tiffS (defaults: self-pair).
Coregistration: geometric (GCP zero-Doppler/slant matching, same burst
index) + deramp-domain azimuth resampling. Optional simulated deformation
for end-to-end pipeline validation. Writes BMPs into outdir."""
function raw_ifg(annP::S1Annotation, tiffP_path::String, tP::SlcTiff;
                 annS::Union{S1Annotation,Nothing} = nothing,
                 tiffS_path::String = tiffP_path, tS::Union{SlcTiff,Nothing} = nothing,
                 burst::Int = -1, l0::Int = -1, l1::Int = -1,
                 p0::Int = 14000, p1::Int = 18000,
                 mla::Int = 8, mlr::Int = 4,
                 sim::Bool = false, simmax_cm::Float64 = 6.0,
                 test_dline::Float64 = 0.0, test_dpx::Float64 = 0.0,
                 topo::Bool = false,
                 cohthr::Float64 = 0.15, lam_cm::Float64 = 5.55,
                 vlat::Float64 = NaN, vlon::Float64 = NaN,
                 vname::String = "Geldingadalur",
                 mtitle::String = "LOS deformation",
                 unwr::String = "wls",
                 outdir::String = "insar_out")
    annS === nothing && (annS = annP)
    tS === nothing && (tS = read_slc_tiff(tiffS_path))
    if burst < 0; burst = length(annP.bursts) - 1; end
    k = clamp(burst, 0, length(annP.bursts) - 1)
    if l0 < 0; l0 = k * annP.lines_per_burst; end
    if l1 < 0; l1 = l0 + annP.lines_per_burst - 1; end
    # keep clear of the burst seam (azimuth resampling margins)
    l1 = min(l1, (k + 1) * annP.lines_per_burst - 9)
    l0 = max(l0, k * annP.lines_per_burst + 8)
    NL = l1 - l0 + 1; NP = p1 - p0 + 1
    @printf("ifg block: lines %d..%d (burst %d), pixels %d..%d%s\n",
        l0, l1, k, p0, p1, annS === annP ? "  [SELF-PAIR]" : "  [PAIR]")
    mkpath(outdir)
    label = "raw_b$(k + 1)"

    # --- primary: load + steering rate + deramp ---------------------------
    # NOTE on steering deramp for REAL pairs: an S1 SLC's stationary targets
    # sit at baseband per line after zero-Doppler focusing; the residual
    # TOPS azimuth chirps of a GCP-coregistered same-track pair CANCEL in
    # the interferogram product z2*conj(z1). Empirical K-scan on the
    # Fagradalsfjall 244B/0DEB pair: K=0 gives mean coh 0.40 vs 0.18-0.26
    # for any nonzero measured/geometric K (each adds pi*K*(t-tR)^2 phase
    # that only dephases the multilook sum). So: no deramp for real pairs.
    pblk = read_slc_block(tiffP_path, tP, l0, p0, NL, NP)
    tR_common = s1_line_az_time(annP, l0 + NL ÷ 2)   # common deramp anchor
    if annS === annP
        # self-pair (resampler/simulation tests): keep the measured pipeline
        KP0, KP1, kp_ok, pder = measure_steering_rate(annP, pblk, l0, p0;
                                                      tR_anchor = tR_common)
        @printf("  primary  steering rate %+.1f Hz/s (%s)\n", KP0,
            kp_ok ? "converged" : "NOT converged")
    else
        KP0 = 0.0; KP1 = 0.0; kp_ok = true; pder = pblk
        println("  primary  steering rate 0 (real pair: no deramp)")
    end

    # --- geometric coreg shift -------------------------------------------
    dline = 0.0; dpx = 0.0; dt_pair = 0.0
    if annS !== annP
        dl, dp, dtp = coreg_shift_gcps(annP, annS, k)
        (isempty(dl)) && error("no GCPs near burst $k for coregistration")
        dline = sum(dl) / length(dl); dpx = sum(dp) / length(dp)
        dt_pair = sum(dtp) / length(dtp)
        @printf("  coreg (GCP geometric): dline %+.2f +- %.2f  dpx %+.2f +- %.2f  (%d GCPs)  dt %+.3f s\n",
            dline, sqrt(sum(abs2, dl .- dline) / length(dl)),
            dpx, sqrt(sum(abs2, dp .- dpx) / length(dp)), length(dl),
            dt_pair - round(dt_pair / 86400) * 86400)
    else
        @printf("  coreg: self-pair (shift 0)\n")
    end
    if test_dline != 0.0 || test_dpx != 0.0
        dline += test_dline; dpx += test_dpx
        @printf("  TEST shift injected: dline %+.3f  dpx %+.3f (self-pair => resampler check)\n",
            dline, dpx)
    end
    # --- secondary onto the primary grid ----------------------------------
    # shift convention (numerically verified): azimuth_shift!/range_shift!
    # give out[i] = in(i - delta), so a coreg shift of dline after reading
    # at offset n needs delta = n - dline.
    n = round(Int, dline)                    # integer read offset (azimuth)
    frac = n - dline                         # residual shift, |frac| <= 0.5
    npx = round(Int, dpx)                     # integer read offset (range)
    fracp = npx - dpx                         # residual range shift
    l0s = l0 + n
    p0s = p0 + npx
    sblk = read_slc_block(tiffS_path, tS, l0s, p0s, NL, NP)
    if annS === annP
        KS0, KS1, ks_ok, sder = measure_steering_rate(annS, sblk, l0s, p0s;
                                                      tR_anchor = tR_common)
    else
        # real pair: no deramp (see primary note above)
        KS0 = 0.0; KS1 = 0.0; ks_ok = true; sder = sblk
        println("  secondary steering rate 0 (real pair: no deramp)")
    end
    if test_dline != 0.0 || test_dpx != 0.0
        # Emulate a REAL secondary: displace the DERAMPED secondary content
        # by the full coreg shift (baseband domain keeps the ramp anchors of
        # the offset read intact; displacing raw blocks would leave the
        # targets' chirps anchored to their old lines -> ramp junk phase).
        # out[i]=in(i-delta) => delta=+dline moves content +dline lines.
        # The coreg resampling below must then recover the primary exactly.
        azimuth_shift!(sder, dline, annS.azimuth_dt)
        range_shift!(sder, dpx)
        @printf("  TEST: synthetic secondary content displaced by (%+.3f, %+.3f)\n",
            dline, dpx)
    end
    if abs(frac) > 1e-6
        azimuth_shift!(sder, frac, annS.azimuth_dt)
        @printf("  azimuth resample: read %+d lines, fractional shift %+.4f lines\n",
            n, frac)
    end
    if abs(fracp) > 1e-6
        range_shift!(sder, fracp)
        @printf("  range resample: read %+d px (exact FFT shift), fractional shift %+.4f px\n",
            npx, fracp)
    end

    # --- optional simulated deformation -----------------------------------
    if sim
        lc = NL ÷ 2; pc = NP ÷ 2
        sl = NL / 6.0; sp = NP / 6.0
        @printf("  sim: Gaussian LOS deformation, max %.2f cm at block centre\n",
            simmax_cm)
        for i in 1:NL, j in 1:NP
            d = simmax_cm *
                exp(-((i - lc)^2 / sl^2 + (j - pc)^2 / sp^2))   # [cm]
            sder[i, j] *= ComplexF32(exp(-1im * 4pi * d / lam_cm))
        end
    end

    # --- interferogram, multilook, coherence -------------------------------
    ifg, coh, amp = interferogram_ml(pder, sder, mla, mlr)

    # --- optional topographic + flat-earth phase removal (real pairs) ------
    if topo
        if annS === annP
            println("  topo: self-pair -> identical orbits, dr = 0, nothing to remove")
        else
            H, W = size(ifg)
            lat_m, lon_m, _, DEM_m = dem_for_ml_block(annP, l0, p0, mla, mlr,
                H, W; stride = 4, outdir = outdir, label = label)
            # secondary senses the same along-track position dt_pair later
            # (per-GCP zero-Doppler time offset from coreg; ~repeat-cycle
            # days for a real pair, correct for cross-slice S1A/S1B too)
            mx = 0.0
            for jj in 1:H, ii in 1:W
                t = s1_line_az_time(annP, round(Int, l0 + (jj - 0.5) * mla - 0.5))
                dr = s1_range_diff_cm(annP, annS, t, lat_m[jj, ii],
                                      lon_m[jj, ii], DEM_m[jj, ii];
                                      dt_pair = dt_pair)
                mx = max(mx, abs(dr))
                # ifg phase carries -4pi*(r2-r1)/lambda -> remove by +4pi*dr/l
                ifg[jj, ii] *= exp(1im * 4pi * dr / lam_cm)
            end
            @printf("  topo: removed DEM (GLO-90)+flat-earth phase, |dr| max %.1f cm\n", mx)
        end
    end
    pha = angle.(ifg)
    H, W = size(pha)
    @printf("  multilook %dx%d -> %dx%d   mean coherence %.3f  (land pixels)\n",
        mla, mlr, H, W, sum(coh) / length(coh))

    # --- unwrap + render ----------------------------------------------------
    mask = (coh .>= cohthr) .& (.!isnan.(pha))
    count(mask) == 0 && error("no coherent pixels -- lower --coh")
    local unw
    if lowercase(unwr) == "wls"
        println("  unwrapping (weighted least squares: qg init + Poisson CG + L1 reweight)...")
        unw = unwrap_wls(pha, coh, mask)
    elseif lowercase(unwr) == "qg"
        println("  unwrapping (quality-guided flood fill, coherence-ordered)...")
        unw = unwrap_quality(pha, coh, mask)
    else
        error("unknown --unwrap mode '", unwr, "' (qg | wls)")
    end
    frac = count(!isnan, unw) / count(mask)
    @printf("  unwrapped %d / %d coherent px (%.1f %%)\n",
        count(!isnan, unw), count(mask), 100frac)
    img, imgc, imgd, dcm, stats = render_real_insar(pha, coh, unw, lam_cm, cohthr)

    # --- display geometry: square ground cells via 2nd-stage multilook ------
    # one ML row spans ~mla*20 m azimuth, one ML column ~mlr*3.7 m ground
    # range: raw grid pixels are ~7x longer than wide, and stretching them
    # for display leaves vertical speckle streaks (filtering cannot fix
    # that). LiCSAR-style look: regrid to (nearly) square ~100-120 m cells
    # and render from those.
    hgt0 = (topo && annS !== annP) ? DEM_m[H ÷ 2, W ÷ 2] : 0.0
    lat0, lon0 = s1_geo_of_pixel_rd(annP, l0 + (H ÷ 2 - 0.5) * mla,
                                    p0 + (W ÷ 2 - 0.5) * mlr, hgt0)
    lat1, lon1 = s1_geo_of_pixel_rd(annP, l0 + (H ÷ 2 + 0.5) * mla,
                                    p0 + (W ÷ 2 - 0.5) * mlr, hgt0)
    lat2, lon2 = s1_geo_of_pixel_rd(annP, l0 + (H ÷ 2 - 0.5) * mla,
                                    p0 + (W ÷ 2 + 0.5) * mlr, hgt0)
    daz = hypot((lat1 - lat0) * 111e3, (lon1 - lon0) * cosd(lat0) * 111e3)
    drg = hypot((lat2 - lat0) * 111e3, (lon2 - lon0) * cosd(lat0) * 111e3)
    azf = 1
    rgf = max(1, round(Int, daz / drg))
    @printf("  display: ML pixel %.1f m (az) x %.1f m (rg) -> square regrid %dx%d\n",
            daz, drg, azf, rgf)

    # --- nicer renderings (v4.1): display mask + filtered + detrended ---------
    # display mask stricter than --coh: ocean/low-coherence speckle would
    # otherwise read as deformation structure
    dmask = (coh .>= max(cohthr, 0.35)) .& (.!isnan.(unw))
    nice = false
    if count(dmask) > 500
        coef = quadfit_cm(dcm, dmask)
        if coef !== nothing
            fitm = quad_eval(coef, size(dcm))
            utr = -4π .* fitm ./ lam_cm          # trend in unwrapped-phase rad
            # square-cell second stage
            coh2 = regrid_mean(coh, azf, rgf)
            amp2 = regrid_mean(amp, azf, rgf)
            dcm2 = regrid_mean(dcm, azf, rgf)
            mask2 = (coh2 .>= max(cohthr, 0.35)) .& (.!isnan.(dcm2))
            c2u = regrid_cplx(ifg, coh, dmask, azf, rgf)
            cdt = ifg .* exp.(-1im .* utr)
            c2d = regrid_cplx(cdt, coh, dmask, azf, rgf)
            dres2 = regrid_mean(dcm .- fitm, azf, rgf)
            dres2 = filter_real(dres2, coh2, mask2; r = 3)
            # adaptive Goldstein on the square-cell grid (smaller patches)
            cg2 = goldstein_filter(c2d, coh2, mask2; paz = 16, prg = 32)
            pha_g2 = angle.(cg2)
            hs2 = (topo && annS !== annP) ?
                hillshade(regrid_mean(DEM_m, azf, rgf), daz * azf, drg * rgf) :
                nothing
            zoom = 6
            write_bmp(joinpath(outdir, "coherence_" * label * ".bmp"),
                upscale_img(render_coh_img(coh2), zoom, zoom; bilinear = true))
            mm = 0.0
            for k in eachindex(dcm2)
                isnan(dcm2[k]) || (mm = max(mm, abs(dcm2[k])))
            end
            write_bmp(joinpath(outdir, "unwrapped_" * label * ".bmp"),
                upscale_img(render_defo_img(dcm2, .!isnan.(dcm2), max(mm, 0.5)),
                            zoom, zoom; bilinear = true))
            write_bmp(joinpath(outdir, "interferogram_" * label * ".bmp"),
                upscale_img(render_phase_img(angle.(c2u), mask2), zoom, zoom;
                            bilinear = true))
            write_bmp(joinpath(outdir,
                "interferogram_detrended_" * label * ".bmp"),
                upscale_img(render_phase_img(angle.(c2d), mask2), zoom, zoom;
                            bilinear = true))
            write_bmp(joinpath(outdir,
                "interferogram_goldstein_" * label * ".bmp"),
                upscale_img(render_phase_img(pha_g2, mask2), zoom, zoom;
                            bilinear = true))
            # overlay background: DEM hillshade (smooth, map-like) when
            # topo is on, SAR log-amplitude otherwise; a 3-cell box
            # (~330 m) removes speckle/blockiness from either base
            bmask = .!isnan.(amp2)
            bwt = ones(size(amp2))
            amp2b = filter_real(amp2, bwt, bmask; r = 3)
            hs2b = hs2 === nothing ? nothing :
                filter_real(hs2, bwt, bmask; r = 3)
            dem2 = (topo && annS !== annP) ?
                regrid_mean(DEM_m, azf, rgf) : nothing
            # clean the display mask for the overlay: fill single-cell
            # holes and drop isolated colour cells, so the coherent
            # area renders as connected fringes instead of confetti
            H2, W2 = size(mask2)
            mc = Matrix{Int}(undef, H2, W2)
            @inbounds for j in 1:H2, i in 1:W2
                s = 0
                for dj in -1:1, di in -1:1
                    s += mask2[clamp(j + dj, 1, H2), clamp(i + di, 1, W2)]
                end
                mc[j, i] = s
            end
            mask2o = mask2 .& (mc .>= 3)
            mask2o = mask2o .| ((.!mask2) .& (mc .>= 6))
            # --- external cartography export (GMT & friends) ---------------
            # GeoASCII xyz: x=lon y=lat z=value. disp = the smoothed
            # detrended LOS field shown in the BMP suite [cm]; dem = GLO-90
            # on the same cells [m] for hillshade backgrounds.
            if topo && annS !== annP
                lat2g = regrid_mean(lat_m, azf, rgf)
                lon2g = regrid_mean(lon_m, azf, rgf)
                dem2g = regrid_mean(DEM_m, azf, rgf)
                open(joinpath(outdir, "disp_" * label * ".xyz"), "w") do io
                    for jj in 1:H2, ii in 1:W2
                        if mask2[jj, ii] && !isnan(dres2[jj, ii])
                            @printf(io, "%.6f %.6f %.3f\n",
                                lon2g[jj, ii], lat2g[jj, ii], dres2[jj, ii])
                        end
                    end
                end
                open(joinpath(outdir, "dem_" * label * ".xyz"), "w") do io
                    for jj in 1:H2, ii in 1:W2
                        if !isnan(dem2g[jj, ii])
                            @printf(io, "%.6f %.6f %.1f\n",
                                lon2g[jj, ii], lat2g[jj, ii], dem2g[jj, ii])
                        end
                    end
                end
                @printf("  exported disp_%s.xyz (%d pts) + dem_%s.xyz for GMT\n",
                    label, count(mask2 .& (.!isnan.(dres2))), label)
                # --- split sidecar: full cell grid + LOS geometry ----------
                # raw split needs everything on one rectangular grid with
                # known dimensions plus the toward-satellite ENU unit
                # vector at the block centre (from the true orbit).
                tc = s1_line_az_time(annP,
                        round(Int, l0 + (H ÷ 2 - 0.5) * mla - 0.5))
                psat = s1_orbit_pos(annP, tc)
                vsat = s1_orbit_vel(annP, tc)
                jc = clamp(H ÷ 2 + 1, 1, H); ic = clamp(W ÷ 2 + 1, 1, W)
                tge = geodetic_ecef(lat_m[jc, ic], lon_m[jc, ic],
                                    DEM_m[jc, ic])
                los = psat .- tge
                los ./= norm(los)
                la_c = lat_m[jc, ic] * pi / 180
                lo_c = lon_m[jc, ic] * pi / 180
                e_e = (-sin(lo_c), cos(lo_c), 0.0)
                e_n = (-sin(la_c) * cos(lo_c), -sin(la_c) * sin(lo_c),
                       cos(la_c))
                e_u = (cos(la_c) * cos(lo_c), cos(la_c) * sin(lo_c),
                       sin(la_c))
                ux = los[1] * e_e[1] + los[2] * e_e[2] + los[3] * e_e[3]
                uy = los[1] * e_n[1] + los[2] * e_n[2] + los[3] * e_n[3]
                uz = los[1] * e_u[1] + los[2] * e_u[2] + los[3] * e_u[3]
                # flight azimuth from the orbit velocity
                vx = vsat[1] * e_e[1] + vsat[2] * e_e[2] + vsat[3] * e_e[3]
                vy = vsat[1] * e_n[1] + vsat[2] * e_n[2] + vsat[3] * e_n[3]
                hdg = mod(atand(vx, vy), 360.0)
                open(joinpath(outdir, "grid_" * label * ".txt"), "w") do io
                    @printf(io, "# %d %d %.6f %.6f %.6f %.2f\n",
                            H2, W2, ux, uy, uz, hdg)
                    for jj in 1:H2, ii in 1:W2
                        @printf(io, "%.6f %.6f %.3f %.1f\n",
                                lon2g[jj, ii], lat2g[jj, ii],
                                dres2[jj, ii], dem2g[jj, ii])
                    end
                end
                @printf("  exported grid_%s.txt (%dx%d cells, LOS enu %+.3f %+.3f %+.3f, heading %.1f deg)\n",
                        label, H2, W2, ux, uy, uz, hdg)
                if !sim
                    write_los_map(joinpath(outdir, "map_" * label * ".bmp"),
                                  lat2g, lon2g, dres2, mask2, dem2g;
                                  title = mtitle,
                                  vlat = vlat, vlon = vlon, vname = vname)
                end
            end
            write_bmp(joinpath(outdir,
                "interferogram_overlay_" * label * ".bmp"),
                upscale_img(render_overlay_img(pha_g2, mask2o, amp2b, hs2b;
                                                dem = dem2),
                            zoom, zoom; bilinear = true))
            s2 = defo_scale(dres2, mask2)
            write_bmp(joinpath(outdir, "unwrapped_defo_" * label * ".bmp"),
                upscale_img(render_defo_img(dres2, mask2, s2), zoom, zoom;
                            bilinear = true))
            @printf("  wrote square-cell display suite %dx%d cells of %.0f x %.0f m (zoom x%d, coh≥%.2f, %d cells, defo ±%.1f cm)\n",
                size(coh2)[1], size(coh2)[2], daz * azf, drg * rgf, zoom,
                max(cohthr, 0.35), count(mask2), s2)
            nice = true
        end
    end
    if !nice
        write_bmp(joinpath(outdir, "coherence_" * label * ".bmp"), imgc)
        write_bmp(joinpath(outdir, "unwrapped_" * label * ".bmp"), imgd)
        write_bmp(joinpath(outdir, "interferogram_" * label * ".bmp"), img)
        @printf("  wrote interferogram/coherence/unwrapped BMPs to %s\n", outdir)
    end

    # --- deformation summary (real pairs, after topo removal) -----------------
    # Robust estimate of the deformation signal: least-squares 2-D quadratic
    # (tilt + curvature: orbit/flat-earth residue + unwrapping offset) over
    # the coherent pixels; the residual range is deformation + atmosphere.
    if annS !== annP && !sim
        # land only (mean amplitude in DN units: ocean ~5, land 20-60):
        # same convention as the validated analyze_ifg methodology
        idx = [k for k in eachindex(dcm)
               if !isnan(dcm[k]) && amp[k] > 30.0]
        if length(idx) > 200
            Hm, Wm = size(dcm)
            X = zeros(length(idx), 6)
            y = zeros(length(idx))
            for (n, k) in enumerate(idx)
                r, c = Tuple(CartesianIndices(dcm)[k])
                x = (c - 0.5) / Wm - 0.5; z = (r - 0.5) / Hm - 0.5
                X[n, :] .= (1.0, x, z, x * x, x * z, z * z)
                y[n] = dcm[k]
            end
            coef = (X' * X) \ (X' * y)
            resid = y .- X * coef
            srt = sort!(resid)
            q = max(1, length(srt) ÷ 1000)
            rmin, rmax = extrema(y)
            @printf("  LOS deformation: raw %+6.2f .. %+6.2f cm; quadratic tilt removed %+6.2f .. %+6.2f cm  (1-99 pct %+6.2f .. %+6.2f, rms %.2f)\n",
                rmin, rmax, srt[q], srt[end - q + 1],
                srt[max(1, length(srt) ÷ 100)], srt[min(end, 99 * length(srt) ÷ 100)],
                sqrt(sum(abs2, resid) / length(resid)))
        end
    end

    # --- validation report ---------------------------------------------------
    if sim
        # compare recovered dcm against the input Gaussian on the coherent mask
        # (unwrapping has an arbitrary constant -> remove the best-fit offset)
        dif = Float64[]
        for jj in 1:H, ii in 1:W
            (mask[jj, ii] && !isnan(dcm[jj, ii])) || continue
            i = (jj - 0.5) * mla; j = (ii - 0.5) * mlr
            d = simmax_cm * exp(-((i - NL / 2)^2 / (NL / 6.0)^2 +
                                  (j - NP / 2)^2 / (NP / 6.0)^2))
            push!(dif, dcm[jj, ii] - d)
        end
        isempty(dif) && error("sim validation: no valid pixels")
        off = sum(dif) / length(dif)
        rms = sqrt(sum(abs2, dif .- off) / length(dif))
        @printf("  SIM validation: %d px, unwrapping constant %+.3f cm,\n",
            length(dif), off)
        @printf("    dcm RMS vs input (offset removed) %.4f cm  (input max %.2f cm)%s\n",
            rms, simmax_cm, rms < 0.05 ? "  (OK)" : "  (CHECK)")
    end
    return (pha = pha, coh = coh, unw = unw, dcm = dcm, stats = stats)
end

# ----------------------------------------------------------------------------
# 7h. GCP geolocation, real DEM (Open-Meteo), geocoding, topographic phase
# ----------------------------------------------------------------------------

"""Unique GCP line/pixel axes + lat/lon [deg] / height grids from the
annotation. (Sentinel-1 geolocation grid lat/lon are already DEGREES,
written in scientific notation, e.g. 6.23126e+01 = 62.3126 deg.)"""
function s1_gcp_grid(ann::S1Annotation)
    ul = sort(unique([g.line for g in ann.gcps]))
    up = sort(unique([g.pixel for g in ann.gcps]))
    nl, np = length(ul), length(up)
    li = Dict(v => i for (i, v) in enumerate(ul))
    pj = Dict(v => j for (j, v) in enumerate(up))
    LAT = fill(NaN, nl, np); LON = fill(NaN, nl, np); HGT = fill(NaN, nl, np)
    for g in ann.gcps
        LAT[li[g.line], pj[g.pixel]] = g.lat
        LON[li[g.line], pj[g.pixel]] = g.lon
        HGT[li[g.line], pj[g.pixel]] = g.hgt
    end
    return ul, up, LAT, LON, HGT
end

"""Bilinear (line, pixel) -> (lat, lon, height) via the GCP grid.
Fractional line/pixel allowed; clamped to the GCP footprint."""
function s1_geolocate(ul::Vector{Int}, up::Vector{Int},
                      LAT::Matrix{Float64}, LON::Matrix{Float64},
                      HGT::Matrix{Float64}, line::Float64, pixel::Float64)
    nl, np = size(LAT)
    line = clamp(line, Float64(ul[1]), Float64(ul[end]))
    pixel = clamp(pixel, Float64(up[1]), Float64(up[end]))
    i = clamp(searchsortedlast(ul, Int(floor(line))), 1, nl - 1)
    j = clamp(searchsortedlast(up, Int(floor(pixel))), 1, np - 1)
    u = (line - ul[i]) / (ul[i + 1] - ul[i])
    v = (pixel - up[j]) / (up[j + 1] - up[j])
    bil(M) = (1 - u) * (1 - v) * M[i, j] + u * (1 - v) * M[i + 1, j] +
             (1 - u) * v * M[i, j + 1] + u * v * M[i + 1, j + 1]
    return bil(LAT), bil(LON), bil(HGT)
end

"""Bilinear sample of M at fractional 1-based grid indices (clamped)."""
function bil_grid(M::Matrix{Float64}, x::Float64, y::Float64)
    H, W = size(M)
    x = clamp(x, 1.0, Float64(H)); y = clamp(y, 1.0, Float64(W))
    i = clamp(Int(floor(x)), 1, H - 1); j = clamp(Int(floor(y)), 1, W - 1)
    u = x - i; v = y - j
    return (1 - u) * (1 - v) * M[i, j] + u * (1 - v) * M[i + 1, j] +
           (1 - u) * v * M[i, j + 1] + u * v * M[i + 1, j + 1]
end

"""Geolocate the multilook grid, fetch the REAL DEM on a strided subsample
of it (Open-Meteo / Copernicus GLO-90, batched 100 pts/request, cache in
outdir that is RESUMABLE across interrupted runs), and bilinear-fill it
onto the full ML grid. Returns (lat, lon, hgt_ann, DEM), H x W matrices."""
function dem_for_ml_block(ann::S1Annotation, l0::Int, p0::Int,
                          mla::Int, mlr::Int, H::Int, W::Int;
                          stride::Int = 4, outdir::String = "insar_out",
                          label::String = "b", gap::Float64 = 0.3)
    stride = max(1, min(stride, H - 1, W - 1))
    ul, up, LAT, LON, HGT = s1_gcp_grid(ann)
    lat = Matrix{Float64}(undef, H, W)
    lon = Matrix{Float64}(undef, H, W)
    hgt = Matrix{Float64}(undef, H, W)
    for jj in 1:H, ii in 1:W
        fl = l0 + (jj - 0.5) * mla - 0.5
        fp = p0 + (ii - 0.5) * mlr - 0.5
        _, _, hgt[jj, ii] = s1_geolocate(ul, up, LAT, LON, HGT, fl, fp)
        lat[jj, ii], lon[jj, ii] =
            s1_geo_of_pixel_rd(ann, fl, fp, hgt[jj, ii])
    end
    is_ = collect(1:stride:H); js = collect(1:stride:W)
    Hs, Ws = length(is_), length(js)
    # cache in a durable location if available (survives environment resets)
    cachedir = outdir
    kd = "/home/user/knowledge/insar-sat-jl/data"
    isdir(kd) && (cachedir = kd)
    cache = joinpath(cachedir,
        "dem_$(label)_rd_s$(stride)_l$(l0)_p$(p0).txt")
    n = Hs * Ws
    la = Float64[]; lo = Float64[]
    for a in 1:Hs, b in 1:Ws
        push!(la, lat[is_[a], js[b]]); push!(lo, lon[is_[a], js[b]])
    end
    DEMs = fill(NaN, Hs, Ws)
    # linear index -> (a, b): row-major over the strided grid
    ab(i::Int) = ((i - 1) ÷ Ws + 1, (i - 1) % Ws + 1)
    done = 0
    if isfile(cache)
        try
            lines = readlines(cache)
            startswith(lines[1], "#") || error("no header")
            hd = split(strip(lines[1][2:end]))
            (parse(Int, hd[1]) == l0 && parse(Int, hd[2]) == p0 &&
             parse(Int, hd[3]) == mla && parse(Int, hd[4]) == mlr &&
             parse(Int, hd[5]) == stride) || error("params changed")
            done = min(length(lines) - 1, n)
            for i in 1:done
                f = split(lines[i + 1])
                length(f) >= 3 || error("bad line")
                a, b = ab(i)
                DEMs[a, b] = parse(Float64, f[3])
            end
        catch
            done = 0                          # corrupt/mismatched -> refetch
        end
    end
    if done > 0
        @printf("  DEM: cache %d / %d pts (%s)\n", done, n, cache)
    end
    if done < n
        @printf("  DEM: fetching %d pts from Open-Meteo (Copernicus GLO-90)\n",
            n - done)
        io = open(cache, done > 0 ? "a" : "w")
        if done == 0
            println(io, "# ", l0, " ", p0, " ", mla, " ", mlr, " ", stride)
        end
        k = done + 1
        while k <= n
            ke = min(k + 99, n)
            url = "https://api.open-meteo.com/v1/elevation?latitude=" *
                  join(string.(la[k:ke]), ",") * "&longitude=" *
                  join(string.(lo[k:ke]), ",")
            vals = fetch_chunk(url, outdir)
            length(vals) == ke - k + 1 ||
                error("API returned ", length(vals), " of ", ke - k + 1)
            for i in k:ke
                a, b = ab(i)
                DEMs[a, b] = vals[i - k + 1]
                @printf(io, "%.6f %.6f %.1f\n", la[i], lo[i], vals[i - k + 1])
            end
            flush(io)
            k = ke + 1
            if k <= n
                @printf("  DEM fetch %3d %%\r", min(100, round(Int, 100ke / n)))
                flush(stdout)
                sleep(gap)
            end
        end
        close(io)
        println("  DEM fetch 100 %")
    end
    DEM = Matrix{Float64}(undef, H, W)
    for jj in 1:H, ii in 1:W
        DEM[jj, ii] = bil_grid(DEMs, 1 + (jj - 1) / stride,
                                    1 + (ii - 1) / stride)
    end
    return lat, lon, hgt, DEM
end

"""Multilooked amplitude only (ml_a x ml_r boxcar RMS)."""
function amplitude_ml(pblk::Matrix{ComplexF32}, ml_a::Int, ml_r::Int)
    NL, NP = size(pblk)
    H = NL ÷ ml_a; W = NP ÷ ml_r
    amp = zeros(Float64, H, W)
    for jj in 1:H, ii in 1:W
        s = 0.0
        for a in 1:ml_a, r in 1:ml_r
            s += abs2(pblk[(jj - 1) * ml_a + a, (ii - 1) * ml_r + r])
        end
        amp[jj, ii] = sqrt(s / (ml_a * ml_r))
    end
    return amp
end

"""GLO-90 height -> RGB (sea blues, green->brown->white land)."""
function dem_shade(h::Float64, hmax::Float64)
    isnan(h) && return (UInt8(0), UInt8(0), UInt8(0))
    if h <= 0.0
        t = clamp(-h / 300.0, 0.0, 1.0)
        return (UInt8(15), UInt8(round(30 + 40t)), UInt8(round(70 + 60t)))
    end
    t = clamp(h / hmax, 0.0, 1.0)
    if t < 0.5
        u = t / 0.5
        r = 40 + 120u; g = 110 + 10u; b = 40 - 20u
    else
        u = (t - 0.5) / 0.5
        r = 160 + 95u; g = 120 + 135u; b = 20 + 235u
    end
    return (round(UInt8, clamp(r, 0, 255)), round(UInt8, clamp(g, 0, 255)),
            round(UInt8, clamp(b, 0, 255)))
end

"""raw dem: geolocate the block via the annotation GCPs, fetch the REAL
DEM (Open-Meteo / Copernicus GLO-90), geocode amplitude + DEM to lat/lon
grids, and CHECK THE COASTLINE: land must be bright in amplitude, ocean
dark. This validates the whole geolocation + DEM chain on real data."""
function raw_dem(ann::S1Annotation, tiffpath::String, t::SlcTiff;
                 burst::Int = -1, l0::Int = -1, l1::Int = -1,
                 p0::Int = 14000, p1::Int = 18000,
                 mla::Int = 8, mlr::Int = 4, stride::Int = 4,
                 outdir::String = "insar_out")
    if burst < 0; burst = length(ann.bursts) - 1; end
    k = clamp(burst, 0, length(ann.bursts) - 1)
    if l0 < 0; l0 = k * ann.lines_per_burst; end
    if l1 < 0; l1 = l0 + ann.lines_per_burst - 1; end
    NL = l1 - l0 + 1; NP = p1 - p0 + 1
    @printf("dem block: lines %d..%d (burst %d), pixels %d..%d\n",
        l0, l1, k, p0, p1)
    mkpath(outdir)
    label = "b$(k + 1)"
    pblk = read_slc_block(tiffpath, t, l0, p0, NL, NP)
    H = NL ÷ mla; W = NP ÷ mlr
    amp = amplitude_ml(pblk, mla, mlr)
    lat, lon, hgt, DEM = dem_for_ml_block(ann, l0, p0, mla, mlr, H, W;
        stride = stride, outdir = outdir, label = label)

    # --- coastline validation ---------------------------------------------
    land = DEM .> 0.5
    ocean = DEM .<= 0.5
    nl_ = count(land); no_ = count(ocean)
    (nl_ >= 500 && no_ >= 500) ||
        error("block is $(100 * nl_ / (nl_ + no_)) % land per DEM -- pick a block straddling the coastline for the check")
    ampL = sum(amp[land]) / nl_
    ampO = sum(amp[ocean]) / no_
    mu = (ampL * nl_ + ampO * no_) / (nl_ + no_)
    sd = sqrt((sum(abs2, amp .- mu)) / (nl_ + no_))
    # point-biserial correlation of amplitude with the DEM land mask
    p = nl_ / (nl_ + no_)
    rpb = (ampL - ampO) / max(sd, 1e-9) * sqrt(p * (1 - p))
    pred_land = amp .> (ampL + ampO) / 2
    acc = count(pred_land .== land) / (nl_ + no_)
    dh = sum(abs.(hgt .- DEM)) / length(DEM)
    @printf("  amp mean: land %.0f  ocean %.0f  ratio %.1f\n",
        ampL, ampO, ampL / ampO)
    @printf("  coastline agreement (amp>thr vs DEM>0): %.1f %%\n", 100acc)
    @printf("  point-biserial corr(amp, land): %+.3f\n", rpb)
    @printf("  annotation height vs GLO-90: mean |dh| %.1f m\n", dh)
    @printf("  -> geolocation+DEM chain %s\n",
        (acc > 0.90 && rpb > 0.5) ? "VALIDATED" : "CHECK")

    # --- geocode: splat ML cells onto a regular lat/lon grid ---------------
    latmin, latmax = extrema(lat)
    lonmin, lonmax = extrema(lon)
    latmid = 0.5 * (latmin + latmax)
    nlon = clamp(round(Int, (lonmax - lonmin) / 0.001), 60, 1600)
    dlon = (lonmax - lonmin) / nlon
    dlat = dlon * cosd(latmid)
    nlat = clamp(round(Int, (latmax - latmin) / dlat), 20, 1600)
    @printf("  geocode grid: %d x %d  (%.4f,%.4f)-(%.4f,%.4f)\n",
        nlat, nlon, latmin, lonmin, latmax, lonmax)
    GS = zeros(nlat, nlon); GDS = zeros(nlat, nlon)
    CNT = zeros(Int, nlat, nlon)
    for jj in 1:H, ii in 1:W
        r = clamp(1 + round(Int, (latmax - lat[jj, ii]) / dlat), 1, nlat)
        c = clamp(1 + round(Int, (lon[jj, ii] - lonmin) / dlon), 1, nlon)
        GS[r, c] += amp[jj, ii]; GDS[r, c] += DEM[jj, ii]; CNT[r, c] += 1
    end
    GA = fill(NaN, nlat, nlon); GD = fill(NaN, nlat, nlon)
    for r in 1:nlat, c in 1:nlon
        if CNT[r, c] > 0
            GA[r, c] = GS[r, c] / CNT[r, c]
            GD[r, c] = GDS[r, c] / CNT[r, c]
        end
    end
    # geocoded amplitude (sqrt stretch at the 99th percentile)
    va = sort([GA[r, c] for r in 1:nlat, c in 1:nlon if !isnan(GA[r, c])])
    a99 = va[min(end, max(1, round(Int, 0.99length(va))))]
    imgA = Array{UInt8}(undef, nlat, nlon, 3)
    for r in 1:nlat, c in 1:nlon
        if isnan(GA[r, c])
            imgA[r, c, 1] = 10; imgA[r, c, 2] = 10; imgA[r, c, 3] = 40
        else
            g = round(UInt8, clamp(255 * sqrt(clamp(GA[r, c] / a99, 0, 1)), 0, 255))
            imgA[r, c, 1] = g; imgA[r, c, 2] = g; imgA[r, c, 3] = g
        end
    end
    # coastline (DEM sign change w.r.t. 4-neighbourhood) drawn in red
    for r in 2:nlat-1, c in 2:nlon-1
        (isnan(GD[r, c]) || isnan(GA[r, c])) && continue
        landc = GD[r, c] > 0
        coast = false
        for (dr, dc) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            nb = GD[r + dr, c + dc]
            isnan(nb) && continue
            ((nb > 0) != landc) && (coast = true)
        end
        if coast
            imgA[r, c, 1] = 0xff; imgA[r, c, 2] = 0x20; imgA[r, c, 3] = 0x20
        end
    end
    write_bmp(joinpath(outdir, "geoamp_coast_$label.bmp"), imgA)
    # geocoded DEM with the same footprint
    vh = sort([GD[r, c] for r in 1:nlat, c in 1:nlon
               if !isnan(GD[r, c]) && GD[r, c] > 0])
    hmax = max(50.0, vh[min(end, max(1, round(Int, 0.98length(vh))))])
    imgD = Array{UInt8}(undef, nlat, nlon, 3)
    for r in 1:nlat, c in 1:nlon
        rr, gg, bb = dem_shade(GD[r, c], hmax)
        imgD[r, c, 1] = rr; imgD[r, c, 2] = gg; imgD[r, c, 3] = bb
    end
    write_bmp(joinpath(outdir, "geodem_$label.bmp"), imgD)
    @printf("  wrote geoamp_coast/geodem BMPs to %s (land hmax %.0f m)\n",
        outdir, hmax)
    return (amp = amp, lat = lat, lon = lon, DEM = DEM, acc = acc, rpb = rpb)
end

# ---- orbit interpolation + topographic phase ----------------------------

const WGS_A  = 6378137.0            # WGS84 semi-major axis [m]
const WGS_E2 = 0.00669437999014133   # first eccentricity squared

"""Hermite (pos+vel) interpolation of the annotation orbit state vectors at
azimuth time t. Returns ECEF position [m]."""
function s1_orbit_pos(ann::S1Annotation, t::Float64)
    ot = ann.orbit.t
    n = length(ot)
    n >= 2 || error("annotation has no orbit state vectors")
    i = 1
    while i < n - 1 && ot[i + 1] <= t; i += 1; end
    t0, t1 = ot[i], ot[i + 1]
    dt = t1 - t0
    s = clamp((t - t0) / dt, 0.0, 1.0)
    p0 = ann.orbit.pos[:, i]; p1 = ann.orbit.pos[:, i + 1]
    v0 = ann.orbit.vel[:, i]; v1 = ann.orbit.vel[:, i + 1]
    h00 = 2s^3 - 3s^2 + 1;  h10 = s^3 - 2s^2 + s
    h01 = -2s^3 + 3s^2;     h11 = s^3 - s^2
    return h00 .* p0 .+ (dt .* h10) .* v0 .+ h01 .* p1 .+ (dt .* h11) .* v1
end

"""WGS84 geodetic (lat/lon deg, height m) -> ECEF [m]."""
function geodetic_ecef(lat_deg::Float64, lon_deg::Float64, h::Float64)
    la = lat_deg * pi / 180; lo = lon_deg * pi / 180
    N = WGS_A / sqrt(1 - WGS_E2 * sin(la)^2)
    return [(N + h) * cos(la) * cos(lo), (N + h) * cos(la) * sin(lo),
            (N * (1 - WGS_E2) + h) * sin(la)]
end

"""Slant-range difference secondary minus primary [cm] at zero-Doppler
time t for ground point (lat, lon, h). Topographic + flat-earth path at
once (true orbit state vectors; no flat-earth approximation).
dt_pair: time offset between the secondary's and the primary's sensing of
the same along-track position (burst-anchor difference); the secondary
orbit is a different pass (days later) and MUST be evaluated at t+dt_pair,
otherwise the Hermite interpolation extrapolates into the void."""
function s1_range_diff_cm(annP::S1Annotation, annS::S1Annotation,
                          t::Float64, lat::Float64, lon::Float64, h::Float64;
                          dt_pair::Float64 = 0.0)
    g = geodetic_ecef(lat, lon, h)
    d1 = s1_orbit_pos(annP, t) .- g
    d2 = s1_orbit_pos(annS, t + dt_pair) .- g
    r1 = sqrt(sum(abs2, d1)); r2 = sqrt(sum(abs2, d2))
    return (r2 - r1) * 100.0
end

# ----------------------------------------------------------------------------
# 7b. S1 PAIR DRIVER: catalogue search (CDSE OData, no auth), ASF range-ZIP
#     download (Earthdata bearer token), automatic block selection and a
#     full pair interferogram run.  stdlib only: LibCURL + hand-written
#     CRC32 / MD5 / raw-DEFLATE (32 KB ring window).
# ----------------------------------------------------------------------------

const HAVE_LIBCURL = try
    Base.eval(Main, :(using LibCURL))
    true
catch
    false
end

# ---- CRC32 (RFC 1952), streaming over chunks -------------------------------
const CRC32_TBL = let t = Vector{UInt32}(undef, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 0x00000001) != 0 ? (c >> 1) ⊻ 0xedb88320 : c >> 1
        end
        t[n + 1] = c
    end
    t
end

function crc32_update!(c::UInt32, data::Vector{UInt8})::UInt32
    @inbounds for b in data
        c = (c >> 8) ⊻ CRC32_TBL[(xor(c & 0xff, b)) + 1]
    end
    return c
end
crc32_bytes(data::Vector{UInt8}) = xor(crc32_update!(0xffffffff, data), 0xffffffff)

function crc32_file(path::String)::UInt32
    c = 0xffffffff
    open(path, "r") do io
        while !eof(io)
            c = crc32_update!(c, read(io, 8 * 2^20))
        end
    end
    return xor(c, 0xffffffff)
end

# ---- MD5 (RFC 1321), streaming ---------------------------------------------
const MD5_S = [7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
               5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
               4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
               6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21]
const MD5_K = let ks = Vector{UInt32}(undef, 64)
    for i in 0:63
        ks[i + 1] = UInt32(trunc(abs(sin(i + 1)) * 4294967296))
    end
    ks
end

mutable struct MD5St
    a::UInt32; b::UInt32; c::UInt32; d::UInt32
    buf::Vector{UInt8}
    len::UInt64
end
md5_init() = MD5St(0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, UInt8[], 0)

function md5_block!(st::MD5St, block::SubArray{UInt8})
    M = reinterpret(UInt32, block)
    A = st.a; B = st.b; C = st.c; D = st.d
    for i in 0:63
        if i < 16
            F = (B & C) | (~B & D); g = i
        elseif i < 32
            F = (D & B) | (~D & C); g = mod(5i + 1, 16)
        elseif i < 48
            F = B ⊻ C ⊻ D; g = mod(3i + 5, 16)
        else
            F = C ⊻ (B | ~D); g = mod(7i, 16)
        end
        F = F + A + MD5_K[i + 1] + M[g + 1]
        A = D; D = C; C = B
        B = B + ((F << MD5_S[i + 1]) | (F >> (32 - MD5_S[i + 1])))
    end
    st.a += A; st.b += B; st.c += C; st.d += D
    return st
end

function md5_update!(st::MD5St, data::Vector{UInt8})
    st.len += length(data)
    data = vcat(st.buf, data)
    st.buf = UInt8[]
    full = length(data) - mod(length(data), 64)
    for off in 1:64:full
        md5_block!(st, view(data, off:off + 63))
    end
    full < length(data) && (st.buf = data[full + 1:end])
    return st
end

function md5_hex(st::MD5St)
    st2 = MD5St(st.a, st.b, st.c, st.d, copy(st.buf), st.len)
    push!(st2.buf, 0x80)
    while mod(length(st2.buf), 64) != 56
        push!(st2.buf, 0x00)
    end
    append!(st2.buf, reinterpret(UInt8, [st2.len * 8]))
    md5_block!(st2, view(st2.buf, 1:64))
    out = UInt8[]
    for w in (st2.a, st2.b, st2.c, st2.d)
        append!(out, reinterpret(UInt8, [w]))
    end
    return join(string.(out, base = 16, pad = 2))
end
md5_hex_msg(msg::Vector{UInt8}) = md5_hex(md5_update!(md5_init(), msg))

function md5_file(path::String)::String
    st = md5_init()
    open(path, "r") do io
        while !eof(io)
            md5_update!(st, read(io, 8 * 2^20))
        end
    end
    return md5_hex(st)
end

# ---- raw DEFLATE (RFC 1951) with a 32 KB ring window, output to a file ------
# (RAM-flat: constant window, no full output buffer; verified byte-identical
#  against zlib on dynamic / stored / fixed test vectors, ~87 MB/s output)

mutable struct DflBitR
    d::Vector{UInt8}
    bp::Int
    bb::UInt32
    bc::Int
end
DflBitR(d) = DflBitR(d, 1, 0x0, 0)

@inline function dfl_bit!(br::DflBitR)::Int
    if br.bc == 0
        br.bb = UInt32(br.d[br.bp]); br.bp += 1; br.bc = 8
    end
    v = Int(br.bb & 1)
    br.bb >>= 1
    br.bc -= 1
    return v
end

# LSB-first multi-bit read (DEFLATE extra bits: first bit read is the LSB)
@inline function dfl_bits!(br::DflBitR, n::Int)::Int
    v = 0
    @inbounds for i in 0:n-1
        v |= dfl_bit!(br) << i
    end
    return v
end

struct DflHuff
    cnt::Vector{Int}
    sym::Vector{Int}
end

function dfl_huff(lengths::Vector{Int})::DflHuff
    cnt = zeros(Int, 16)
    for l in lengths
        l > 0 && (cnt[l + 1] += 1)
    end
    offs = zeros(Int, 18)
    for l in 1:15
        offs[l + 1] = offs[l] + cnt[l]
    end
    sym = zeros(Int, sum(cnt))
    for (i, l) in enumerate(lengths)
        if l > 0
            offs[l + 1] += 1
            sym[offs[l + 1]] = i - 1
        end
    end
    return DflHuff(cnt, sym)
end

@inline function dfl_dec!(br::DflBitR, h::DflHuff)::Int
    code = 0; first = 0; index = 0
    @inbounds for l in 1:15
        code = (code << 1) | dfl_bit!(br)
        count = h.cnt[l + 1]
        d = code - first
        0 <= d < count && return h.sym[index + d + 1]
        index += count
        first = (first + count) << 1
    end
    error("inflate: invalid huffman code")
end

const DFL_LBASE = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,
                   99,115,131,163,195,227,258]
const DFL_LEXT  = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
const DFL_DBASE = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,
                   769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577]
const DFL_DEXT  = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,
                   12,12,13,13]
const DFL_CLORD = UInt8[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]

mutable struct DflRing
    io::IOStream
    win::Vector{UInt8}
    wpos::Int
end
DflRing(path) = DflRing(open(path, "w"), zeros(UInt8, 32768), 0)

@inline function dflw_byte!(w::DflRing, b::UInt8)
    write(w.io, b)
    w.win[(w.wpos & 32767) + 1] = b
    w.wpos += 1
end
@inline function dflw_match!(w::DflRing, dist::Int, len::Int)
    dist > w.wpos && error("inflate: distance too far back")
    s = w.wpos - dist
    for k in 0:len-1
        dflw_byte!(w, w.win[((s + k) & 32767) + 1])
    end
end

"""Inflate a raw DEFLATE stream (RFC 1951, no zlib wrapper) into a file,
returning the number of blocks. RAM-flat via a 32 KB ring window."""
function inflate_to_file(raw::Vector{UInt8}, path::String)::Int
    w = DflRing(path)
    br = DflBitR(raw)
    nblocks = 0
    while true
        bfinal = dfl_bit!(br)
        btype = dfl_bits!(br, 2)
        if btype == 0                       # stored block
            br.bc = 0                       # byte-align
            len = dfl_bits!(br, 16)
            nlen = dfl_bits!(br, 16)
            (len != xor(nlen, 0xffff)) && error("inflate: stored len mismatch")
            for _ in 1:len
                dflw_byte!(w, UInt8(br.d[br.bp])); br.bp += 1
            end
        else
            if btype == 1                   # fixed huffman
                lit = dfl_huff(vcat(fill(8, 144), fill(9, 112),
                                     fill(7, 24), fill(8, 8)))
                dst = dfl_huff(fill(5, 30))
            else                            # dynamic huffman
                hlit = dfl_bits!(br, 5) + 257
                hdist = dfl_bits!(br, 5) + 1
                hclen = dfl_bits!(br, 4) + 4
                clens = zeros(Int, 19)
                for i in 1:hclen
                    clens[DFL_CLORD[i] + 1] = dfl_bits!(br, 3)
                end
                clh = dfl_huff(clens)
                lens = zeros(Int, hlit + hdist)
                i = 1
                while i <= hlit + hdist
                    s = dfl_dec!(br, clh)
                    if s < 16
                        lens[i] = s; i += 1
                    elseif s == 16
                        i == 1 && error("inflate: repeat with no previous")
                        nrep = 3 + dfl_bits!(br, 2); v = lens[i - 1]
                        for _ in 1:nrep; lens[i] = v; i += 1; end
                    elseif s == 17
                        nrep = 3 + dfl_bits!(br, 3)
                        for _ in 1:nrep; lens[i] = 0; i += 1; end
                    else
                        nrep = 11 + dfl_bits!(br, 7)
                        for _ in 1:nrep; lens[i] = 0; i += 1; end
                    end
                end
                lit = dfl_huff(lens[1:hlit])
                dst = dfl_huff(lens[hlit + 1:end])
            end
            while true
                s = dfl_dec!(br, lit)
                if s < 256
                    dflw_byte!(w, UInt8(s))
                elseif s == 256
                    break
                else
                    lcode = s - 257
                    len = DFL_LBASE[lcode + 1] + dfl_bits!(br, DFL_LEXT[lcode + 1])
                    ds = dfl_dec!(br, dst)
                    dist = DFL_DBASE[ds + 1] + dfl_bits!(br, DFL_DEXT[ds + 1])
                    dflw_match!(w, dist, len)
                end
            end
        end
        nblocks += 1
        bfinal == 1 && break
    end
    close(w.io)
    return nblocks
end

# ---- HTTP via LibCURL directly (redirects OFF: we hop 303 -> presigned) -----
function _http_write_cb(ptr::Ptr{Cvoid}, sz::Csize_t, nm::Csize_t,
                        ud::Ptr{Cvoid})::Csize_t
    n = Int(sz) * Int(nm)
    v = unsafe_pointer_to_objref(Ptr{Ref{Vector{UInt8}}}(ud))[]
    append!(v, unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(ptr), n))
    return Csize_t(n)
end
const HTTP_WCB = @cfunction(_http_write_cb, Csize_t,
                           (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}))

"""CA bundle path for CURLOPT_CAINFO, or "" to let the backend use its
default (Windows: the system store via Schannel). Explicit env override
wins; on Linux the distro bundle must be set explicitly because LibCURL's
default lookup can fail in stripped environments (CURLE 60)."""
function http_ca_bundle()::String
    for p in (get(ENV, "SSL_CERT_FILE", ""),
              get(ENV, "CURL_CA_BUNDLE", ""),
              "/etc/ssl/certs/ca-certificates.crt",
              "/etc/pki/tls/certs/ca-bundle.crt",
              "/etc/ssl/ca-bundle.pem",
              "/usr/local/etc/openssl/cert.pem",
              "/etc/openssl/certs/ca-bundle.crt",
              joinpath(Sys.BINDIR, "certs", "cacert.pem"),
              joinpath(Sys.BINDIR, "..", "etc", "certs", "cacert.pem"))
        (isempty(p) || !isfile(p)) && continue
        return String(p)
    end
    return ""
end

"""Single GET with manual redirect control. Returns (curl_rc, code, header,
body). `rng` = "a-b" byte range. CAINFO only set when a bundle file exists;
on Windows the bundled LibCURL uses the system certificate store instead
(setting a nonexistent path there fails with CURLE 77)."""
function http_get(url::String, hdrs::Vector{String} = String[];
                  rng::Union{Nothing,String} = nothing, timeout_s::Int = 120)
    HAVE_LIBCURL || error("LibCURL stdlib unavailable (need Julia >= 1.7)")
    curl = curl_easy_init()
    curl == C_NULL && error("curl_easy_init failed")
    body = Vector{UInt8}(); head = Vector{UInt8}()
    bref = Ref(body); href = Ref(head)
    hl = C_NULL
    for h in hdrs
        hl = curl_slist_append(hl, h)
    end
    rng !== nothing && (hl = curl_slist_append(hl, "Range: bytes=" * rng))
    curl_easy_setopt(curl, CURLOPT_URL, String(url))
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1)
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0)
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1)
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30)
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, Clong(timeout_s))
    ca = http_ca_bundle()
    !isempty(ca) && curl_easy_setopt(curl, CURLOPT_CAINFO, ca)
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "")
    hl != C_NULL && curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hl)
    # GC.@preserve pins bref/href: they are handed to C as raw pointers
    # (WRITEDATA/HEADERDATA). Without pinning the GC may move the Refs
    # mid-transfer (EXCEPTION_ACCESS_VIOLATION in the write callback,
    # seen on Windows/Julia 1.13; on 1.11/Linux the Ref was stack-
    # allocated by luck). Canonical ccall pattern: pointer_from_objref
    # must stay inside GC.@preserve together with the whole perform.
    GC.@preserve bref href begin
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, HTTP_WCB)
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, pointer_from_objref(bref))
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HTTP_WCB)
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, pointer_from_objref(href))
        rc = curl_easy_perform(curl)
    end
    code = Ref{Clong}()
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, code)
    curl_easy_cleanup(curl)
    return (Int(rc), Int(code[]), String(head), body)
end

"""`location:` header value of a response (empty string if absent)."""
function http_location(head::String)::String
    for line in split(head, "\r\n")
        if startswith(lowercase(line), "location:")
            return String(strip(line[11:end]))
        end
    end
    return ""
end

# ---- CDSE OData catalogue (no auth) -----------------------------------------
"""Extract the bracket-balanced JSON array after "coordinates":[ in txt."""
function json_coord_array(txt::AbstractString)::Union{String,Nothing}
    mp = findfirst("coordinates\":[", txt)
    mp === nothing && return nothing
    i = last(mp)
    depth = 0
    j = i
    while j <= lastindex(txt)
        c = txt[j]
        c == '[' && (depth += 1)
        if c == ']'
            depth -= 1
            depth == 0 && break
        end
        j += 1
    end
    return String(txt[i:j])
end

"""Point-in-polygon (ray cast) on a flat [lon,lat] pair list."""
function point_in_poly(coords::Vector{Tuple{Float64,Float64}},
                       lon::Float64, lat::Float64)::Bool
    n = length(coords)
    n >= 3 || return false
    inside = false
    j = n
    for i in 1:n
        xi = coords[i][1]; yi = coords[i][2]
        xj = coords[j][1]; yj = coords[j][2]
        if ((yi > lat) != (yj > lat)) &&
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

function json_pairs(s::AbstractString)::Vector{Tuple{Float64,Float64}}
    out = Tuple{Float64,Float64}[]
    for m in eachmatch(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", s)
        push!(out, (parse(Float64, m[1]), parse(Float64, m[2])))
    end
    return out
end

"""All SENTINEL1 IW SLC products of a UTC day window whose footprint contains
(lat, lon). Returns (name, t_start, t_stop, orbit_dir) tuples, name without
.SAFE; orbit_dir is "" when the catalogue omits the attribute."""
function odata_find_day(t0::String, t1::String, lat::Float64, lon::Float64)
    q = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products" *
        "?\$top=1000&\$filter=Collection/Name%20eq%20%27SENTINEL-1%27" *
        "%20and%20contains(Name,%27_IW_SLC__1SDV%27)" *
        "%20and%20ContentDate/Start%20ge%20" * t0 *
        "%20and%20ContentDate/Start%20le%20" * t1 *
        "&\$expand=Attributes"
    rc, code, _, body = http_get(q; timeout_s = 90)
    (rc != 0 || code != 200) && error("OData query failed (rc=$rc code=$code)")
    txt = String(body)
    out = Tuple{String,Float64,Float64,String}[]
    idxs = [m.offset for m in eachmatch(r"\"Name\":\"(S1[AB]_IW_SLC[^\"]+)\"", txt)]
    for k in eachindex(idxs)
        o = idxs[k]
        nm = match(r"\"Name\":\"(S1[AB]_IW_SLC[^\"]+)\"", txt[o:end])[1]
        endo = k < length(idxs) ? idxs[k + 1] : lastindex(txt)
        seg = txt[o:endo-1]
        cm = match(r"\"ContentDate\":\{\"Start\":\"([^\"]+)\",\"End\":\"([^\"]+)\"", seg)
        cm === nothing && continue
        arr = json_coord_array(seg)
        arr === nothing && continue
        coords = json_pairs(arr)
        (length(coords) >= 3 && point_in_poly(coords, lon, lat)) || continue
        odm = match(r"\"Name\":\"orbitDirection\",\"Value\":\"(ASCENDING|DESCENDING)\"", seg)
        od = odm === nothing ? "" : String(odm[1])
        endswith(nm, ".SAFE") && (nm = nm[1:end-5])
        push!(out, (String(nm), parse_s1_time(cm[1]), parse_s1_time(cm[2]), od))
    end
    return out
end

"""All catalogue candidates within +/- dtol days of the target date whose
footprint contains the point. With passdir != "" only that orbit direction
(ASCENDING/DESCENDING) is kept. Returns (name, t_start) tuples."""
function odata_candidates(date::String, lat::Float64, lon::Float64;
                          dtol::Int = 1, passdir::String = "")
    y, m, d = parse.(Int, split(date, "-"))
    base = Dates.Date(y, m, d)
    out = Vector{Tuple{String,Float64}}()
    seen = Set{String}()
    for off in 0:dtol
        for s in (off == 0 ? (0,) : (-off, off))
            day = Dates.format(base + Dates.Day(s), "yyyy-mm-dd")
            for (nm, t, _, od) in odata_find_day(day * "T00:00:00.000Z",
                                             day * "T23:59:59.999Z", lat, lon)
                (isempty(passdir) || od == passdir) || continue
                if !(nm in seen)
                    push!(seen, nm)
                    push!(out, (nm, t))
                end
            end
        end
    end
    return out
end

"""Circular distance of a time offset to a multiple of 24 h (seconds)."""
tod_dist(t::Float64) = abs(mod(t, 86400.0) - (mod(t, 86400.0) > 43200 ? 86400.0 : 0.0))

"""Pick the candidate whose start is closest to date@12:00 UTC. If tod_ref is
given, restrict to products whose start time-of-day matches tod_ref within
45 min (same track + pass => nearly identical UTC time)."""
function odata_pick(cands::Vector{Tuple{String,Float64}}, date::String;
                     tod_ref::Union{Nothing,Float64} = nothing)
    isempty(cands) && error("no catalogue candidates near ", date)
    y, m, d = parse.(Int, split(date, "-"))
    tref = Dates.datetime2unix(Dates.DateTime(y, m, d, 12, 0, 0))
    pool = cands
    if tod_ref !== nothing
        pool = [c for c in cands if tod_dist(c[2] - tod_ref) < 45 * 60]
        isempty(pool) && return ("", 0.0)
    end
    return pool[argmin([abs(c[2] - tref) for c in pool])]
end

# ---- ASF range ZIP download --------------------------------------------------
"""Bearer token: EARTHDATA_TOKEN env var, else earthdata.env next to the
script or in the CWD.  NEVER print the token."""
function earthdata_token()::String
    tok = get(ENV, "EARTHDATA_TOKEN", "")
    if isempty(tok)
        for p in (joinpath(dirname(abspath(PROGRAM_FILE)), "earthdata.env"),
                  "earthdata.env")
            if isfile(p)
                for line in eachline(p)
                    s = strip(line)
                    startswith(s, "EARTHDATA_TOKEN=") &&
                        return String(split(s, "="; limit = 2)[2])
                end
            end
        end
    end
    isempty(tok) && error("no Earthdata token: set EARTHDATA_TOKEN or " *
                           "provide earthdata.env (EARTHDATA_TOKEN=...)")
    return tok
end

"""Resolve an ASF product to (presigned_url, zip_size). Hop 1: bearer token
request -> 303 + Location (a GET-signed presigned URL that supports Range);
hop 2: probe Range 0-0 -> 206, total size from Content-Range."""
function asf_presigned(prodname::String, tok::String)::Tuple{String,Int}
    u1 = "https://sentinel1.asf.alaska.edu/SLC/SA/" * prodname * ".zip"
    rc, code, head, _ = http_get(u1, ["Authorization: Bearer " * tok];
                                rng = "0-0", timeout_s = 60)
    loc = http_location(head)
    lhost = isempty(loc) ? "" :
            (m = match(r"https?://[^/]+", loc); m === nothing ? "?" : m.match)
    prox = !isempty(get(ENV, "https_proxy", "")) ||
           !isempty(get(ENV, "HTTPS_PROXY", ""))
    diag = "toklen=$(length(tok))" * (prox ? ", proxy env SET" : "")
    # accept any redirect (303 on HTTP/2, 302 seen on Windows/HTTP/1.1);
    # the Location is the presigned CloudFront URL either way
    if code in (301, 302, 303, 307, 308) && !isempty(loc)
        occursin("urs.earthdata.nasa.gov", loc) &&
            error("ASF saw NO valid Authorization header (redirect to " *
                  "Earthdata login, $code; $diag). Either earthdata.env " *
                  "is corrupted (token must be ONE line, no spaces or " *
                  "backslashes; a valid edl-jwt is 690 chars) or something " *
                  "on this machine/network strips the header (AV, TLS-" *
                  "intercepting proxy).")
        rc2, code2, head2, _ = http_get(loc; rng = "0-0", timeout_s = 60)
        (rc2 == 0 && code2 == 206) ||
            error("presigned probe failed (rc=$rc2 code=$code2, host=$lhost, $diag)")
        m = match(r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", lowercase(head2))
        m === nothing && error("no Content-Range on presigned probe")
        return (loc, parse(Int, m[3]))
    elseif code == 200
        # direct hit (no redirect): size from Content-Length
        m = match(r"content-length:\s*(\d+)", lowercase(head))
        m === nothing && error("200 without Content-Length")
        return (u1, parse(Int, m[1]))
    end
    error("ASF resolve failed (rc=$rc code=$code" *
          (isempty(lhost) ? "" : ", location host: " * lhost) *
          "; $diag)")
end

struct ZipMember
    name::String
    method::Int
    crc::UInt32
    csz::Int
    usz::Int
    lho::Int
end

"""Read the ZIP end-of-central-directory over HTTP ranges and parse the
central directory. Returns the member list."""
function zip_central_dir(loc::String, zsize::Int)::Vector{ZipMember}
    t0 = max(0, zsize - 65536)
    rc, code, _, tail = http_get(loc; rng = string(t0) * "-" * string(zsize - 1),
                                 timeout_s = 90)
    (rc == 0 && code == 206) || error("zip tail fetch failed (rc=$rc code=$code)")
    eocd = 0
    for i in length(tail) - 21:-1:1
        if tail[i] == 0x50 && tail[i+1] == 0x4b && tail[i+2] == 0x05 && tail[i+3] == 0x06
            eocd = i
            break
        end
    end
    eocd == 0 && error("EOCD signature not found in zip tail")
    nmem = Int(reinterpret(UInt16, tail[eocd+10:eocd+11])[1])
    cdsize = Int(reinterpret(UInt32, tail[eocd+12:eocd+15])[1])
    cdoff = Int(reinterpret(UInt32, tail[eocd+16:eocd+19])[1])
    (nmem == 0 || cdsize == 0) && error("EOCD parse garbage (nmem=$nmem)")
    if cdoff >= t0
        cd = tail[cdoff - t0 + 1:cdoff - t0 + cdsize]
    else
        rc2, code2, _, cd = http_get(loc; rng = string(cdoff) * "-" *
                                                string(cdoff + cdsize - 1),
                                     timeout_s = 90)
        (rc2 == 0 && code2 == 206 && length(cd) == cdsize) ||
            error("central directory fetch failed")
    end
    members = ZipMember[]
    p = 1
    while p + 46 <= length(cd) + 1 && reinterpret(UInt32, cd[p:p+3])[1] == 0x02014b50
        meth = Int(reinterpret(UInt16, cd[p+10:p+11])[1])
        crc = reinterpret(UInt32, cd[p+16:p+19])[1]
        csz = Int(reinterpret(UInt32, cd[p+20:p+23])[1])
        usz = Int(reinterpret(UInt32, cd[p+24:p+27])[1])
        nlen = Int(reinterpret(UInt16, cd[p+28:p+29])[1])
        elen = Int(reinterpret(UInt16, cd[p+30:p+31])[1])
        clen = Int(reinterpret(UInt16, cd[p+32:p+33])[1])
        lho = Int(reinterpret(UInt32, cd[p+42:p+45])[1])
        nm = String(cd[p+46:p+45+nlen])
        push!(members, ZipMember(nm, meth, crc, csz, usz, lho))
        p += 46 + nlen + elen + clen
    end
    length(members) == nmem ||
        println("  zip: parsed ", length(members), " of ", nmem, " members (odd)")
    return members
end

"""Fetch one byte range with retries; re-resolves the presigned URL on
transport errors or 403 (expired signature)."""
function zip_range!(loc::Vector{String}, tok::String, prodname::String,
                    a::Int, b::Int; timeout_s::Int = 180)::Vector{UInt8}
    for attempt in 1:6
        rc, code, _, body = http_get(loc[1]; rng = string(a) * "-" * string(b),
                                     timeout_s = timeout_s)
        if rc == 0 && code == 206 && length(body) == b - a + 1
            return body
        end
        if code == 403 || rc != 0
            println("  http retry (rc=$rc code=$code attempt=$attempt): ",
                    "re-resolving presigned URL")
            loc[1], _ = asf_presigned(prodname, tok)
            sleep(2)
        else
            sleep(2)
        end
    end
    error("range fetch failed for bytes $a-$b")
end

"""Download and extract one ZIP member (local header walk + chunked range
fetch + raw-DEFLATE inflate or stored copy) with CRC32 verification."""
function zip_fetch_member(loc::Vector{String}, tok::String, prodname::String,
                          mem::ZipMember, outpath::String;
                          chunk::Int = 16 * 2^20, quiet::Bool = false)
    # local file header: fixed 30 B + name + extra
    lh = zip_range!(loc, tok, prodname, mem.lho, mem.lho + 29)
    (reinterpret(UInt32, lh[1:4])[1] == 0x04034b50) ||
        error("local header signature mismatch")
    nlen = Int(reinterpret(UInt16, lh[27:28])[1])
    elen = Int(reinterpret(UInt16, lh[29:30])[1])
    dstart = mem.lho + 30 + nlen + elen
    # chunked fetch into a persistent .part file: an interrupted transfer
    # (dropped link, killed process) resumes at the last 16 MB boundary
    # instead of restarting the whole member from zero
    part = outpath * ".part"
    done = 0
    if isfile(part)
        done = filesize(part)
        if done > mem.csz
            rm(part); done = 0
        elseif done < mem.csz && done % chunk != 0
            done = (done ÷ chunk) * chunk
            truncate(part, done)
        end
        done < mem.csz &&
            @printf("    resuming at %.1f / %.1f MB\n", done / 2^20, mem.csz / 2^20)
    end
    if done < mem.csz
        open(part, done == 0 ? "w" : "a") do io
            while done < mem.csz
                b = min(done + chunk, mem.csz) - 1
                write(io, zip_range!(loc, tok, prodname, dstart + done, dstart + b))
                flush(io)
                done = b + 1
                quiet || @printf("    %6.1f / %6.1f MB\r", done / 2^20, mem.csz / 2^20)
            end
        end
    end
    # mmap the member instead of read(): a 1 GB part in RAM plus the
    # inflate workspace OOMs small hosts; the mapping stays out of RSS
    comp = Mmap.mmap(part)
    if mem.method == 8
        inflate_to_file(comp, outpath)
    elseif mem.method == 0
        open(outpath, "w") do io
            write(io, comp)
        end
    else
        error("unsupported zip method ", mem.method)
    end
    rm(part)
    filesize(outpath) == mem.usz ||
        error("size mismatch after inflate ($(filesize(outpath)) vs $(mem.usz))")
    got = crc32_file(outpath)
    got == mem.crc ||
        error(@sprintf("CRC32 mismatch for %s: %08x != %08x",
                       basename(outpath), got, mem.crc))
end

"""Download iw-vv (default iw2) measurement TIFF + annotation XML of an ASF
S1 SLC product into datadir, reusing files already present (naming convention
asf<disc>_<iw>_vv.tiff / <disc>_<iw>_vv.xml). Verifies CRC32 from the zip
central directory and MD5 from manifest.safe. Returns (tiff, xml)."""
function s1_pair_download(prodname::String, datadir::String, tok::String;
                          iw::String = "iw2")
    iw in ("iw1", "iw2", "iw3") || error("iw must be iw1|iw2|iw3")
    disc = split(prodname, "_")[end]
    tiff = joinpath(datadir, "asf" * disc * "_" * iw * "_vv.tiff")
    xml = joinpath(datadir, disc * "_" * iw * "_vv.xml")
    if isfile(tiff) && isfile(xml)
        println("  ", disc, ": ", iw, "-vv pair already cached, reusing")
        return (tiff, xml)
    end
    mkpath(datadir)
    println("  ", disc, ": resolving ASF presigned URL...")
    loc, zsize = asf_presigned(prodname, tok)
    @printf("  %s: zip %.2f GB, reading central directory...\n", disc, zsize / 2^30)
    members = zip_central_dir(loc, zsize)
    locv = [loc]
    annm = nothing; tifm = nothing; manm = nothing
    for m in members
        occursin("/annotation/", m.name) && occursin(iw * "-slc-vv", m.name) &&
            (annm = m)
        occursin("/measurement/", m.name) && occursin(iw * "-slc-vv", m.name) &&
            (tifm = m)
        endswith(m.name, "manifest.safe") && (manm = m)
    end
    tifm === nothing && error("no ", iw, "-vv measurement member in zip")
    annm === nothing && error("no ", iw, "-vv annotation member in zip")
    @printf("  %s: annotation %d B, tiff %.1f MB (compressed)\n",
            disc, annm.usz, tifm.csz / 2^20)
    println("  ", disc, ": fetching annotation...")
    zip_fetch_member(locv, tok, prodname, annm, xml)
    md5map = Dict{String,String}()
    if manm !== nothing
        mp = xml * ".manifest.tmp"
        try
            zip_fetch_member(locv, tok, prodname, manm, mp; quiet = true)
            man = read(mp, String)
            # <dataObject ...><byteStream ...><fileLocation href="..."> +
            # <checksum checksumName="MD5">hex</checksum> (multiline blocks)
            for dm in eachmatch(
                    r"<dataObject [^>]*>.*?</dataObject>"s, man)
                seg = dm.match
                hm = match(r"checksumName=\"MD5\">\s*([0-9a-fA-F]{32})", seg)
                fm = match(r"href=\"([^\"]+)\"", seg)
                if hm !== nothing && fm !== nothing
                    md5map[basename(fm[1])] = lowercase(hm[1])
                end
            end
            rm(mp)
        catch e
            println("  warning: manifest.safe MD5 check skipped (", e, ")")
        end
    end
    @printf("  %s: fetching tiff (%.0f MB compressed, chunked)...\n",
            disc, tifm.csz / 2^20)
    t0 = time()
    zip_fetch_member(locv, tok, prodname, tifm, tiff)
    dt = time() - t0
    @printf("  %s: tiff done in %.0f s (%.1f MB/s net)\n",
            disc, dt, tifm.csz / 2^20 / max(dt, 0.1))
    md5got = md5_file(tiff)
    println("  ", disc, ": tiff MD5 ", md5got)
    md5ok = false
    for (fn, md5) in md5map
        if endswith(fn, ".tiff") && occursin(iw * "-slc-vv", fn)
            if lowercase(md5) == md5got
                md5ok = true
                println("  ", disc, ": MD5 matches manifest.safe ✓")
            else
                error("MD5 mismatch vs manifest.safe: $md5got != $md5")
            end
        end
    end
    md5ok || println("  ", disc, ": (no manifest MD5 entry found; CRC32 verified)")
    return (tiff, xml)
end

# ---- automatic block choice --------------------------------------------------
"""RD-solve the target in the primary: returns (burst k with the target most
central in its steering window, target line, target pixel)."""
function s1_target_burst(ann::S1Annotation, lat::Float64, lon::Float64;
                         h::Float64 = 50.0)
    t_rd, r = s1_rd_time(ann, lat, lon, h)
    LPB = ann.lines_per_burst
    wS = LPB * ann.azimuth_dt
    bestk = -1; bestd = Inf
    for k in 0:length(ann.bursts) - 1
        u = t_rd - ann.bursts[k + 1].t_az
        (u < 0 || u >= wS) && continue
        d = abs(u - wS / 2)
        d < bestd && (bestd = d; bestk = k)
    end
    bestk < 0 && error("target outside the primary burst windows")
    line = bestk * LPB + (t_rd - ann.bursts[bestk + 1].t_az) / ann.azimuth_dt
    px = (2 * r / S1_C - ann.slant_t0) * ann.range_sr
    return bestk, line, px
end

"""Choose the interferogram block around the target, clear of burst seams in
BOTH images (secondary seams sit at j*LPB - dline in primary lines). Returns
(burst, l0, l1, p0, p1); shrinks the azimuth size in steps of 100 (min 200)
if no seam-free placement exists."""
function pair_auto_block(annP::S1Annotation, annS::S1Annotation, dline::Float64,
                        k::Int, line_rd::Float64, px_rd::Float64,
                        NL0::Int, NP::Int)
    LPB = annP.lines_per_burst
    marg = 40
    NL = NL0
    l0 = 0
    while true
        lo_min = max(k * LPB + marg, 1)
        lo_max = min((k + 1) * LPB - marg - NL + 1, annP.nlines - NL + 1,
                     round(Int, annS.nlines - dline) - NL + 1)
        if lo_max < lo_min
            NL <= 200 && error("block does not fit burst $k (NL=$NL)")
            NL -= 100
            continue
        end
        # secondary seams inside this burst (primary coordinates)
        seams = Float64[]
        for j in 0:length(annS.bursts) - 1
            ss = j * annS.lines_per_burst - dline
            (ss > k * LPB && ss < (k + 1) * LPB) && push!(seams, ss)
        end
        lc = clamp(round(Int, line_rd - NL / 2), lo_min, lo_max)
        l0 = lc
        ok = true
        for _ in 1:10                          # re-stabilise over all seams
            ok = true
            for ss in seams
                if l0 - marg <= ss <= l0 + NL - 1 + marg
                    left_l0 = floor(Int, ss) - marg - NL + 1
                    right_l0 = ceil(Int, ss) + marg
                    left_ok = left_l0 >= lo_min
                    right_ok = right_l0 + NL - 1 <= lo_max
                    if left_ok && right_ok
                        l0 = abs(left_l0 - lc) <= abs(right_l0 - lc) ?
                             left_l0 : right_l0
                    elseif left_ok
                        l0 = left_l0
                    elseif right_ok
                        l0 = right_l0
                    else
                        ok = false
                        break
                    end
                end
            end
            ok && break
        end
        ok && break
        NL <= 200 && error("no seam-free block of NL=$NL in burst $k")
        NL -= 100
    end
    l1 = l0 + NL - 1
    # pixel window: stay inside the burst's valid samples
    fv = -1; lv = typemax(Int)
    for l in l0:l1
        li = l - k * LPB
        (li < 0 || li >= length(annP.bursts[k + 1].first_valid)) && continue
        a = annP.bursts[k + 1].first_valid[li + 1]
        b = annP.bursts[k + 1].last_valid[li + 1]
        a >= 0 && (fv = max(fv, a); lv = min(lv, b))
    end
    fv < 0 && (fv = 0; lv = annP.nsamples - 1)
    p0 = clamp(round(Int, px_rd - NP / 2), fv + marg, lv - marg - NP + 1)
    (p0 + NP - 1 > lv - marg || p0 < fv + marg) &&
        error("pixel window $NP too wide for valid samples [$fv, $lv]")
    return k, l0, l1, p0, p0 + NP - 1
end

# ---- the pair driver ---------------------------------------------------------
"""raw pair: catalogue search + ASF download + auto block + interferogram.
Options: --lat --lon --date --date2 [--dtol N] [--pass auto|desc|asc]
[--p1 NAME] [--p2 NAME] [--topo] [--nlines N] [--npix N] [--coh C] [--datadir DIR]
[--vname NAME] [--unwrap qg|wls]  (with --topo: native Vedurstofa-style LOS map BMP)"""
function raw_pair_driver(opts::Dict{String,String}, outdir::String)
    lat = num(opts, "lat", NaN)
    lon = num(opts, "lon", NaN)
    (isnan(lat) || isnan(lon)) && error("raw pair needs --lat and --lon")
    d1 = get(opts, "date", "")
    d2 = get(opts, "date2", "")
    (isempty(d1) || isempty(d2)) &&
        error("raw pair needs --date YYYY-MM-DD and --date2 YYYY-MM-DD")
    dtol = Int(num(opts, "dtol", 1.0))
    passwant = get(opts, "pass", "auto")
    NL0 = Int(num(opts, "nlines", 700.0))
    NP = Int(num(opts, "npix", 2200.0))
    datadir = get(opts, "datadir", "data")
    tok = earthdata_token()
    @printf("pair driver: target %.4f, %.4f  dates %s -> %s (tol %d d)\n",
            lat, lon, d1, d2, dtol)
    println("catalogue: CDSE OData search (footprint point-in-polygon)...")
    pd = passwant in ("desc", "asc") ?
         (passwant == "asc" ? "ASCENDING" : "DESCENDING") : ""
    c1 = odata_candidates(d1, lat, lon; dtol = dtol, passdir = pd)
    c2 = odata_candidates(d2, lat, lon; dtol = dtol, passdir = pd)
    p1ovr = get(opts, "p1", "")
    p2ovr = get(opts, "p2", "")
    s2t = 0.0
    if !isempty(p1ovr)
        # explicit product override (--p1, optionally --p2): catalogue
        # footprints overlap between slices; this pins the exact slice
        p1name = p1ovr
        println("primary   : ", p1name, "  (--p1 override)")
        s2name = p2ovr
        if isempty(s2name)
            s2name, s2t = odata_pick(c2, d2)
        else
            println("secondary : ", s2name, "  (--p2 override)")
        end
    else
        p1name, p1t = odata_pick(c1, d1)
        @printf("primary   : %s  (start %s)\n", p1name,
                Dates.format(Dates.unix2datetime(p1t), "yyyy-mm-dd HH:MM:SS"))
        # secondary: same track + pass => nearly identical UTC start time
        s2name, s2t = odata_pick(c2, d2; tod_ref = p1t)
    end
    isempty(s2name) &&
        error("no same-time-of-day product near ", d2,
              " (different track/pass); try other dates or --lat/--lon")
    s2t != 0.0 && @printf("secondary : %s  (start %s)\n", s2name,
            Dates.format(Dates.unix2datetime(s2t), "yyyy-mm-dd HH:MM:SS"))
    s2t != 0.0 && @printf("pair dt   : %.3f d\n", (s2t - p1t) / 86400)
    println("download: ASF range-ZIP (Earthdata bearer), iw2-vv...")
    tiff1, xml1 = s1_pair_download(p1name, datadir, tok)
    tiff2, xml2 = s1_pair_download(s2name, datadir, tok)
    A1 = parse_s1_annotation(xml1)
    A2 = parse_s1_annotation(xml2)
    if passwant in ("desc", "asc")
        startswith(lowercase(A1.pass_dir), passwant) ||
            error("primary pass is $(A1.pass_dir), wanted $passwant")
        startswith(lowercase(A2.pass_dir), passwant) ||
            error("secondary pass is $(A2.pass_dir), wanted $passwant")
    end
    println("annotations: primary $(A1.pass_dir) $(A1.mission), ",
            "secondary $(A2.pass_dir) $(A2.mission), ",
            "LPB $(A1.lines_per_burst)")
    k, line_rd, px_rd = s1_target_burst(A1, lat, lon)
    @printf("target RD : burst %d, line %.1f, px %.1f\n", k, line_rd, px_rd)
    dl, dp, dtp = coreg_shift_gcps(A1, A2, k)
    isempty(dl) && error("no coreg GCPs at burst $k (wrong slice?)")
    dline = sum(dl) / length(dl)
    @printf("coreg     : dline %+.2f +- %.2f (%d GCPs)\n",
            dline, sqrt(sum(abs2, dl .- dline) / length(dl)), length(dl))
    kk, l0, l1, p0, p1 = pair_auto_block(A1, A2, dline, k, line_rd, px_rd, NL0, NP)
    @printf("auto block: burst %d, lines %d..%d (%d), px %d..%d (%d)\n",
            kk, l0, l1, l1 - l0 + 1, p0, p1, p1 - p0 + 1)
    println("running interferogram pipeline (raw ifg)...")
    raw_ifg(A1, tiff1, read_slc_tiff(tiff1);
            annS = A2, tiffS_path = tiff2,
            burst = kk, l0 = l0, l1 = l1, p0 = p0, p1 = p1,
            topo = haskey(opts, "topo"),
            cohthr = num(opts, "coh", 0.15),
            unwr = get(opts, "unwrap", "wls"),
            vlat = lat, vlon = lon,
            vname = get(opts, "vname", "Geldingadalur"),
            mtitle = "LOS deformation (" * replace(d1, "-" => "") * "-" *
                     replace(d2, "-" => "") * ")",
            outdir = outdir)
    println("pair complete.")
end

# ----------------------------------------------------------------------------
# 13. Ascending/descending decomposition (raw split)
# ----------------------------------------------------------------------------
# Reads the grid_<label>.txt sidecars that raw ifg/pair --topo writes (one
# per orbit direction), bilinearly samples the ascending LOS field onto the
# descending cell grid, and solves the 2-observation LOS system per cell.
# The sidecar disp is the interferometric range change in cm (positive =
# ground moved AWAY from the satellite), i.e. d = -(m . los_hat) with the
# sidecar (ux, uy, uz) the unit ENU vector from ground toward the satellite:
#     d_desc = -(up*uz_d + east*ux_d + north*uy_d)
#     d_asc  = -(up*uz_a + east*ux_a + north*uy_a)
# with the standard dN = 0 assumption (both Sentinel-1 orbits fly close to
# N-S, so north is weakly observed; the remaining 2x2 system in (up, east)
# is well conditioned because the two geometries look to opposite sides).
# Both inputs are the tilt-removed display fields, so the outputs inherit
# that referencing. Outputs: vertical + east grids [cm] as xyz, and two
# native maps (hillshade/DEM from the descending side).

"""Parse a grid_<label>.txt sidecar ->
(H2, W2, ux, uy, uz, heading, lon, lat, disp, dem)."""
function split_read_grid(path::String)
    isfile(path) || error("no grid file: ", path)
    lines = readlines(path)
    startswith(lines[1], "#") || error("grid file has no header: ", path)
    hd = split(strip(lines[1][2:end]))
    H2 = parse(Int, hd[1]); W2 = parse(Int, hd[2])
    ux = parse(Float64, hd[3]); uy = parse(Float64, hd[4])
    uz = parse(Float64, hd[5]); hdg = parse(Float64, hd[6])
    lon = fill(NaN, H2, W2); lat = fill(NaN, H2, W2)
    disp = fill(NaN, H2, W2); dem = fill(NaN, H2, W2)
    k = 0
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        k += 1
        k > H2 * W2 && error("grid file too long: ", path)
        f = split(ln)
        j = (k - 1) ÷ W2 + 1; i = (k - 1) % W2 + 1
        lon[j, i] = parse(Float64, f[1]); lat[j, i] = parse(Float64, f[2])
        disp[j, i] = parse(Float64, f[3]); dem[j, i] = parse(Float64, f[4])
    end
    k == H2 * W2 ||
        error("grid file truncated: ", path, " (", k, " of ", H2 * W2, ")")
    return H2, W2, ux, uy, uz, hdg, lon, lat, disp, dem
end

"""Least-squares affine fit  lat = a + b*j + c*i, lon = d + e*j + f*i."""
function split_geo_affine(latg::Matrix{Float64}, long::Matrix{Float64})
    H2, W2 = size(latg)
    n = Float64(H2 * W2)
    sj = si = sjj = sii = sji = 0.0
    sa = sl = sa_j = sa_i = sl_j = sl_i = 0.0
    for j in 1:H2, i in 1:W2
        fj = Float64(j); fi = Float64(i)
        la = latg[j, i]; lo = long[j, i]
        sj += fj; si += fi; sjj += fj * fj; sii += fi * fi; sji += fj * fi
        sa += la; sl += lo
        sa_j += la * fj; sa_i += la * fi
        sl_j += lo * fj; sl_i += lo * fi
    end
    A = [n sj si; sj sjj sji; si sji sii]
    a, b, c = A \ [sa, sa_j, sa_i]
    d, e, f = A \ [sl, sl_j, sl_i]
    return a, b, c, d, e, f
end

"""(lon, lat) -> fractional (j, i) in the affine-fitted grid."""
function split_geo_inv(af, lo::Float64, la::Float64)
    a, b, c, d, e, f = af
    det = b * f - c * e
    abs(det) < 1e-14 && return NaN, NaN
    dj = ((la - a) * f - (lo - d) * c) / det
    di = (-(la - a) * e + (lo - d) * b) / det
    return dj, di
end

"""NaN-aware bilinear sample of M at fractional (j, i); NaN corners are
weighted out instead of poisoning the cell."""
function split_bil(M::Matrix{Float64}, jf::Float64, iF::Float64)
    H2, W2 = size(M)
    (jf >= 1 && jf <= H2 && iF >= 1 && iF <= W2) || return NaN
    jf <= 1 && (jf = 1.001); jf >= H2 && (jf = H2 - 0.001)
    iF <= 1 && (iF = 1.001); iF >= W2 && (iF = W2 - 0.001)
    j0 = floor(Int, jf); i0 = floor(Int, iF)
    u = jf - j0; v = iF - i0
    sw = 0.0; s = 0.0
    for (dj, di, w) in ((0, 0, (1 - u) * (1 - v)), (0, 1, (1 - u) * v),
                        (1, 0, u * (1 - v)), (1, 1, u * v))
        x = M[j0 + dj, i0 + di]
        if !isnan(x)
            s += w * x; sw += w
        end
    end
    return sw > 0 ? s / sw : NaN
end

"""raw split --desc DIR --asc DIR: 2-geometry LOS decomposition."""
function raw_split(opts::Dict{String,String}, outdir::String)
    descd = get(opts, "desc", ""); ascd = get(opts, "asc", "")
    (isempty(descd) || isempty(ascd)) &&
        error("raw split needs --desc DIR --asc DIR (raw pair/ifg --topo outputs)")
    find_grid(dir, ovr) = begin
        isempty(ovr) || return joinpath(dir, ovr)
        cands = sort(filter(f -> startswith(f, "grid_") && endswith(f, ".txt"),
                            readdir(dir)))
        isempty(cands) && error("no grid_*.txt in ", dir,
            " -- rerun raw pair/ifg --topo with a current insar_sat.jl")
        length(cands) > 1 &&
            println("  note: several grid files in ", dir, ", using ", cands[end])
        return joinpath(dir, cands[end])
    end
    pD = find_grid(descd, get(opts, "descgrid", ""))
    pA = find_grid(ascd, get(opts, "ascgrid", ""))
    Hd, Wd, uxd, uyd, uzd, hdgd, lond, latd, dispd, demd = split_read_grid(pD)
    Ha, Wa, uxa, uya, uza, hdga, lona, lata, dispa, dema = split_read_grid(pA)
    @printf("  desc: %s  %dx%d cells, heading %.1f deg, LOS enu (%+.3f %+.3f %+.3f)\n",
            pD, Hd, Wd, hdgd, uxd, uyd, uzd)
    @printf("  asc : %s  %dx%d cells, heading %.1f deg, LOS enu (%+.3f %+.3f %+.3f)\n",
            pA, Ha, Wa, hdga, uxa, uya, uza)
    det0 = uzd * uxa - uza * uxd
    abs(det0) < 0.05 &&
        error("LOS vectors too parallel for a split (det ", round(det0, digits=3),
              ") -- are both geometries really asc/desc?")
    af_a = split_geo_affine(lata, lona)
    up = fill(NaN, Hd, Wd); ea = fill(NaN, Hd, Wd)
    nv = 0
    for j in 1:Hd, i in 1:Wd
        dd = dispd[j, i]
        isnan(dd) && continue
        ja, ia = split_geo_inv(af_a, lond[j, i], latd[j, i])
        isnan(ja) && continue
        da = split_bil(dispa, ja, ia)
        isnan(da) && continue
        up[j, i] = -(uxa * dd - uxd * da) / det0
        ea[j, i] = -(uzd * da - uza * dd) / det0
        nv += 1
    end
    @printf("  split: %d / %d desc cells solved (up+east, dN = 0 assumed)\n",
            nv, Hd * Wd)
    nv == 0 && error("no overlap between the two grids -- check --desc/--asc")
    # stats (robust: 1-99 pct)
    ups = sort([abs(up[j, i]) for j in 1:Hd, i in 1:Wd if !isnan(up[j, i])])
    eas = sort([abs(ea[j, i]) for j in 1:Hd, i in 1:Wd if !isnan(ea[j, i])])
    p98(v) = v[max(1, length(v) - max(1, length(v) ÷ 100))]
    @printf("  |up|  1-99 pct max %.2f cm;  |east| 1-99 pct max %.2f cm\n",
            p98(ups), p98(eas))
    vlat = num(opts, "vlat", NaN); vlon = num(opts, "vlon", NaN)
    vname = get(opts, "vname", "Geldingadalur")
    m2 = .!isnan.(up)
    write_los_map(joinpath(outdir, "map_split_vertical.bmp"),
                  latd, lond, up, m2, demd;
                  title = "VERTICAL deformation (cm), dN=0 split",
                  vlat = vlat, vlon = vlon, vname = vname)
    write_los_map(joinpath(outdir, "map_split_east.bmp"),
                  latd, lond, ea, m2, demd;
                  title = "EAST deformation (cm), dN=0 split",
                  vlat = vlat, vlon = vlon, vname = vname)
    for (nm, fld) in (("vertical", up), ("east", ea))
        open(joinpath(outdir, "split_" * nm * ".xyz"), "w") do io
            for j in 1:Hd, i in 1:Wd
                if !isnan(fld[j, i])
                    @printf(io, "%.6f %.6f %.3f\n",
                            lond[j, i], latd[j, i], fld[j, i])
                end
            end
        end
        @printf("  wrote split_%s.xyz (%d pts)\n", nm, count(!isnan, fld))
    end
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
# 11b. Interferogram over a REAL image (s2 --insar): same physics, no DEM ->
#      no stratified atmosphere / no topographic term; fringes are shaded by
#      the real image luminance so the terrain stays recognizable.
# ----------------------------------------------------------------------------

function render_insar_over(img::Array{UInt8,3}, px::Float64, seed::Int,
                           cfg::NamedTuple)
    H, W = size(img, 1), size(img, 2)
    l = los_vector(cfg.heading, cfg.inc)
    lam = cfg.lam_cm / 100.0
    dlos = Matrix{Float64}(undef, H, W)
    source_field!(dlos, W, H, px, cfg.source, cfg.xc, cfg.yc, cfg.srcpars, l)
    m = maximum(abs, dlos)
    m > 0 && (dlos .*= (cfg.cm / 100.0) / m)
    φ = (4π / lam) .* dlos
    if cfg.atm > 0.0
        ga = NG(UInt64(seed) + 0xB00)
        ramp_a = cfg.atm * 0.9
        Threads.@threads for j in 1:H
            for i in 1:W
                turb = 2.6 * (fbm(ga, 0.006i, 0.006j; oct = 6) - 0.5) * 2.0
                ramp = ramp_a * (2.0(i / W) + (j / H) - 1.0)
                φ[j,i] += cfg.atm * turb + ramp
            end
        end
    end
    coh = Matrix{Float64}(undef, H, W)
    Threads.@threads for j in 1:H
        for i in 1:W
            g0 = φ[j,i]
            gx = i < W ? φ[j,i+1] : g0; gpx = i > 1 ? φ[j,i-1] : g0
            gy = j > 1 ? φ[j-1,i] : g0; gpy = j < H ? φ[j+1,i] : g0
            grad = max(abs(gx - gpx), abs(gy - gpy)) / 2
            coh[j,i] = clamp(cfg.coh * exp(-grad / 2.2), 0.05, 1.0)
        end
    end
    out = Array{UInt8}(undef, H, W, 3)
    looks = 4
    Threads.@threads for j in 1:H
        for i in 1:W
            g = coh[j,i]
            σ = clamp(sqrt(max(1.0 - g^2, 1e-9) / (2.0 * looks * max(g^2, 1e-6))),
                      0.02, 2.5)
            ph = φ[j,i] + σ * randn()
            rgb = insar_cmap(mod(ph / (2π), 1.0))
            lum = (0.299img[j,i,1] + 0.587img[j,i,2] + 0.114img[j,i,3]) / 255.0
            f = 0.30 + 0.85 * lum
            out[j,i,1] = to8(rgb[1] * f)
            out[j,i,2] = to8(rgb[2] * f)
            out[j,i,3] = to8(rgb[3] * f)
        end
    end
    stats = (max_los_cm = maximum(abs, dlos) * 100.0,
             fringes = maximum(abs, dlos) / (lam / 2),
             mean_coh = sum(coh) / (H * W))
    return out, stats
end

# ----------------------------------------------------------------------------
# 11c. REAL interferogram processing: a quality-guided phase unwrapper and
#      rendering of a real LiCSAR interferogram (wrapped fringes masked by
#      the real coherence, the coherence map itself, and LOS displacement
#      from OUR OWN unwrapping, in cm; positive = toward the satellite).
#      No synthetic physics here -- everything comes from the products.
# ----------------------------------------------------------------------------

wrapd(x::Float64) = x - 2π * round(x / (2π))

const DISP_STOPS = [
    (0.00, (0.15, 0.25, 0.80)),
    (0.25, (0.45, 0.65, 0.98)),
    (0.50, (1.00, 1.00, 1.00)),
    (0.75, (0.99, 0.72, 0.35)),
    (1.00, (0.80, 0.15, 0.10)),
]
disp_cmap(t::Float64) = sample_stops(DISP_STOPS, clamp01(t)) .* 255.0

mutable struct QGHeap
    q::Vector{Float64}
    idx::Vector{Int}
end
QGHeap() = QGHeap(Float64[], Int[])

function qg_push!(h::QGHeap, q::Float64, i::Int)
    push!(h.q, q); push!(h.idx, i)
    k = length(h.q)
    @inbounds while k > 1
        p = k >> 1
        h.q[p] >= h.q[k] && break
        h.q[p], h.q[k] = h.q[k], h.q[p]
        h.idx[p], h.idx[k] = h.idx[k], h.idx[p]
        k = p
    end
    return h
end

function qg_pop!(h::QGHeap)
    q0 = h.q[1]; i0 = h.idx[1]
    n = length(h.q)
    h.q[1] = h.q[n]; h.idx[1] = h.idx[n]
    pop!(h.q); pop!(h.idx)
    n -= 1
    k = 1
    @inbounds while true
        l = 2k; r = l + 1; m = k
        if l <= n && h.q[l] > h.q[m]; m = l; end
        if r <= n && h.q[r] > h.q[m]; m = r; end
        m == k && break
        h.q[k], h.q[m] = h.q[m], h.q[k]
        h.idx[k], h.idx[m] = h.idx[m], h.idx[k]
        k = m
    end
    return q0, i0
end

# quality-guided unwrapping: flood fill ordered by coherence (max-heap);
# each pixel is integrated against its first already-unwrapped neighbour
# via wrapped phase differences. Returns NaN outside the region reached.
function unwrap_quality(pha::Matrix{Float64}, coh::Matrix{Float64},
                        mask::AbstractMatrix{Bool})::Matrix{Float64}
    H, W = size(pha)
    unw = fill(NaN, H, W)
    done = falses(H, W)
    queued = falses(H, W)
    best = -1.0; blin = 0
    for j in 1:H, i in 1:W
        if mask[j,i] && coh[j,i] > best
            best = coh[j,i]; blin = (j - 1) * W + i
        end
    end
    blin == 0 && return unw
    bj = (blin - 1) ÷ W + 1; bi = (blin - 1) % W + 1
    unw[bj,bi] = 0.0
    done[bj,bi] = true
    heap = QGHeap()
    for (dj, di) in ((-1, 0), (1, 0), (0, -1), (0, 1))
        jj = bj + dj; ii = bi + di
        if 1 <= jj <= H && 1 <= ii <= W && mask[jj,ii] &&
           !done[jj,ii] && !queued[jj,ii]
            queued[jj,ii] = true
            qg_push!(heap, coh[jj,ii], (jj - 1) * W + ii)
        end
    end
    while !isempty(heap.q)
        _, lin = qg_pop!(heap)
        j = (lin - 1) ÷ W + 1; i = (lin - 1) % W + 1
        queued[j,i] = false
        done[j,i] && continue
        ref = NaN
        for (dj, di) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            jj = j + dj; ii = i + di
            if 1 <= jj <= H && 1 <= ii <= W && done[jj,ii]
                ref = unw[jj,ii] + wrapd(pha[j,i] - pha[jj,ii])
                break
            end
        end
        isnan(ref) && continue
        unw[j,i] = ref
        done[j,i] = true
        for (dj, di) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            jj = j + dj; ii = i + di
            if 1 <= jj <= H && 1 <= ii <= W && mask[jj,ii] &&
               !done[jj,ii] && !queued[jj,ii]
                queued[jj,ii] = true
                qg_push!(heap, coh[jj,ii], (jj - 1) * W + ii)
            end
        end
    end
    return unw
end

# weighted least-squares (Poisson) unwrapping, matrix-free:
#   minimize over u   sum_edges  w_ij ((u_i - u_j) - d_ij)^2
# with d_ij = wrapd(pha_i - pha_j) (Itoh) and w_ij = coh_i * coh_j. This is
# the global whole-image version of what the QG flood fill integrates along
# a single path: one bad branch decision can no longer propagate, because
# every pixel value is a compromise over ALL its weighted neighbour
# differences. The weighted Laplacian is applied as a 4-neighbour stencil
# (no sparse arrays), solved with Jacobi-preconditioned conjugate
# gradients, warm-started from the QG flood-fill solution so CG converges
# fast. `irls` reweight passes then downweight edges whose residual exceeds
# eps_l1 rad (iteratively reweighted least squares -> L1/TV behaviour:
# noise-smoothed integration, but true discontinuities and undersampled
# fringe fronts survive). Connected components other than the main region
# get their own gauge (anchored at their best-coherence pixel, u = 0), so
# coherent islands stay internally consistent. A final congruence pass
# re-attaches the wrapped phase, making the product structurally
# equivalent to the QG result (u == pha mod 2pi everywhere).
function unwrap_wls(pha::Matrix{Float64}, coh::Matrix{Float64},
                    mask::AbstractMatrix{Bool};
                    irls::Int = 2, eps_l1::Float64 = 0.4,
                    cg_tol::Float64 = 1e-4, cg_max::Int = 20000,
                    verbose::Bool = true)::Matrix{Float64}
    H, W = size(pha)
    unw = fill(NaN, H, W)
    eff = mask .& (.!isnan.(pha))          # NaN-phase px cannot be integrated
    n = count(eff)
    n == 0 && return unw
    t0 = time()
    # NOTE on indexing: every array access below is CARTESIAN. Never mix
    # formula-based linear indices ((j-1)*W+i) with real array accesses:
    # this file also runs under interpreters whose linear indexing is not
    # column-major, and the existing heap/stack round-trips (encode here,
    # decode with the same formula, never index an array) are layout-safe.
    # k <-> (pj[k], pi[k]) is the single source of truth for pixel order.
    lin = zeros(Int, H, W)                  # lin[j,i] = k (0 = not in mask)
    pj  = Vector{Int}(undef, n)
    pi  = Vector{Int}(undef, n)
    k = 0
    for j in 1:H, i in 1:W
        if eff[j, i]
            k += 1
            lin[j, i] = k
            pj[k] = j; pi[k] = i
        end
    end
    # 4 fixed neighbour directions per pixel: 1:(j,i-1)  2:(j,i+1)  E edges
#                                        3:(j-1,i)  4:(j+1,i)  N edges
    # (0 = none outside the mask). Edge factors: E edge (j,i)-(j,i+1) ->
    # fE[j,i]; N edge (j,i)-(j+1,i) -> fN[j,i].
    nb = zeros(Int, n, 4)
    @inbounds for kk in 1:n
        j = pj[kk]; i = pi[kk]
        i > 1 && lin[j, i - 1] > 0 && (nb[kk, 1] = lin[j, i - 1])
        i < W && lin[j, i + 1] > 0 && (nb[kk, 2] = lin[j, i + 1])
        j > 1 && lin[j - 1, i] > 0 && (nb[kk, 3] = lin[j - 1, i])
        j < H && lin[j + 1, i] > 0 && (nb[kk, 4] = lin[j + 1, i])
    end
    # connected components; anchor each at its best-coherence pixel (u = 0)
    comp = zeros(Int, H, W)
    anch = falses(n)
    ncomp = 0
    for kk in 1:n
        comp[pj[kk], pi[kk]] > 0 && continue
        ncomp += 1
        stack = Int[kk]                     # stack of k indices
        comp[pj[kk], pi[kk]] = ncomp
        bestc = -1.0; bestk = 0
        while !isempty(stack)
            p = pop!(stack)
            jj = pj[p]; ii = pi[p]
            if coh[jj, ii] > bestc
                bestc = coh[jj, ii]; bestk = p
            end
            for (dj, di) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                j2 = jj + dj; i2 = ii + di
                if 1 <= j2 <= H && 1 <= i2 <= W && eff[j2, i2] &&
                   comp[j2, i2] == 0
                    comp[j2, i2] = ncomp
                    push!(stack, lin[j2, i2])
                end
            end
        end
        anch[bestk] = true
    end
    # warm start: QG flood fill, unreached pixels 0
    u = unwrap_quality(pha, coh, eff)
    qgfrac = count(!isnan, u) / n
    x = Vector{Float64}(undef, n)
    for kk in 1:n
        x[kk] = isnan(u[pj[kk], pi[kk]]) ? 0.0 : u[pj[kk], pi[kk]]
    end
    verbose && @printf("  wls: %d coherent px, %d component(s), qg warm start %.1f %%\n",
                       n, ncomp, 100qgfrac)
    cohv = Vector{Float64}(undef, n)
    for kk in 1:n
        cohv[kk] = coh[pj[kk], pi[kk]]
    end
    d = zeros(n); b = zeros(n)
    fE = fill(1.0, H, W)
    fN = fill(1.0, H, W)
    # matrix-free operator: weighted Laplacian (anchor rows -> identity)
    applyA!(y, v) = begin
        @inbounds for kk in 1:n
            if anch[kk]
                y[kk] = v[kk]
            else
                j = pj[kk]; i = pi[kk]
                ci = cohv[kk]
                acc = d[kk] * v[kk]
                q = nb[kk, 1]
                q > 0 && !anch[q] && (acc -= ci * cohv[q] * fE[j, i - 1] * v[q])
                q = nb[kk, 2]
                q > 0 && !anch[q] && (acc -= ci * cohv[q] * fE[j, i] * v[q])
                q = nb[kk, 3]
                q > 0 && !anch[q] && (acc -= ci * cohv[q] * fN[j - 1, i] * v[q])
                q = nb[kk, 4]
                q > 0 && !anch[q] && (acc -= ci * cohv[q] * fN[j, i] * v[q])
                y[kk] = acc
            end
        end
        return y
    end
    r  = Vector{Float64}(undef, n)
    z  = similar(r); p = similar(r); Ap = similar(r)
    iters = 0
    for pass in 0:irls
        # assemble diagonal / rhs with the current edge factors
        fill!(d, 0.0); fill!(b, 0.0)
        @inbounds for kk in 1:n
            anch[kk] && (d[kk] = 1.0; continue)
            j = pj[kk]; i = pi[kk]
            ci = cohv[kk]
            q = nb[kk, 1]
            q > 0 && (w = ci * cohv[q] * fE[j, i - 1]; d[kk] += w;
                      b[kk] += w * wrapd(pha[j, i] - pha[j, i - 1]))
            q = nb[kk, 2]
            q > 0 && (w = ci * cohv[q] * fE[j, i]; d[kk] += w;
                      b[kk] += w * wrapd(pha[j, i] - pha[j, i + 1]))
            q = nb[kk, 3]
            q > 0 && (w = ci * cohv[q] * fN[j - 1, i]; d[kk] += w;
                      b[kk] += w * wrapd(pha[j, i] - pha[j - 1, i]))
            q = nb[kk, 4]
            q > 0 && (w = ci * cohv[q] * fN[j, i]; d[kk] += w;
                      b[kk] += w * wrapd(pha[j, i] - pha[j + 1, i]))
        end
        # Jacobi-preconditioned CG (warm-started x)
        applyA!(Ap, x); r .= b .- Ap
        z .= r ./ d; p .= z
        rz = dot(r, z)
        bnorm = norm(b)
        tol2 = (cg_tol * bnorm)^2
        it = 0
        while it < cg_max && dot(r, r) > tol2
            applyA!(Ap, p)
            alpha = rz / dot(p, Ap)
            x .+= alpha .* p
            r .-= alpha .* Ap
            z .= r ./ d
            rznew = dot(r, z)
            p .= z .+ (rznew / rz) .* p
            rz = rznew
            it += 1
        end
        iters += it
        verbose && @printf("  wls: pass %d/%d  cg %d iters, |r|/|b| = %.1e\n",
                           pass + 1, irls + 1, it,
                           sqrt(dot(r, r)) / max(bnorm, 1e-12))
        # IRLS reweight: soften edges with residuals beyond eps_l1 rad
        if pass < irls
            @inbounds for kk in 1:n
                j = pj[kk]; i = pi[kk]
                q = nb[kk, 2]
                if q > 0
                    res = (x[kk] - x[q]) - wrapd(pha[j, i] - pha[j, i + 1])
                    fE[j, i] *= 1.0 / sqrt(1.0 + (res / eps_l1)^2)
                end
                q = nb[kk, 4]
                if q > 0
                    res = (x[kk] - x[q]) - wrapd(pha[j, i] - pha[j + 1, i])
                    fN[j, i] *= 1.0 / sqrt(1.0 + (res / eps_l1)^2)
                end
            end
        end
    end
    verbose && @printf("  wls: total %d cg iterations, %.1f s\n",
                       iters, time() - t0)
    # congruence: re-attach the wrapped phase (u == pha mod 2pi)
    for kk in 1:n
        j = pj[kk]; i = pi[kk]
        unw[j, i] = pha[j, i] + 2π * round((x[kk] - pha[j, i]) / (2π))
    end
    return unw
end

# render the three real-InSAR products
function render_real_insar(pha::Matrix{Float64}, coh::Matrix{Float64},
                          unw::Matrix{Float64}, lam_cm::Float64,
                          cohthr::Float64)
    H, W = size(pha)
    mask = (coh .>= cohthr) .& (.!isnan.(pha))
    dcm = similar(pha)                        # LOS displacement [cm]
    @inbounds for k in eachindex(dcm)
        u = unw[k]
        dcm[k] = isnan(u) ? NaN : -u * lam_cm / (4π)
    end
    m = 0.0
    @inbounds for k in eachindex(dcm)
        isnan(dcm[k]) || (m = max(m, abs(dcm[k])))
    end
    m > 0 || (m = 1.0)
    img  = Array{UInt8}(undef, H, W, 3)
    imgc = Array{UInt8}(undef, H, W, 3)
    imgd = Array{UInt8}(undef, H, W, 3)
    Threads.@threads for j in 1:H
        @inbounds for i in 1:W
            g = coh[j,i]
            c = coh_cmap(g)
            imgc[j,i,1] = to8(c[1]); imgc[j,i,2] = to8(c[2]); imgc[j,i,3] = to8(c[3])
            if mask[j,i]
                t = mod(pha[j,i] / (2π), 1.0)
                rgb = insar_cmap(t)
                f = 0.72 + 0.28 * g
                img[j,i,1] = to8(rgb[1] * f)
                img[j,i,2] = to8(rgb[2] * f)
                img[j,i,3] = to8(rgb[3] * f)
            else
                v = to8(20.0 + 40.0 * g)
                img[j,i,1] = v; img[j,i,2] = v; img[j,i,3] = v
            end
            if isnan(unw[j,i])
                imgd[j,i,1] = 24; imgd[j,i,2] = 24; imgd[j,i,3] = 24
            else
                c2 = disp_cmap((dcm[j,i] / m + 1.0) / 2.0)
                imgd[j,i,1] = to8(c2[1]); imgd[j,i,2] = to8(c2[2])
                imgd[j,i,3] = to8(c2[3])
            end
        end
    end
    stats = (max_cm = m,
             mean_coh = count(mask) > 0 ? sum(coh[mask]) / count(mask) : 0.0,
             nunw = count(x -> !isnan(x), unw),
             ncoh = count(mask))
    return img, imgc, imgd, dcm, stats
end

# ---- nicer real-pair renderings: masked, filtered, detrended (v4.1) ---------

"""Coherence-weighted masked boxcar filter on a complex interferogram.
`r` = half-width (5x5 at r=2 on the multilooked grid): smooths speckle
without wiping the dense fringes near the vent. Masked pixels carry no
weight; a pixel whose whole window is masked falls back to itself."""
function filter_ifg_cpx(c::Matrix{ComplexF64}, coh::Matrix{Float64},
                        mask::AbstractMatrix{Bool}; r::Int = 2)
    H, W = size(c)
    out = zeros(ComplexF64, H, W)
    @inbounds for j in 1:H, i in 1:W
        acc = zero(ComplexF64); ws = 0.0
        for dj in -r:r, di in -r:r
            jj = clamp(j + dj, 1, H); ii = clamp(i + di, 1, W)
            if mask[jj, ii]
                wt = coh[jj, ii]
                acc += wt * c[jj, ii]; ws += wt
            end
        end
        out[j, i] = ws > 0 ? acc / ws : c[j, i]
    end
    return out
end

"""LSQ 2-D quadratic (1, x, z, x^2, xz, z^2) of dcm [cm] over masked
pixels; returns coefficients or nothing when too few points."""
function quadfit_cm(dcm::Matrix{Float64}, mask::AbstractMatrix{Bool})
    idx = [k for k in eachindex(dcm) if mask[k] && !isnan(dcm[k])]
    length(idx) < 200 && return nothing
    Hm, Wm = size(dcm)
    X = zeros(length(idx), 6); y = zeros(length(idx))
    for (n, k) in enumerate(idx)
        r, c = Tuple(CartesianIndices(dcm)[k])
        x = (c - 0.5) / Wm - 0.5; z = (r - 0.5) / Hm - 0.5
        X[n, :] .= (1.0, x, z, x * x, x * z, z * z)
        y[n] = dcm[k]
    end
    return (X' * X) \ (X' * y)
end

"""Evaluate a quadfit_cm coefficient vector on the full grid."""
function quad_eval(coef::Vector{Float64}, sz::Tuple{Int,Int})
    Hm, Wm = sz
    out = zeros(Hm, Wm)
    @inbounds for j in 1:Hm, i in 1:Wm
        x = (i - 0.5) / Wm - 0.5; z = (j - 0.5) / Hm - 0.5
        out[j, i] = coef[1] + coef[2] * x + coef[3] * z +
                    coef[4] * x * x + coef[5] * x * z + coef[6] * z * z
    end
    return out
end

"""98th-percentile symmetric display scale for a detrended map [cm]."""
function defo_scale(dres::Matrix{Float64}, mask::AbstractMatrix{Bool})
    v = [abs(dres[k]) for k in eachindex(dres) if mask[k] && !isnan(dres[k])]
    isempty(v) && return 1.0
    sort!(v)
    return max(v[round(Int, 0.98 * length(v))], 0.5)
end

"""Coh-weighted masked boxcar filter on a REAL field (use after detrending:
the residual is smooth, so averaging kills per-pixel noise ~window size)."""
function filter_real(v::Matrix{Float64}, coh::Matrix{Float64},
                     mask::AbstractMatrix{Bool}; r::Int = 2)
    H, W = size(v)
    out = zeros(H, W)
    @inbounds for j in 1:H, i in 1:W
        acc = 0.0; ws = 0.0
        for dj in -r:r, di in -r:r
            jj = clamp(j + dj, 1, H); ii = clamp(i + di, 1, W)
            if mask[jj, ii] && !isnan(v[jj, ii])
                wt = coh[jj, ii]
                acc += wt * v[jj, ii]; ws += wt
            end
        end
        out[j, i] = ws > 0 ? acc / ws : v[j, i]
    end
    return out
end

"""Nearest-neighbour upscale of an RGB image: frow rows per pixel,
fcol columns per pixel (frow > fcol corrects the az/rg pixel aspect).
`bilinear=true` interpolates instead of replicating: smoother display
for grayscale-dominated products (overlay); fringes stay crisp with
the default nearest neighbour."""
function upscale_img(img::Array{UInt8,3}, frow::Int, fcol::Int;
                     bilinear::Bool = false)
    H, W = size(img, 1), size(img, 2)
    out = Array{UInt8}(undef, H * frow, W * fcol, 3)
    if !bilinear
        @inbounds for j in 1:H, i in 1:W
            for dj in 1:frow, di in 1:fcol
                for c in 1:3
                    out[(j - 1) * frow + dj, (i - 1) * fcol + di, c] =
                        img[j, i, c]
                end
            end
        end
        return out
    end
    # bilinear in source-cell coordinates (centres at 0.5, 0.5)
    OH, OW = H * frow, W * fcol
    @inbounds for J in 1:OH, I in 1:OW
        y = (J - 0.5) / frow          # source row centre coord
        x = (I - 0.5) / fcol
        j0 = clamp(floor(Int, y), 1, H - 1)
        i0 = clamp(floor(Int, x), 1, W - 1)
        ty = clamp(y - j0, 0.0, 1.0)
        tx = clamp(x - i0, 0.0, 1.0)
        for c in 1:3
            v = (1 - ty) * ((1 - tx) * img[j0, i0, c] +
                            tx * img[j0, i0 + 1, c]) +
                ty * ((1 - tx) * img[j0 + 1, i0, c] +
                      tx * img[j0 + 1, i0 + 1, c])
            out[J, I, c] = unsafe_trunc(UInt8, v)
        end
    end
    return out
end

"""2-D FFT of a power-of-2 patch via the radix-2 fft_c! (rows, then
columns). No scaling; callers divide by H*W for the inverse."""
function fft2_patch(P::Matrix{ComplexF64}, forward::Bool)
    A = copy(P)
    H, W = size(A)
    for j in 1:H
        r = vec(A[j, :]); fft_c!(r, forward); A[j, :] .= r
    end
    for i in 1:W
        c = vec(A[:, i]); fft_c!(c, forward); A[:, i] .= c
    end
    return A
end

"""Adaptive Goldstein filter (Werner 2000 / Baran 2003): patch-wise
spectral weighting S' = S * (|S|/max|S||^alpha with alpha = 1 - mean
patch coherence. Coherent patches stay sharp, speckle patches get
smoothed. Patches paz x prg (powers of 2), 50 % overlap, Hann blend,
mirror-padded edges; masked pixels are filled with the patch's
coherent mean. Returns the filtered complex interferogram."""
function goldstein_filter(c::Matrix{ComplexF64}, coh::Matrix{Float64},
                          mask::AbstractMatrix{Bool};
                          paz::Int = 32, prg::Int = 64)
    H, W = size(c)
    hopz = paz ÷ 2; hopr = prg ÷ 2
    HP = max(paz, ceil(Int, (H - paz) / hopz) * hopz + paz)
    WP = max(prg, ceil(Int, (W - prg) / hopr) * hopr + prg)
    P  = Matrix{ComplexF64}(undef, HP, WP)
    CH = zeros(Float64, HP, WP); MK = zeros(Bool, HP, WP)
    @inbounds for j in 1:HP, i in 1:WP
        jj = j <= H ? j : max(1, 2H - j)
        ii = i <= W ? i : max(1, 2W - i)
        P[j, i] = c[jj, ii]; CH[j, i] = coh[jj, ii]; MK[j, i] = mask[jj, ii]
    end
    wz = [0.5 - 0.5 * cos(2π * (t - 1) / (paz - 1)) for t in 1:paz]
    wr = [0.5 - 0.5 * cos(2π * (t - 1) / (prg - 1)) for t in 1:prg]
    out = zeros(ComplexF64, HP, WP)
    wsum = zeros(Float64, HP, WP)
    for z0 in 1:hopz:HP - paz + 1, r0 in 1:hopr:WP - prg + 1
        sel = MK[z0:z0+paz-1, r0:r0+prg-1]
        nsel = count(sel)
        gbar = nsel > 0 ?
            sum(CH[z0:z0+paz-1, r0:r0+prg-1][sel]) / nsel : 0.0
        mval = nsel > 0 ?
            sum(P[z0:z0+paz-1, r0:r0+prg-1][sel]) / nsel : 0.0im
        Pt = Matrix{ComplexF64}(undef, paz, prg)
        @inbounds for a in 1:paz, b in 1:prg
            Pt[a, b] = sel[a, b] ? P[z0+a-1, r0+b-1] : mval
        end
        S = fft2_patch(Pt, true)
        alpha = clamp(1.0 - gbar, 0.0, 0.9)
        amax = maximum(abs.(S))
        if amax > 0
            @inbounds for k in eachindex(S)
                S[k] *= (abs(S[k]) / amax)^alpha
            end
        end
        Pt = fft2_patch(S, false) ./ (paz * prg)
        @inbounds for a in 1:paz, b in 1:prg
            wgt = wz[a] * wr[b] + 1e-3
            out[z0+a-1, r0+b-1] += wgt * Pt[a, b]
            wsum[z0+a-1, r0+b-1] += wgt
        end
    end
    out ./= wsum
    return out[1:H, 1:W]
end

"""Simple NW-lit hillshade [0..1] of a DEM grid [m]; drow/dcol = pixel
sizes [m]. NaN DEM falls back to flat (1.0)."""
function hillshade(dem::Matrix{Float64}, drow::Float64, dcol::Float64)
    H, W = size(dem)
    out = ones(H, W)
    @inbounds for j in 1:H, i in 1:W
        zc = dem[j, i]
        isnan(zc) && continue
        ju = clamp(j + 1, 1, H); jd = clamp(j - 1, 1, H)
        iu = clamp(i + 1, 1, W); idn = clamp(i - 1, 1, W)
        zr = isnan(dem[ju, i]) ? zc : dem[ju, i]
        zl = isnan(dem[jd, i]) ? zc : dem[jd, i]
        zu = isnan(dem[j, iu]) ? zc : dem[j, iu]
        zd = isnan(dem[j, idn]) ? zc : dem[j, idn]
        dzdr = (zr - zl) / (2.0 * drow)      # up-slope in +row
        dzdc = (zu - zd) / (2.0 * dcol)      # up-slope in +col
        nx, ny, nz = -dzdr, -dzdc, 1.0
        nl = sqrt(nx * nx + ny * ny + nz * nz)
        # light from NW, 55 deg elevation
        out[j, i] = clamp((-0.42 * nx - 0.42 * ny + 0.8 * nz) / nl, 0.0, 1.0)
    end
    return out
end

"""Fringes over a terrain background. When `dem` is given the base is
the hillshade alone (smooth, map-like): land cells shaded by relief,
ocean/lake cells (DEM <= 1 m) flat dark. Without `dem` the base is the
log-amplitude (SAR texture) optionally multiplied by the hillshade.
Wrapped phase at `beta` opacity on masked pixels."""
function render_overlay_img(pha::Matrix{Float64}, mask::AbstractMatrix{Bool},
                            amp::Matrix{Float64}, hs::Union{Nothing,Matrix{Float64}};
                            beta::Float64 = 0.72,
                            dem::Union{Nothing,Matrix{Float64}} = nothing)
    H, W = size(pha)
    v = [amp[k] for k in eachindex(amp) if !isnan(amp[k]) && amp[k] > 0]
    (isempty(v) || length(v) < 50) && (v = [0.0, 1.0])
    sort!(v)
    a1 = v[max(1, round(Int, 0.05 * length(v)))]
    a2 = v[min(end, round(Int, 0.95 * length(v)))]
    a2 > a1 || (a2 = a1 + 1.0)
    la1, la2 = log1p(a1), log1p(a2)
    img = Array{UInt8}(undef, H, W, 3)
    @inbounds for j in 1:H, i in 1:W
        if dem !== nothing
            d = isnan(dem[j, i]) ? 0.0 : dem[j, i]
            shade = hs === nothing ? 1.0 : (0.35 + 0.65 * hs[j, i])
            gv = d <= 1.0 ? 52.0 : 58.0 + 165.0 * shade
        else
            g = isnan(amp[j, i]) ? a1 : clamp(amp[j, i], a1, a2)
            t = (log1p(g) - la1) / (la2 - la1)
            shade = hs === nothing ? 1.0 : (0.35 + 0.65 * hs[j, i])
            gv = 40.0 + 180.0 * clamp(t, 0.0, 1.0) * shade
        end
        if mask[j, i]
            rgb = insar_cmap(mod(pha[j, i] / (2π), 1.0))
            img[j, i, 1] = to8((1 - beta) * gv + beta * rgb[1])
            img[j, i, 2] = to8((1 - beta) * gv + beta * rgb[2])
            img[j, i, 3] = to8((1 - beta) * gv + beta * rgb[3])
        elseif dem !== nothing && ((isnan(dem[j, i]) ? 0.0 : dem[j, i]) <= 1.0)
            # water: cool slate tint reads as ocean/coastline
            img[j, i, 1] = to8(0.80 * gv)
            img[j, i, 2] = to8(0.92 * gv)
            img[j, i, 3] = to8(1.10 * gv)
        else
            img[j, i, 1] = to8(0.72 * gv)
            img[j, i, 2] = to8(0.72 * gv)
            img[j, i, 3] = to8(0.72 * gv)
        end
    end
    return img
end

"""Wrapped-phase fringe image; masked-out pixels dark (near-black)."""
function render_phase_img(pha::Matrix{Float64}, mask::AbstractMatrix{Bool})
    H, W = size(pha)
    img = Array{UInt8}(undef, H, W, 3)
    @inbounds for j in 1:H, i in 1:W
        if mask[j, i]
            rgb = insar_cmap(mod(pha[j, i] / (2π), 1.0))
            img[j, i, 1] = to8(rgb[1]); img[j, i, 2] = to8(rgb[2])
            img[j, i, 3] = to8(rgb[3])
        else
            img[j, i, 1] = 18; img[j, i, 2] = 18; img[j, i, 3] = 26
        end
    end
    return img
end

"""Symmetric (+-s) deformation map; masked-out pixels dark."""
function render_defo_img(dres::Matrix{Float64}, mask::AbstractMatrix{Bool}, s::Float64)
    H, W = size(dres)
    img = Array{UInt8}(undef, H, W, 3)
    @inbounds for j in 1:H, i in 1:W
        if mask[j, i] && !isnan(dres[j, i])
            c2 = disp_cmap(clamp((dres[j, i] / s + 1.0) / 2.0, 0.0, 1.0))
            img[j, i, 1] = to8(c2[1]); img[j, i, 2] = to8(c2[2])
            img[j, i, 3] = to8(c2[3])
        else
            img[j, i, 1] = 24; img[j, i, 2] = 24; img[j, i, 3] = 30
        end
    end
    return img
end


# ----------------------------------------------------------------------------
# 12. Native cartography: Vedurstofa-style LOS deformation map
# (pure stdlib: no GMT, no font files, no image packages)
# ----------------------------------------------------------------------------

# compact 5x7 bitmap font; 7 rows joined by "/" (uppercase only)
const MAP_FONT = Dict{Char,String}(
  '0' => "01110/10001/10011/10101/11001/10001/01110",
  '1' => "00100/01100/00100/00100/00100/00100/01110",
  '2' => "01110/10001/00001/00010/00100/01000/11111",
  '3' => "11111/00010/00100/00010/00001/10001/01110",
  '4' => "00010/00110/01010/10010/11111/00010/00010",
  '5' => "11111/10000/11110/00001/00001/10001/01110",
  '6' => "00110/01000/10000/11110/10001/10001/01110",
  '7' => "11111/00001/00010/00100/01000/01000/01000",
  '8' => "01110/10001/10001/01110/10001/10001/01110",
  '9' => "01110/10001/10001/01111/00001/00010/01100",
  'A' => "01110/10001/10001/11111/10001/10001/10001",
  'B' => "11110/10001/10001/11110/10001/10001/11110",
  'C' => "01110/10001/10000/10000/10000/10001/01110",
  'D' => "11100/10010/10001/10001/10001/10010/11100",
  'E' => "11111/10000/10000/11110/10000/10000/11111",
  'F' => "11111/10000/10000/11110/10000/10000/10000",
  'G' => "01110/10001/10000/10111/10001/10001/01111",
  'H' => "10001/10001/10001/11111/10001/10001/10001",
  'I' => "01110/00100/00100/00100/00100/00100/01110",
  'J' => "00111/00010/00010/00010/00010/10010/01100",
  'K' => "10001/10010/10100/11000/10100/10010/10001",
  'L' => "10000/10000/10000/10000/10000/10000/11111",
  'M' => "10001/11011/10101/10101/10001/10001/10001",
  'N' => "10001/10001/11001/10101/10011/10001/10001",
  'O' => "01110/10001/10001/10001/10001/10001/01110",
  'P' => "11110/10001/10001/11110/10000/10000/10000",
  'Q' => "01110/10001/10001/10001/10101/10010/01101",
  'R' => "11110/10001/10001/11110/10100/10010/10001",
  'S' => "01111/10000/10000/01110/00001/00001/11110",
  'T' => "11111/00100/00100/00100/00100/00100/00100",
  'U' => "10001/10001/10001/10001/10001/10001/01110",
  'V' => "10001/10001/10001/10001/10001/01010/00100",
  'W' => "10001/10001/10001/10101/10101/11011/10001",
  'X' => "10001/10001/01010/00100/01010/10001/10001",
  'Y' => "10001/10001/01010/00100/00100/00100/00100",
  'Z' => "11111/00001/00010/00100/01000/10000/11111",
  ' ' => "00000/00000/00000/00000/00000/00000/00000",
  '-' => "00000/00000/00000/11111/00000/00000/00000",
  '.' => "00000/00000/00000/00000/00000/01100/01100",
  '(' => "00010/00100/01000/01000/01000/00100/00010",
  ')' => "01000/00100/00010/00010/00010/00100/01000",
  '[' => "01110/01000/01000/01000/01000/01000/01110",
  ']' => "01110/00010/00010/00010/00010/00010/01110",
  '+' => "00000/00100/00100/11111/00100/00100/00000",
  '/' => "00001/00010/00010/00100/01000/01000/10000",
  ':' => "00000/01100/01100/00000/01100/01100/00000",
  '°' => "01100/10010/10010/01100/00000/00000/00000")

map_text_w(s::AbstractString, scale::Int) = max(0, length(s) * 6 * scale - scale)

function map_draw_text!(img::Array{UInt8,3}, x0::Int, y0::Int, s::AbstractString;
                        scale::Int = 2, col::NTuple{3,Int} = (0, 0, 0))
    H, W = size(img, 1), size(img, 2)
    x = x0
    for ch in s
        rows = split(get(MAP_FONT, uppercase(ch), MAP_FONT[' ']), '/')
        for r in 1:7, c in 1:5
            rows[r][c] == '1' || continue
            for sy in 0:scale-1, sx in 0:scale-1
                yy = y0 + (r - 1) * scale + sy
                xx = x + (c - 1) * scale + sx
                if 1 <= yy <= H && 1 <= xx <= W
                    img[yy, xx, 1] = col[1]; img[yy, xx, 2] = col[2]
                    img[yy, xx, 3] = col[3]
                end
            end
        end
        x += 6 * scale
    end
end

"""Classic jet colormap, t in [0,1] -> (r,g,b) floats 0..255."""
function map_jet(t::Float64)
    r = clamp(1.5 - abs(4t - 3), 0.0, 1.0)
    g = clamp(1.5 - abs(4t - 2), 0.0, 1.0)
    b = clamp(1.5 - abs(4t - 1), 0.0, 1.0)
    (255r, 255g, 255b)
end

"""NaN-aware masked boxcar smoothing on a cell grid; holes up to
r*passes cells across get filled from their valid neighbours."""
function map_boxsmooth(v::Matrix{Float64}, r::Int, passes::Int)
    out = copy(v)
    H, W = size(v)
    for _ in 1:passes
        src = copy(out)
        for j in 1:H, i in 1:W
            s = 0.0; n = 0
            for dj in -r:r, di in -r:r
                x = src[clamp(j + dj, 1, H), clamp(i + di, 1, W)]
                if !isnan(x); s += x; n += 1; end
            end
            out[j, i] = n > 0 ? s / n : NaN
        end
    end
    return out
end

"""Hillshade 0..1 from a cell DEM (metres), light from NW, 45 deg up."""
function map_hillshade(dem::Matrix{Float64}; cellm::Float64 = 111.0,
                       southrows::Bool = false)
    H, W = size(dem)
    sh = fill(0.5, H, W)
    lx, ly, lz = cosd(315) * cosd(45), sind(315) * cosd(45), sind(45)
    for j in 2:H-1, i in 2:W-1
        dzdx = (dem[j, i+1] - dem[j, i-1]) / (2cellm)
        dzdy = (dem[j+1, i] - dem[j-1, i]) / (2cellm)
        southrows && (dzdy = -dzdy)      # row 1 = north -> flip y
        nrm = hypot(dzdx, dzdy, 1.0)
        sh[j, i] = clamp((-dzdx * lx + dzdy * ly + lz) / nrm, 0.0, 1.0)
    end
    return sh
end

function map_inpoly(px::Float64, py::Float64, pts::Vector{NTuple{2,Float64}})
    inside = false
    a = pts[end]
    for b in pts
        if (b[2] > py) != (a[2] > py) &&
           px < (a[1] - b[1]) * (py - b[2]) / (a[2] - b[2]) + b[1]
            inside = !inside
        end
        a = b
    end
    return inside
end

function map_star!(img::Array{UInt8,3}, cx::Float64, cy::Float64, r::Float64)
    H, W = size(img, 1), size(img, 2)
    pts = NTuple{2,Float64}[]
    for k in 0:9
        rr = iseven(k) ? r : 0.42r
        ang = -pi / 2 + k * pi / 5
        push!(pts, (cx + rr * cos(ang), cy + rr * sin(ang)))
    end
    x0, x1 = floor(Int, cx - r), ceil(Int, cx + r)
    y0, y1 = floor(Int, cy - r), ceil(Int, cy + r)
    star = falses(y1 - y0 + 3, x1 - x0 + 3)      # 1-px pad for border test
    for yy in y0:y1, xx in x0:x1
        star[yy-y0+2, xx-x0+2] = map_inpoly(xx + 0.5, yy + 0.5, pts)
    end
    for yy in y0:y1, xx in x0:x1
        (1 <= yy <= H && 1 <= xx <= W) || continue
        if star[yy-y0+2, xx-x0+2]
            nb = star[yy-y0+1,xx-x0+2] || star[yy-y0+3,xx-x0+2] ||
                 star[yy-y0+2,xx-x0+1] || star[yy-y0+2,xx-x0+3]
            c = nb ? 255 : 0        # white outline, black fill
            img[yy, xx, 1] = c; img[yy, xx, 2] = c; img[yy, xx, 3] = c
        end
    end
end


"""Vedurstofa-style LOS deformation map, pure stdlib:
gray hillshade land + light-blue ocean (from the GLO-90 DEM: ocean = 0 m),
jet-coloured detrended LOS field on the coherent area only, coastline,
lon/lat frame with ticks, horizontal colorbar, title, optional vent star.
The geolocated cell grid is slightly rotated; an affine fit maps output
pixels (lon/lat) back onto fractional grid coordinates for sampling."""
function write_los_map(path::String, latg::Matrix{Float64}, long::Matrix{Float64},
                       disp::Matrix{Float64}, mask::AbstractMatrix{Bool},
                       dem::Matrix{Float64};
                       title::String = "LOS deformation",
                       vlat::Float64 = NaN, vlon::Float64 = NaN,
                       vname::String = "Geldingadalur")
    H2, W2 = size(latg)
    (H2 < 8 || W2 < 8) && (println("  map: grid too small, skipping"); return)

    # ---- affine fit  lat = a + b*j + c*i,  lon = d + e*j + f*i -------
    n = Float64(H2 * W2)
    sj = si = sjj = sii = sji = 0.0
    sa = sl = sa_j = sa_i = sl_j = sl_i = 0.0
    for j in 1:H2, i in 1:W2
        fj = Float64(j); fi = Float64(i)
        la = latg[j, i]; lo = long[j, i]
        sj += fj; si += fi; sjj += fj*fj; sii += fi*fi; sji += fj*fi
        sa += la; sl += lo
        sa_j += la*fj; sa_i += la*fi; sl_j += lo*fj; sl_i += lo*fi
    end
    A = [n sj si; sj sjj sji; si sji sii]
    ba = [sa, sa_j, sa_i]; bl = [sl, sl_j, sl_i]
    a, b, c = A \ ba
    d, e, f = A \ bl
    det = b * f - c * e
    if abs(det) < 1e-12
        println("  map: degenerate geolocation grid, skipping map")
        return
    end

    # ---- cell-grid preprocessing --------------------------------------
    # data support: within ~3 cells (~330 m) of a coherent cell
    support = copy(Bool.(mask))
    for _ in 1:3
        grown = falses(H2, W2)
        for j in 1:H2, i in 1:W2
            if support[j, i]
                grown[max(1,j-1):min(H2,j+1), max(1,i-1):min(W2,i+1)] .= true
            end
        end
        support = grown
    end
    maskf = map_boxsmooth(Float64.(support), 1, 1)
    dispf = copy(disp)
    for k in eachindex(dispf)
        (isnan(dispf[k]) || !mask[k]) && (dispf[k] = NaN)
    end
    dispf = map_boxsmooth(dispf, 3, 3)          # ~350 m smoothing + hole fill
    dems = map_boxsmooth(dem, 2, 1)
    southrows = latg[2, 1] > latg[1, 1]         # row 1 = south?
    shade = map_hillshade(dems; southrows = southrows)

    v99 = [abs(dispf[k]) for k in eachindex(dispf) if mask[k]]
    isempty(v99) && (v99 = [defo_scale(disp, mask)])
    zs = max(2, 2 * ceil(Int, maximum(v99) / 2))   # even-integer colour range

    # ---- layout ---------------------------------------------------------
    latmin = minimum(latg); latmax = maximum(latg)
    lonmin = minimum(long); lonmax = maximum(long)
    # white context margin around the data (like a GMT region pad)
    mrg = 0.10
    lonmin -= mrg * (lonmax - lonmin); lonmax += mrg * (lonmax - lonmin)
    latmin -= mrg * (latmax - latmin); latmax += mrg * (latmax - latmin)
    latm = 0.5 * (latmin + latmax)
    kx = cosd(latm)
    Xspan = (lonmax - lonmin) * kx
    Yspan = latmax - latmin
    S = 980.0 / max(Xspan, Yspan)                # px per (lat-equivalent) degree
    mapW = max(64, round(Int, Xspan * S))
    mapH = max(64, round(Int, Yspan * S))
    ML, MR, MT, MB = 86, 86, 54, 118             # margins: labels/colorbar
    Htot = mapH + MT + MB
    Wtot = mapW + ML + MR
    img = fill(UInt8(255), Htot, Wtot, 3)

    # ---- background + displacement (affine inverse + bilinear) ---------
    sample(g::Matrix{Float64}, la::Float64, lo::Float64) = begin
        dy = (la - a) * f - (lo - d) * c
        dx = (lo - d) * b - (la - a) * e
        fj0 = dy / det; fi0 = dx / det
        if fj0 < 1.0 || fj0 > Float64(H2) || fi0 < 1.0 || fi0 > Float64(W2)
            return NaN           # outside the data grid (margin)
        end
        fj = clamp(fj0, 1.0, Float64(H2))
        fi = clamp(fi0, 1.0, Float64(W2))
        j0 = min(floor(Int, fj), H2 - 1); i0 = min(floor(Int, fi), W2 - 1)
        fy = fj - j0; fx = fi - i0
        v00 = g[j0, i0]; v01 = g[j0, i0+1]; v10 = g[j0+1, i0]; v11 = g[j0+1, i0+1]
        s = 0.0; w = 0.0
        isnan(v00) || (s += v00 * (1-fx)*(1-fy); w += (1-fx)*(1-fy))
        isnan(v01) || (s += v01 * fx*(1-fy);    w += fx*(1-fy))
        isnan(v10) || (s += v10 * (1-fx)*fy;    w += (1-fx)*fy)
        isnan(v11) || (s += v11 * fx*fy;        w += fx*fy)
        w > 1e-6 ? s / w : NaN
    end

    land = falses(mapH, mapW)
    for py in 1:mapH, px in 1:mapW
        la = latmax - (py - 0.5) / S
        lo = lonmin + (px - 0.5) / (S * kx)
        demv = sample(dems, la, lo)
        isLand = !isnan(demv) && demv > 0.5
        land[py, px] = isLand
        if isnan(demv)
            img[MT+py, ML+px, 1] = 250; img[MT+py, ML+px, 2] = 250
            img[MT+py, ML+px, 3] = 252
        elseif isLand
            shv = clamp(sample(shade, la, lo), 0.0, 1.0)
            g = 205 + 45 * shv
            img[MT+py, ML+px, 1] = to8(g); img[MT+py, ML+px, 2] = to8(g)
            img[MT+py, ML+px, 3] = to8(0.98g)
        else
            img[MT+py, ML+px, 1] = 208; img[MT+py, ML+px, 2] = 224
            img[MT+py, ML+px, 3] = 238
        end
        mv = sample(maskf, la, lo)
        if mv > 0.4
            dv = sample(dispf, la, lo)
            if !isnan(dv)
                jc = map_jet(clamp((dv + zs) / (2zs), 0.0, 1.0))
                img[MT+py, ML+px, 1] = to8(jc[1])
                img[MT+py, ML+px, 2] = to8(jc[2])
                img[MT+py, ML+px, 3] = to8(jc[3])
            end
        end
    end

    # ---- coastline (land/ocean boundary from the DEM) -------------------
    for py in 1:mapH, px in 1:mapW
        if land[py, px] != get(land, (clamp(py+1,1,mapH), px), land[py,px]) ||
           land[py, px] != get(land, (clamp(py-1,1,mapH), px), land[py,px]) ||
           land[py, px] != get(land, (py, clamp(px+1,1,mapW)), land[py,px]) ||
           land[py, px] != get(land, (py, clamp(px-1,1,mapW)), land[py,px])
            img[MT+py, ML+px, 1] = 70; img[MT+py, ML+px, 2] = 70
            img[MT+py, ML+px, 3] = 70
        end
    end

    # ---- frame + ticks + labels -----------------------------------------
    for t in 0:1   # 2-px frame
        for px in 1:mapW
            img[MT+t, ML+px, 1] = 0; img[MT+t, ML+px, 2] = 0; img[MT+t, ML+px, 3] = 0
            img[MT+mapH+1-t, ML+px, 1] = 0; img[MT+mapH+1-t, ML+px, 2] = 0
            img[MT+mapH+1-t, ML+px, 3] = 0
        end
        for py in 1:mapH
            img[MT+py, ML+t, 1] = 0; img[MT+py, ML+t, 2] = 0; img[MT+py, ML+t, 3] = 0
            img[MT+py, ML+mapW+1-t, 1] = 0; img[MT+py, ML+mapW+1-t, 2] = 0
            img[MT+py, ML+mapW+1-t, 3] = 0
        end
    end
    stepnice(span::Float64) = begin
        raw = span / 5.0
        p = 10.0^floor(log10(raw)); m = raw / p
        (m < 1.5 ? 1.0 : m < 3.5 ? 2.0 : m < 7.5 ? 5.0 : 10.0) * p
    end
    lonstep = stepnice(lonmax - lonmin)
    latstep = stepnice(latmax - latmin)
    lonv = ceil(lonmin / lonstep) * lonstep
    while lonv <= lonmax + 1e-9
        px = ML + 1 + round(Int, (lonv - lonmin) * kx * S)
        lab = @sprintf("%.2f", lonv)
        lw = map_text_w(lab, 1)
        for t in 1:8   # ticks top+bottom
            img[MT-t, px, 1] = 0; img[MT-t, px, 2] = 0; img[MT-t, px, 3] = 0
            img[MT+mapH+1+t, px, 1] = 0; img[MT+mapH+1+t, px, 2] = 0
            img[MT+mapH+1+t, px, 3] = 0
        end
        map_draw_text!(img, px - div(lw, 2), MT - 24, lab, scale = 1)
        map_draw_text!(img, px - div(lw, 2), MT + mapH + 12, lab, scale = 1)
        lonv += lonstep
    end
    latv = ceil(latmin / latstep) * latstep
    while latv <= latmax + 1e-9
        py = MT + 1 + round(Int, (latmax - latv) * S)
        lab = @sprintf("%.2f", latv)
        lw = map_text_w(lab, 1)
        for t in 1:8
            img[py, ML-t, 1] = 0; img[py, ML-t, 2] = 0; img[py, ML-t, 3] = 0
            img[py, ML+mapW+1+t, 1] = 0; img[py, ML+mapW+1+t, 2] = 0
            img[py, ML+mapW+1+t, 3] = 0
        end
        map_draw_text!(img, ML - lw - 12, py - 3, lab, scale = 1)
        map_draw_text!(img, ML + mapW + 12, py - 3, lab, scale = 1)
        latv += latstep
    end

    # ---- colorbar --------------------------------------------------------
    bw = max(200, round(Int, 0.55 * mapW)); bh = 12
    bx = ML + div(mapW - bw, 2)
    by = MT + mapH + 40
    for px in 1:bw
        jc = map_jet((px - 0.5) / bw)
        for q in 1:bh
            img[by+q, bx+px, 1] = to8(jc[1]); img[by+q, bx+px, 2] = to8(jc[2])
            img[by+q, bx+px, 3] = to8(jc[3])
        end
    end
    for q in 0:bh+1   # bar frame
        img[by+q, bx, 1] = 0; img[by+q, bx, 2] = 0; img[by+q, bx, 3] = 0
        img[by+q, bx+bw+1, 1] = 0; img[by+q, bx+bw+1, 2] = 0
        img[by+q, bx+bw+1, 3] = 0
    end
    for px in 0:bw+1
        img[by, bx+px, 1] = 0; img[by, bx+px, 2] = 0; img[by, bx+px, 3] = 0
        img[by+bh+1, bx+px, 1] = 0; img[by+bh+1, bx+px, 2] = 0
        img[by+bh+1, bx+px, 3] = 0
    end
    for k in -2:2     # tick labels at -zs..+zs
        v = k * zs / 2
        px = bx + 1 + round(Int, (v + zs) / (2zs) * (bw - 1))
        lab = @sprintf("%d", round(Int, v))
        map_draw_text!(img, px - div(map_text_w(lab, 1), 2), by + bh + 6, lab, scale = 1)
    end
    cap = "LOS displacement [cm]"
    map_draw_text!(img, bx + div(bw - map_text_w(cap, 1), 2), by + bh + 22, cap, scale = 1)

    # ---- title + vent marker ---------------------------------------------
    map_draw_text!(img, ML + div(mapW - map_text_w(title, 2), 2), 16, title, scale = 2)
    if !isnan(vlat) && !isnan(vlon) &&
       latmin <= vlat <= latmax && lonmin <= vlon <= lonmax
        cx = ML + 1 + round(Int, (vlon - lonmin) * kx * S)
        cy = MT + 1 + round(Int, (latmax - vlat) * S)
        map_star!(img, Float64(cx), Float64(cy), 11.0)
        map_draw_text!(img, cx + 16, cy - 8, vname, scale = 2)
    end

    write_bmp(path, img)
end

"""Second-stage multilook: mean of a real field over azf x rgf blocks
(partial blocks at the edges); NaN values are skipped."""
function regrid_mean(v::Matrix{Float64}, azf::Int, rgf::Int)
    H, W = size(v)
    H2 = cld(H, azf); W2 = cld(W, rgf)
    out = Matrix{Float64}(undef, H2, W2)
    @inbounds for jj in 1:H2, ii in 1:W2
        j0 = (jj - 1) * azf + 1; j1 = min(j0 + azf - 1, H)
        i0 = (ii - 1) * rgf + 1; i1 = min(i0 + rgf - 1, W)
        s = 0.0; n = 0
        for j in j0:j1, i in i0:i1
            x = v[j, i]
            if !isnan(x); s += x; n += 1; end
        end
        out[jj, ii] = n > 0 ? s / n : NaN
    end
    return out
end

"""Second-stage multilook of a complex interferogram:
coherence-weighted mean over azf x rgf blocks; only `mask` pixels
contribute weight."""
function regrid_cplx(c::Matrix{ComplexF64}, coh::Matrix{Float64},
                     mask::AbstractMatrix{Bool}, azf::Int, rgf::Int)
    H, W = size(c)
    H2 = cld(H, azf); W2 = cld(W, rgf)
    out = Matrix{ComplexF64}(undef, H2, W2)
    @inbounds for jj in 1:H2, ii in 1:W2
        j0 = (jj - 1) * azf + 1; j1 = min(j0 + azf - 1, H)
        i0 = (ii - 1) * rgf + 1; i1 = min(i0 + rgf - 1, W)
        acc = 0.0 + 0.0im; ws = 0.0
        for j in j0:j1, i in i0:i1
            if mask[j, i]
                acc += coh[j, i] * c[j, i]; ws += coh[j, i]
            end
        end
        out[jj, ii] = ws > 0 ? acc / ws : 0.0 + 0.0im
    end
    return out
end

"""Coherence map render: every cell through the coherence colormap."""
function render_coh_img(coh::Matrix{Float64})
    H, W = size(coh)
    img = Array{UInt8}(undef, H, W, 3)
    @inbounds for j in 1:H, i in 1:W
        c = coh_cmap(clamp01(coh[j, i]))
        img[j, i, 1] = to8(c[1]); img[j, i, 2] = to8(c[2])
        img[j, i, 3] = to8(c[3])
    end
    return img
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
      s2              REAL Sentinel-2 satellite picture (ESA Copernicus
                      cloudless mosaic via the open EOX tile service, no
                      key): --place NAME or --lat/--lon; add --insar to
                      draw a simulated interferogram on top of the real
                      image
      raw             RAW Sentinel-1 SLC processing (v4): annotation parser,
                      burst timing, TOPS deramp/reramp machinery.
                        raw info    --ann XML [--tiff TIFF] [--scan]
                        raw selftest --ann XML --tiff TIFF [--burst N]
                                      [--l0 L] [--l1 L] [--p0 P] [--p1 P]
                        raw ifg     --ann XML --tiff TIFF [--ann2 XML --tiff2 X]
                                      [--sim] [--topo] [--mla A] [--mlr R]
                                      [--burst N] [--tdl D] [--tdp P] [--coh C]
                                      [--unwrap qg|wls]
                        raw pair    --lat F --lon F --date D1 --date2 D2
                                      [--dtol N] [--pass auto|desc|asc] [--topo]
                                      [--nlines N] [--npix N] [--datadir DIR]
                                      [--unwrap qg|wls]
                                      (end-to-end: CDSE catalogue search, ASF
                                      range-ZIP download (Earthdata token in
                                      earthdata.env or EARTHDATA_TOKEN), auto
                                      block clear of burst seams, pair
                                      interferogram; iw2-vv, cached in data/)
                        raw split   --desc DIR --asc DIR [--vlat F] [--vlon F]
                                      [--vname NAME] [--descgrid F] [--ascgrid F]
                                      (asc/desc LOS decomposition: reads the
                                      grid_*.txt sidecars of two --topo runs,
                                      solves up+east per cell assuming dN=0,
                                      writes split_vertical/east xyz + 2 maps)
                        raw dem     --ann XML --tiff TIFF [--burst N]
                                      [--l0 L] [--l1 L] [--p0 P] [--p1 P]
                                      [--mla A] [--mlr R] [--stride S]
                                      (geolocation + real DEM + coastline check)
                      (SLC + annotation from the Copernicus Data Space S3;
                      see README section "raw")
      real            REAL interferogram: downloads a genuine Sentinel-1
                      interferogram from the open COMET LiCSAR archive
                      (no key; default: the 2021 Fagradalsfjall dike,
                      Iceland), decodes the GeoTIFF products with the
                      built-in inflater/TIFF reader, unwraps the phase
                      with its own quality-guided unwrapper and renders
                      fringes, coherence and LOS displacement in cm
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
      --lat N --lon N   location for s2/real, decimal degrees
      --zoom Z        web-mercator zoom for s2 (default 11; ~76 m/px at
                      lat 37 for the 10 m Sentinel-2 mosaic)
      --tiles N|AxB  s2 tile grid, 256 px each (default 4 = 4x4)
      --year Y       s2cloudless mosaic year (2017-2023, default 2023)
      --insar        s2: also render a simulated interferogram on top
      --label NAME   s2/real: output filename label (default: place)
      --frameset F   real: LiCSAR frameset ID (expert; also needs --pair)
      --pair P       real: epoch pair YYYYMMDD_YYYYMMDD (default: the
                      scene's showcase pair)
      --crop D       real: crop half-width in degrees of latitude around
                      the target (default 0.2); --full for the whole frame
      --full         real: process the entire frame
      --unwref       real: also fetch the LiCSAR unwrapped reference and
                      verify our unwrapping against it (up to a constant)
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
      --coh F         simulator: baseline coherence 0..1 (default 0.9);
                      real: mask/unwrap threshold (default 0.15)
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
    if cmd ∉ ("demo", "terrain", "sar", "insar", "place", "s2", "real", "raw")
        println("unknown command '", cmd, "' -- try `help`")
        return
    end

    outdir = get(opts, "out", "insar_out")
    mkpath(outdir)
    seed = Int(num(opts, "seed", 1.0))
    clouds = !haskey(opts, "noclouds")
    println("insar_sat.jl -- mode: ", cmd, place == "" ? "" : " (" * place * ")")

    # ----------------------------------------------------- real Sentinel-2 --
    if cmd == "s2"
        lat = num(opts, "lat", NaN)
        lon = num(opts, "lon", NaN)
        label = get(opts, "label", "")
        if place != ""
            pi_info = PLACES[place]
            lat = pi_info.summit[1]
            lon = pi_info.summit[2]
            label == "" && (label = place)
        else
            (isnan(lat) || isnan(lon)) &&
                error("s2 needs --lat and --lon (or --place NAME)")
            label == "" && (label = string(round(lon, digits = 3)) * "_" *
                                   string(round(lat, digits = 3)))
        end
        year = Int(num(opts, "year", 2023.0))
        year in (2017, 2018, 2019, 2020, 2021, 2022, 2023) ||
            println("  warning: mosaic year ", year,
                    " may not exist (2017-2023); trying anyway")
        zoom = clamp(Int(num(opts, "zoom", 11.0)), 1, 17)
        tspec = get(opts, "tiles", "4")
        local nx::Int, ny::Int
        if occursin('x', tspec)
            a, b = split(tspec, 'x')
            nx = clamp(parse(Int, a), 1, 8)
            ny = clamp(parse(Int, b), 1, 8)
        else
            nx = ny = clamp(parse(Int, tspec), 1, 8)
        end
        println("  source: Sentinel-2 ", year, " cloudless mosaic (ESA via EOX)")
        println("  tiles ", nx, "x", ny, " @ zoom ", zoom)
        img, tx0, ty0 = s2_fetch_tiles(lat, lon, zoom, nx, ny, year, outdir)
        write_bmp(joinpath(outdir, "satellite_s2_" * label * ".bmp"), img)
        px = 156543.03392 * cosd(lat) / 2.0^zoom
        println("  image ", size(img, 2), "x", size(img, 1),
                " px, ~", round(px, digits = 1), " m/px at this latitude")
        if haskey(opts, "insar")
            source = get(opts, "source", "mogi")
            depth0 = place == "" ? 3500.0 : PLACES[place].depth
            srcpars = Dict{String,Float64}(
                "depth"  => num(opts, "depth", depth0),
                "strike" => num(opts, "strike", 35.0),
                "length" => num(opts, "length", 8000.0))
            H, W = size(img, 1), size(img, 2)
            local xc::Float64, yc::Float64
            if haskey(opts, "x") || haskey(opts, "y")
                xc = 1 + num(opts, "x", 0.5) * (W - 1)
                yc = 1 + num(opts, "y", 0.5) * (H - 1)
            elseif place != ""
                xc, yc = s2_pixel_of(PLACES[place].summit[1],
                                     PLACES[place].summit[2], zoom, tx0, ty0)
            else
                xc = W / 2; yc = H / 2
            end
            cfg = (source = source, srcpars = srcpars, xc = xc, yc = yc,
                   cm = num(opts, "cm", 14.0),
                   heading = num(opts, "heading", 193.0),
                   inc = num(opts, "inc", 35.0),
                   lam_cm = num(opts, "lambda", 5.55),
                   atm = num(opts, "atm", 0.35),
                   coh = num(opts, "coh", 0.9))
            println("  simulating InSAR over the real image: ", source,
                    " source, depth ", round(Int, srcpars["depth"]),
                    " m, max ", cfg.cm, " cm LOS")
            img_i, stats = render_insar_over(img, px, seed, cfg)
            write_bmp(joinpath(outdir, "interferogram_s2_" * label * ".bmp"),
                      img_i)
            @printf("  deformation: %.2f cm max LOS -> %.1f fringes, mean coherence %.3f\n",
                    stats.max_los_cm, stats.fringes, stats.mean_coh)
        end
        println("done.")
        return
    end

    # ------------------------------------------------- raw S1 SLC (v4) --
    if cmd == "raw"
        sub = length(pos) >= 2 ? pos[2] : "info"
        if sub == "pair"
            raw_pair_driver(opts, outdir)
            println("done.")
            return
        elseif sub == "split"
            raw_split(opts, outdir)
            println("done.")
            return
        end
        annpath = get(opts, "ann", "")
        isempty(annpath) && error("raw needs --ann <S1 annotation xml>")
        A = parse_s1_annotation(annpath)
        tp = get(opts, "tiff", "")
        T = isempty(tp) ? nothing : read_slc_tiff(tp)
        if sub == "info"
            raw_info(A, tp, T; scan = haskey(opts, "scan"))
        elseif sub == "selftest"
            T === nothing && error("raw selftest needs --tiff <slc measurement>")
            raw_selftest(A, tp, T;
                burst = Int(num(opts, "burst", length(A.bursts) - 1.0)),
                l0 = Int(num(opts, "l0", -1.0)),
                l1 = Int(num(opts, "l1", -1.0)),
                p0 = Int(num(opts, "p0", 14000.0)),
                p1 = Int(num(opts, "p1", 18000.0)))
        elseif sub == "ifg"
            T === nothing && error("raw ifg needs --tiff <slc measurement>")
            ann2p = get(opts, "ann2", "")
            tiff2p = get(opts, "tiff2", tp)
            A2 = isempty(ann2p) ? nothing : parse_s1_annotation(ann2p)
            T2 = isempty(tiff2p) ? nothing : read_slc_tiff(tiff2p)
            raw_ifg(A, tp, T;
                annS = A2, tiffS_path = tiff2p, tS = T2,
                burst = Int(num(opts, "burst", -1.0)),
                l0 = Int(num(opts, "l0", -1.0)),
                l1 = Int(num(opts, "l1", -1.0)),
                p0 = Int(num(opts, "p0", 14000.0)),
                p1 = Int(num(opts, "p1", 18000.0)),
                mla = Int(num(opts, "mla", 8.0)),
                mlr = Int(num(opts, "mlr", 4.0)),
                sim = haskey(opts, "sim"),
                simmax_cm = num(opts, "simmax", 6.0),
                test_dline = num(opts, "tdl", 0.0),
                test_dpx = num(opts, "tdp", 0.0),
                topo = haskey(opts, "topo"),
                cohthr = num(opts, "coh", 0.15),
                vlat = num(opts, "vlat", NaN),
                vlon = num(opts, "vlon", NaN),
                vname = get(opts, "vname", "Geldingadalur"),
                mtitle = get(opts, "mtitle", "LOS deformation"),
                unwr = get(opts, "unwrap", "wls"),
                outdir = outdir)
        elseif sub == "dem"
            T === nothing && error("raw dem needs --tiff <slc measurement>")
            raw_dem(A, tp, T;
                burst = Int(num(opts, "burst", -1.0)),
                l0 = Int(num(opts, "l0", -1.0)),
                l1 = Int(num(opts, "l1", -1.0)),
                p0 = Int(num(opts, "p0", 14000.0)),
                p1 = Int(num(opts, "p1", 18000.0)),
                mla = Int(num(opts, "mla", 8.0)),
                mlr = Int(num(opts, "mlr", 4.0)),
                stride = Int(num(opts, "stride", 4.0)),
                outdir = outdir)
        else
            println("raw: unknown subcommand '", sub, "' (info | selftest | ifg | pair | split | dem)")
        end
        println("done.")
        return
    end

    # -------------------------------------------------- real InSAR (LiCSAR) --
    if cmd == "real"
        scene = nothing
        if place != ""
            for sc in LICSAR_SCENES
                sc.place == place && (scene = sc)
            end
            scene === nothing && error("no real-InSAR scene for place '", place,
                "' -- available: ", join([sc.place for sc in LICSAR_SCENES], ", "))
        elseif haskey(opts, "frameset")
            fs = get(opts, "frameset", "")
            isempty(fs) && error("--frameset needs a LiCSAR frameset ID")
            isempty(get(opts, "pair", "")) &&
                error("--frameset also needs --pair YYYYMMDD_YYYYMMDD")
            scene = LiCSScene(fs, lics_trackdir(fs), -180.0, 180.0, -90.0, 90.0,
                              get(opts, "pair", ""), (NaN, NaN), "",
                              "custom frameset " * fs)
        else
            scene = LICSAR_SCENES[1]
        end
        pair = get(opts, "pair", scene.pair)
        latC, lonC = scene.site
        haskey(opts, "lat") && (latC = num(opts, "lat", latC))
        haskey(opts, "lon") && (lonC = num(opts, "lon", lonC))
        label = get(opts, "label", "")
        isempty(label) && (label = isempty(scene.place) ? scene.frameset : scene.place)
        cache = joinpath(outdir, "licsar_" * scene.frameset)
        mkpath(cache)
        println("  scene: ", scene.desc)
        println("  frameset ", scene.frameset, ", pair ", pair,
                " (COMET LiCSAR archive via CEDA, cached in ", cache, ")")
        phaf = joinpath(cache, pair * ".geo.diff_pha.tif")
        ccf  = joinpath(cache, pair * ".geo.cc.tif")
        lics_download(lics_url(scene, pair, "diff_pha.tif"), phaf)
        lics_download(lics_url(scene, pair, "cc.tif"), ccf)
        pha_head = tiff_head(phaf)
        cc_head = tiff_head(ccf)
        (pha_head.W == cc_head.W && pha_head.H == cc_head.H) ||
            error("phase / coherence grid mismatch")
        pha_head.geo === nothing && error("interferogram has no geo tags")
        geo = pha_head.geo
        W, H = pha_head.W, pha_head.H
        if haskey(opts, "full")
            y0, y1, x0, x1 = 1, H, 1, W
        else
            half = num(opts, "crop", 0.2)
            half > 0 || error("--crop must be > 0")
            isnan(latC) && error("--lat/--lon needed to centre the crop")
            halflon = half / max(cosd(latC), 0.05)
            y0 = max(1, Int(floor(1 + (geo.lat0 - (latC + half)) / (-geo.dlat))))
            y1 = min(H, Int(ceil(1 + (geo.lat0 - (latC - half)) / (-geo.dlat))))
            x0 = max(1, Int(floor(1 + (lonC - halflon - geo.lon0) / geo.dlon)))
            x1 = min(W, Int(ceil(1 + (lonC + halflon - geo.lon0) / geo.dlon)))
            (y1 >= y0 && x1 >= x0) || error("crop window outside the frame")
        end
        @printf("  frame %dx%d px, crop rows %d:%d cols %d:%d (%.2f-%.2f E, %.2f-%.2f N)\n",
                W, H, y0, y1, x0, x1,
                geo.lon0 + geo.dlon * (x0 - 1), geo.lon0 + geo.dlon * (x1 - 1),
                geo.lat0 + geo.dlat * (y1 - 1), geo.lat0 + geo.dlat * (y0 - 1))
        println("  decoding wrapped phase + coherence (pure-Julia inflate + GeoTIFF)...")
        pha = tiff_read(phaf, y0, y1, x0, x1)
        ccv = tiff_read(ccf, y0, y1, x0, x1)
        coh = ccv ./ 255.0
        pha .= wrapd.(pha)                    # defensive re-wrap into (-pi, pi]
        any(isnan, pha) && println("  warning: NaN pixels in the phase (masked out)")
        cohthr = num(opts, "coh", 0.15)
        lam_cm = num(opts, "lambda", 5.55)
        mask = (coh .>= cohthr) .& (.!isnan.(pha))
        count(mask) == 0 && error("no coherent pixels -- lower the --coh threshold")
        local unw
        if lowercase(get(opts, "unwrap", "wls")) == "wls"
            println("  unwrapping (weighted least squares: qg init + Poisson CG + L1 reweight)...")
            unw = unwrap_wls(pha, coh, mask)
        else
            println("  unwrapping (quality-guided flood fill, coherence-ordered)...")
            unw = unwrap_quality(pha, coh, mask)
        end
        frac = count(!isnan, unw) / count(mask)
        @printf("  unwrapped %d / %d coherent px (%.1f %%)\n",
                count(!isnan, unw), count(mask), 100frac)
        frac < 0.02 && println("  warning: almost nothing unwrapped -- coherence threshold too high?")
        img, imgc, imgd, dcm, stats = render_real_insar(pha, coh, unw, lam_cm, cohthr)
        write_bmp(joinpath(outdir, "interferogram_real_" * label * ".bmp"), img)
        write_bmp(joinpath(outdir, "coherence_real_" * label * ".bmp"), imgc)
        write_bmp(joinpath(outdir, "unwrapped_real_" * label * ".bmp"), imgd)
        Hc, Wc = size(pha)
        cy = (Hc + 1) ÷ 2; cx = (Wc + 1) ÷ 2
        dc = dcm[cy, cx]
        @printf("  deformation: %.2f cm max |LOS| = %.1f fringes (relative to the most coherent reference px), mean coherence %.3f\n",
                stats.max_cm, stats.max_cm / (lam_cm / 2), stats.mean_coh)
        if isnan(dc)
            println("  centre pixel not unwrapped (masked as incoherent)")
        else
            @printf("  LOS displacement at the crop centre: %+.2f cm (positive = toward satellite)\n",
                    dc)
        end
        if haskey(opts, "unwref")
            unwf = joinpath(cache, pair * ".geo.unw.tif")
            lics_download(lics_url(scene, pair, "unw.tif"), unwf)
            ref = tiff_read(unwf, y0, y1, x0, x1)
            diffs = Float64[]
            for k in eachindex(unw)
                if !isnan(unw[k]) && !isnan(ref[k])
                    push!(diffs, wrapd(unw[k] - ref[k]))
                end
            end
            if length(diffs) > 100
                s = sum(cos, diffs); t = sum(sin, diffs)
                R = hypot(s, t) / length(diffs)
                @printf("  check vs LiCSAR unwrapped: %d px agree up to a constant %.4f rad (concentration %.4f)\n",
                        length(diffs), atan(t, s), R)
                R > 0.999 && println("  -> our unwrapping reproduces the reference field")
            else
                println("  check vs LiCSAR unwrapped: too little overlap")
            end
        end
        println("done.")
        return
    end

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

# run only as a script (allows `include` for testing without side effects)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
