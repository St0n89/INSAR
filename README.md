# Roundworld (insar_sat.jl) — satellite imagery & InSAR: simulations + real interferograms

*From the Discworld's point of view, Roundworld is the experiment —
this is the other one: a pure-Julia-stdlib satellite imagery & InSAR
laboratory that prefers real photons to hand-waving.*

Pure **Julia standard library** (Printf, Random, Downloads, LinearAlgebra).
No packages, no build step, no API key. One file. Runs on Julia ≥ 1.9.

## What it does

**Satellite pictures** — natural-colour renders of either
- a synthetic volcanic island (domain-warped fBm terrain, ridged mountains,
  biome coloring: water, beaches, farmland, forest, rock, snow; hillshading,
  aerial haze, optional cirrus clouds), or
- **real terrain** — elevation grids fetched from the free Open-Meteo
  elevation API (no key): Etna, Fuji, Matterhorn, Reykjanes, Grand Canyon,
  cached locally after the first fetch.

**InSAR fringe pictures** — simulated SAR interferograms:
- deformation sources: `mogi` (magma chamber), `dyke`, `sill`, `slip`
  (finite strike-slip fault), all projected onto the radar line-of-sight
  (configurable heading / incidence, default Sentinel-1 geometry)
- topographic phase from a perpendicular baseline (`--bperp`) and
  DEM-error fringes (`--demerr`)
- atmospheric delay: turbulent fBm screen + elevation-correlated stratified
  term + orbital ramp (`--atm`)
- coherence-dependent decorrelation noise (water incoherent, gradient loss),
  rendered as wrapped fringes with the classic cyclic InSAR rainbow
  (1 fringe = λ/2 LOS motion = 2.77 cm for C-band)
- SAR amplitude images (land-cover backscatter + Rayleigh speckle, 4 looks)

**Real interferograms** — the `real` command computes fringes from
genuinely *measured* radar phase:
- downloads a real Sentinel-1 interferogram from the open **COMET LiCSAR**
  archive (CEDA, free, no key); default scene: the **March 2021
  Fagradalsfjall dike intrusion**, Iceland — the interferogram that caught
  the Geldingadalur eruption in the act
- the products are float32 GeoTIFFs (Deflate + TIFF predictors) — decoded
  by a **hand-written RFC-1951 inflater and GeoTIFF reader in pure Julia**
  (validated byte-for-byte against reference decoders)
- the tool runs its **own quality-guided phase unwrapper**
  (coherence-ordered flood fill) and converts phase to LOS displacement in
  cm; with `--unwref` it also fetches the LiCSAR unwrapped reference and
  verifies the agreement (the Fagradalsfjall pair reproduces the reference
  field with concentration 1.0000)

Outputs are hand-written 24-bit **BMP** files (same writer style as
`newton_fractal.jl`): `satellite_*`, `sar_*`, `interferogram_*`,
`coherence_*`, `unwrapped_*`.

## Quick start

```bash
julia insar_sat.jl                  # demo: synthetic island, full suite
julia insar_sat.jl etna             # real DEM + summit uplift fringes
julia insar_sat.jl fuji             # ...same for Fuji etc.
julia insar_sat.jl real             # REAL interferogram (Fagradalsfjall 2021)
julia insar_sat.jl s2 --place etna --insar  # REAL Sentinel-2 + fringes
julia insar_sat.jl places           # list all real-data places
julia insar_sat.jl help             # all options
```

Recommended: `julia -t auto insar_sat.jl demo` (pixel loops are threaded).

## Examples

```bash
# earthquake instead of volcano: strike-slip M~7 pattern, 25 cm LOS
julia insar_sat.jl insar --source slip --strike 20 --length 9000 \
       --depth 6000 --cm 25

# topographic fringes from a 200 m perpendicular baseline + 5% DEM error
julia insar_sat.jl etna insar --bperp 200 --demerr 0.05 --atm 0

# clean magma chamber: no atmosphere, perfect coherence, ascending track
julia insar_sat.jl insar --atm 0 --coh 0.98 --heading 347 --cm 18

# Iceland dyke intrusion across the peninsula, grid refetch
julia insar_sat.jl reykjanes --source dyke --depth 2500 --cm 20
```

## Real Sentinel-2 imagery (`s2` command)

```bash
# real ESA Sentinel-2 pixels of any place on Earth — no API key
julia insar_sat.jl s2 --place etna --zoom 12 --tiles 4
# same, with a simulated interferogram drawn over the real image
julia insar_sat.jl s2 --place etna --zoom 12 --tiles 4 --insar --cm 18
# any coordinate, any year of the mosaic
julia insar_sat.jl s2 --lat 37.50 --lon 15.09 --zoom 12 --year 2021
```

- data: the **ESA Copernicus "Sentinel-2 cloudless" mosaic** (10 m/px
  base imagery) served as open WMTS tiles by EOX (`tiles.maps.eox.at`,
  free, no key, no account); mosaics exist for **2017–2023**
- the tiles are baseline JPEG (SOF0, 4:2:0, no restarts) — they are decoded
  by a **baseline JPEG decoder written in pure Julia stdlib** (Huffman,
  dequant, IDCT, YCbCr→RGB — `LinearAlgebra` is the only extra stdlib),
  so the tool keeps its zero-package promise; the decoder was validated
  byte-for-byte against an independent reference decoder
- `--zoom 11..13` ≈ 122..30 m/px (at lat 37.5), `--tiles N|AxB` up to
  8×8 tiles (2048 px); missing tiles are filled dark and reported
- `--insar` places a simulated source at the summit (per-place depth) or
  `--x/--y`/image centre and shades the cyclic rainbow by the real image
  luminance, so fringes stay readable over real terrain; no DEM → no
  topographic term and no stratified atmosphere, but turbulent + ramp
  atmosphere, coherence decay, and phase noise are modelled as usual

**Honesty note (s2):** this is real Sentinel-2 *pixels*, not per-scene
products. Raw Sentinel-2 scenes need the free **Copernicus Data Space
Ecosystem** account — out of scope for a stdlib tool.

## REAL interferograms (`real` command)

```bash
# the showcase: Fagradalsfjall dike, Iceland, 15-21 March 2021
julia insar_sat.jl real                          # ~44x50 km around the vent
julia insar_sat.jl real --crop 0.1               # zoom on the fringes
julia insar_sat.jl real --unwref                 # + verify vs LiCSAR unwrapped
julia insar_sat.jl real --full                   # the whole 250 km frame
# any LiCSAR frameset/pair (expert; see the COMET LiCSAR portal)
julia insar_sat.jl real --frameset 016A_02504_162118 \
       --pair 20210315_20210321
```

Outputs: `interferogram_real_*` (wrapped fringes, masked by real coherence),
`coherence_real_*`, `unwrapped_real_*` (LOS displacement in cm from our own
unwrapping, diverging blue→white→red).

- data: **COMET LiCSAR** products (LiCSBAS processing of real Sentinel-1
  SLCs by the University of Leeds) on the CEDA archive — free, no key:
  `https://comet.nerc.ac.uk/COMET-LiCS-portal/`
- the float32 GeoTIFFs use Deflate compression and the tricky TIFF
  *floating-point predictor*; both are decoded by pure-Julia code written
  from the RFC/libtiff sources and **validated byte-for-byte** against
  independent Python (zlib + hand-rolled) decoders on the real files
- unwrapping: **quality-guided** region growing (coherence-ordered flood
  fill, seed at the most coherent pixel); coherence below `--coh T`
  (default 0.15) is masked out; displacement d = −λφ/(4π), positive toward
  the satellite; values are relative to the reference pixel
- `--unwref` cross-checks our unwrapped field against the LiCSAR
  unwrapped product: agreement up to a single constant is expected (both
  are re-referenced); the Fagradalsfjall pair matches with circular
  concentration 1.0000 over 200k+ pixels
- products are cached in `out/licsar_<frameset>/`; only the crop rows are
  inflated (strips), so even `--full` stays single-pass
- grid: 0.001° pixels (~111 m EW, ~56-70 m NS here), geocoded WGS-84;
  geo tags provide the lon/lat mapping

**Honesty note (real):** the interferogram itself is produced by the
COMET LiCSAR pipeline from real SLCs — this tool decodes, unmasks,
**unwraps** (its own algorithm) and renders it. Computing the
interferogram from scratch (SLC focusing, coregistration, topographic
phase removal) needs the free **Copernicus Data Space Ecosystem**
account (`https://dataspace.copernicus.eu/`) plus ISCE/SNAP — out of
scope for a stdlib tool.

## RAW Sentinel-1 SLC processing (`raw` command, v4)

The `raw` command is the from-scratch route: real level-1 IW SLC data,
pure stdlib — a hand-written FFT, TIFF reader, TOPS burst model, GCP
coregistration, interferogram, coherence, unwrapping and deformation
statistics, all the way from the original product ZIPs.

    # annotation + measurement summary, land scan, burst timing check
    julia insar_sat.jl raw info --ann s1a-iw1-....xml --tiff s1a-iw1-....tiff --scan

    # TOPS machinery self-test on a burst block: measures the antenna
    # steering phase rate, deramps, checks the azimuth resampling roundtrip
    julia insar_sat.jl raw selftest --ann ... --tiff ... --burst 8 \
        --l0 12840 --l1 13490 --p0 14000 --p1 18000

    # the fully automated pair driver: catalog search -> ASF download
    # -> auto block -> coreg -> interferogram, fringes + deformation
    julia insar_sat.jl raw pair --lat 63.885 --lon -22.280 \
        --date 2021-03-07 --date2 2021-03-19 --topo

Implemented (all validated end-to-end on the Fagradalsfjall 2021 dike
intrusion): S1 annotation XML parser (burst timing, state vectors, GCP
grid, FM-rate/DC polynomials — line timing validated to 0.25 ms against
all 210 GCPs), stripped complex-int16 TIFF reader, hand-written radix-2
FFT (1e-14 vs naive DFT), TOPS burst timing model, deramp/shift/reramp
azimuth resampling, range resampling, range-Doppler geolocation,
GCP-based coregistration (incl. cross-satellite S1A/S1B pairs),
interferogram + coherence, GLO-90 topographic phase removal
(Open-Meteo elevation API, cached + resumable), statistical
unwrap, and a 2-D-quadratic deformation report in centimeters LOS.

`raw pair` needs a NASA EarthData bearer token for the ASF download
route: put `EARTHDATA_TOKEN=...` in `earthdata.env` next to the script
(or export the variable). That file is private — it is git-ignored,
never commit it.

Everything has a manual fallback: `raw info`, `raw selftest`, `raw ifg`
take explicit `--ann`/`--tiff`/block arguments when you already have
the files; `raw pair` is the one-command route for arbitrary
coordinates and dates (it picks the nearest same-track products).

## Real-data notes (Open-Meteo elevation API)

- free, no key; grids are fetched in 100-coordinate batches
- the anonymous endpoint is **minutely rate-limited**: the tool self-throttles
  (≈5 s between requests), pauses 65 s on HTTP 429, and caches the DEM as
  `out/dem_<place>.txt`
- fetches are **resumable**: on interruption (rate limit, `--maxsec T`
  budget, Ctrl-C + rerun) progress is kept in `dem_<place>.txt.part` and the
  same command continues where it left off
- `--grid N` controls fetch resolution (default 64 rows, aspect-correct
  width); the grid is bilinearly upsampled to `--size` for rendering
- `--fast` paces aggressively — only use it when you are not hitting the limit

## Physics & conventions (honesty notes)

- LOS unit vector points from ground toward the satellite; phase
  φ = 4π/λ · d_LOS, so **one color cycle = λ/2 ≈ 2.77 cm** of range change
- Mogi: `u_z = u0·d³/(r²+d²)^{3/2}`, `u_r = u0·d²·r/(r²+d²)^{3/2}` —
  standard point-source half-space solution
- dyke/sill are approximated by lines/grids of Mogi sources
- slip uses a buried screw-dislocation surface profile
  `u = (b/π)·atan(x/D)` with a finite-length taper — qualitatively correct
  quadrupole pattern, not Okada'85
- coherence: γ_eff = γ0 · exp(−|∇φ|/2.2), water = 0.04; phase noise
  σ = √((1−γ²)/(2Lγ²)) with L = 4 looks (standard CRLB for interferometric
  phase)
- the simulator's "unwrapped" product is the model displacement field,
  i.e. what a phase-unwrapping step should recover — but the `real`
  command performs an *actual* quality-guided phase unwrapping of
  measured phase, validated against the reference product

## Options

```
--size N        render size in px (default 900)
--seed N        noise seed
--out DIR       output directory (default insar_out)
--place NAME    real DEM instead of synthetic terrain
--grid N        fetched DEM grid height (default 64)
--maxsec T      fetch time budget; progress saved, rerun to resume
--fast          aggressive API pacing
--source S      mogi | dyke | sill | slip | none
--depth M       source depth [m]
--cm X          max LOS displacement [cm]
--x F --y F     source position (fraction of image)
--strike D      fault strike [deg] (slip)
--length M      fault half-length [m] (slip)
--heading D     satellite heading [deg] (193 descending / 347 ascending)
--inc D         incidence angle [deg] (default 35)
--lambda CM     wavelength [cm] (default 5.55, Sentinel-1 C-band)
--bperp M       perpendicular baseline [m] -> topographic fringes
--demerr F      DEM error fraction for bperp fringes
--atm F         atmospheric strength 0..1 (default 0.35)
--coh F         baseline coherence 0..1 (default 0.9)
--relief M      synthetic relief [m] (default 1900)
--waterfrac F   synthetic sea fraction (default 0.30)
--snowline M    override snowline elevation
--noclouds      disable cirrus overlay
--frameset F    real: LiCSAR frameset ID (expert; needs --pair)
--pair P         real: epoch pair YYYYMMDD_YYYYMMDD (default: showcase pair)
--crop D         real: crop half-width [deg latitude] (default 0.2)
--full           real: process the entire frame
--coh T          real: coherence mask/unwrap threshold (default 0.15)
--unwref         real: verify our unwrapping vs the LiCSAR reference
--zoom Z         s2: web-mercator zoom (default 11)
--tiles N|AxB    s2: tile grid (default 4)
--year Y         s2: mosaic year (2017-2023, default 2023)
--label NAME     s2/real: filename label (default: place)
```

BMP → PNG conversion (the tool intentionally stays stdlib-only):
`magick out/*.bmp out/%d.png`, or ImageMagick-free:
`python3 -c "from PIL import Image; [Image.open(f).save(f[:-4]+'.png') for f in __import__('glob').glob('out/*.bmp')]"`

---

<p align="center"><em>“The universe, they said, depends for its operation on ALL of its rules.”<br>
— Roundworld is a world in a bottle, and so is this one.</em></p>

**GNU Terry Pratchett.**
