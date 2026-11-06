# insar_sat.jl — satellite imagery & InSAR fringe simulator

Pure **Julia standard library** (Printf, Random, Downloads). No packages, no
build step, no API key. One file. Runs on Julia ≥ 1.9.

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

Outputs are hand-written 24-bit **BMP** files (same writer style as
`newton_fractal.jl`): `satellite_*`, `sar_*`, `interferogram_*`,
`coherence_*`, `unwrapped_*`.

## Quick start

```bash
julia insar_sat.jl                  # demo: synthetic island, full suite
julia insar_sat.jl etna             # real DEM + summit uplift fringes
julia insar_sat.jl fuji             # ...same for Fuji etc.
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
- the "unwrapped" product is the model displacement field, i.e. what a
  phase-unwrapping step should recover — not an actual unwrapping algorithm

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
```

BMP → PNG conversion (the tool intentionally stays stdlib-only):
`magick out/*.bmp out/%d.png`, or ImageMagick-free:
`python3 -c "from PIL import Image; [Image.open(f).save(f[:-4]+'.png') for f in __import__('glob').glob('out/*.bmp')]"`
