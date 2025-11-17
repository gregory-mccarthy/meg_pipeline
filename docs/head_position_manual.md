# Head Position Processing in the MEG Preprocessing Pipeline  
### (Centering, cHPI Pose Estimation, Movement Compensation, and YAML Configuration)

## 1. The Overall Goal

Head-position processing in MEG has **two distinct purposes**, which are often conflated:

1. **Estimating the subject’s pose over time**  
   – derived from **continuous HPI coils (cHPI)**.  
   – yields time-varying translations + rotations of the head relative to the sensors.  
   – used **only** for *movement compensation* in Maxwell filtering.

2. **Estimating the head center (SSS “origin”) in the HEAD frame**  
   – derived from **digitized head-shape points** (Polhemus) in the *head* coordinate system.  
   – used for **spherical harmonic expansion** of the magnetic field during Maxwell filtering.  
   – must be stable across runs if runs are to be combined.

These two products — **pose** and **center** — serve fundamentally different mathematical roles in SSS/tSSS and must not be mixed up.

Your pipeline separates the two clearly:

- **cHPI pose → head_pos array or .pos file**  
- **head center → sphere origin fitted from digitization**  

Both are required for robust, reproducible Maxwell filtering.

---

## 2. Why Head Position Matters in MEG

Magnetometers and gradiometers measure the magnetic field in a fixed coordinate frame attached to the dewar.  
The subject, however, **moves inside the helmet**.

MEGIN’s continuous HPI system embeds five (or more) small coils oscillating at fixed frequencies (≈293–321 Hz). These act as *fiducials* in continuous time. MNE:

1. Extracts each coil’s amplitude and phase.  
2. Solves a nonlinear localization problem to track each coil in 3D.  
3. Computes the best-fit rigid-body transform to map coil positions into a **6-DOF pose** (x, y, z, yaw, pitch, roll).  
4. Produces a **t × 7** matrix  
   ```
   [ time, tx, ty, tz, rx, ry, rz ]
   ```

This matrix is what SSS uses for **movement compensation**.

Meanwhile, the **SSS origin** determines the *mathematical center of the spherical model* used to expand internal and external field components. It is **not** derived from cHPI. It comes from Polhemus digitization.

---

## 3. Coordinate Frames: Device, Head, and Meaning of Each

MNE must move cleanly between:

- **Device frame**: fixed to the MEG helmet (SQUIDs).  
- **Head frame**: fixed to the subject, defined by nasion + LPA + RPA digitization.  
- **Coil frame**: location of each cHPI coil relative to the head.

The crucial facts:

| Quantity | Frame | Source | Purpose |
|---------|-------|--------|---------|
| **Head center / SSS origin** | HEAD | Polhemus digitization | Defines the sphere for SSS math |
| **cHPI coil positions** | DEVICE | MEG sensors | Tracks movement |
| **Rigid transform (DEVICE→HEAD)** | Both | Derived from fiducials | Aligns spaces |
| **head_pos time series** | DEVICE? (time) | cHPI | Movement compensation only |

If the **origin** is wrong → SSS is mathematically incorrect, leading to:  
- distorted fields  
- inflated amplitudes  
- failure of shielding separation  
- downstream bad epoch classification (as you observed)

If the **pose** is wrong or misaligned with the cropped time window →  
- tSSS fails to apply movement correction properly  
- large distortions occur in gradiometers  
- autoreject sees the data as nonsense

Your refactor makes pose and origin independent and explicit.

---

## 4. The Two Products You Must Get Right

### 4.1 Head-Movement Time Series (`head_pos`)

This is the **array or .pos file**:

```
time, tx, ty, tz, rx, ry, rz
```

- **time is in absolute seconds**, usually matching raw.first_time.  
- Rotations are in radians.  
- Must be clipped to the same window as the Raw object.  
- If cropped incorrectly, movement compensation in SSS becomes garbage.

### 4.2 Head Center (“SSS Origin”)

Derived from digitized head-shape points:

- Polhemus captures EXTRA + CARDINAL points in the HEAD frame.  
- `fit_sphere_to_headshape(info)` returns a sphere center in **HEAD** coordinates.  
- This is fed to `origin=` in `maxwell_filter`.  
- Should **not vary across runs** for a given subject/session.

Your old `apply_maxwell_filter()` used this correctly.

---

## 5. Why Sub-windowed Data Is Hard

When you analyze **only a 10-minute window** from a 60-minute recording:

- `raw.first_time` shifts (e.g., from 0 to ~643.5 s).  
- The `.pos` file still spans 0–3600 s.  
- So your pipeline must **clip and realign** the head-pos time axis.  
- If this step is even slightly wrong → SSS sees mismatched times → fields explode → autoreject flags everything.

Your new `align_headpos_to_cropped_raw()` resolves this, provided the origin is still fit correctly.

---

## 6. Using Multiple Runs and Consistent Centers

If you want **Run 1, Run 2, …** to be processed in a comparable way:

- Use the **same SSS origin** for all runs (usually median or fitted origin).  
- Optionally use the **same destination coordinate** (e.g., median pose of Run 1).  
- cHPI poses may be taken from different runs (`source="run"` in YAML) if needed.

Consistent origins → better between-run alignment  
Consistent destinations → stable transforms for movement compensation  
Consistent clipping → no timing mismatches

---

## 7. YAML Parameter Reference

Below is the updated block used in your pipeline.

```yaml
head_position_processing:

  # ------------------------
  # HOW TO GET HEAD-POSE DATA
  # ------------------------
  source: "compute"              # "compute" | "file" | "coordinates" | "run" | "none"

  # If source="file"
  file_path: "/path/to/headpos.pos"

  # If source="coordinates" (no movement comp)
  coordinates: [-0.075, 0.012, 0.030]

  # If source="run" (borrow pose from run N)
  reference_run: 1

  # ------------------------
  # HOW TO CHOOSE DESTINATION
  # ------------------------
  destination: "median"          # "median" | "mean" | "first" | "last" | "coordinates" | "reference" | "none"

  # destination="coordinates"
  destination_coordinates: [-0.075, 0.012, 0.030]

  # destination="reference"
  destination_reference_run: 1

  # ------------------------
  # MOVEMENT COMPENSATION
  # ------------------------
  movement_compensation: true    # true = use head_pos; false = static maxwell

  # ------------------------
  # ADVANCED OPTIONS
  # ------------------------
  force_recompute: false         # ignore existing .pos file
  write_subset: true             # write clipped, windowed .pos file
  subset_naming: "desc-crop"     # filename tag for cropped .pos
```

---

## 8. Parameter Interpretation and When to Use Each

### 8.1 `source: compute`
Compute cHPI pose **directly from the current raw segment**.  
Best when:
- you want movement compensation tailored to the analysis window  
- no .pos file exists  
- the cHPI coils are stable

### 8.2 `source: file`
Load an existing `.pos`.  
Best when:
- you already computed pose for the *full run*  
- you want reproducible analysis across subsets  
- computing pose is very slow (1+ hour for 60 minutes)

### 8.3 `source: run`
Useful when:
- analyzing multiple runs  
- you want them aligned to a common pose reference (Run 1)

### 8.4 `source: coordinates`
No movement compensation; just use a static origin.  
Rare, but appropriate when:
- cHPI failed  
- subject barely moved

### 8.5 `destination` options
Controls the **target pose** used when aligning SSS before spatial filtering.

- `"median"` → best all-purpose choice  
- `"first"` → stable if early posture matters  
- `"coordinates"` → for reproducible research across sessions  
- `"none"` → disables movement compensation

### 8.6 `movement_compensation`
- `true` → use full tSSS + dynamic transforms  
- `false` → SSS only, no movement paths

Turn this off only when cHPI data are bad.

### 8.7 `write_subset`
Writes a clipped `.pos` file matching the analysis window.  
This is extremely helpful for debug and documentation.

---

## 9. Practical Workflow Examples

### 9.1 Full-run preprocessing (1 hour)
```yaml
source: "compute"
destination: "median"
movement_compensation: true
```

### 9.2 Analyze only a 10-min window, but reuse full-run head pose
```yaml
source: "file"
file_path: "..._headpos.pos"
destination: "median"
movement_compensation: true
```

### 9.3 Multi-run experiment with aligned centers
```yaml
source: "run"
reference_run: 1
destination: "reference"
destination_reference_run: 1
movement_compensation: true
```

### 9.4 No cHPI available (rare)
```yaml
source: "coordinates"
coordinates: [-0.07, 0.02, 0.03]
movement_compensation: false
```

---

## 10. Conceptual Summary

| Concept | Description |
|--------|-------------|
| **cHPI-derived pose** | Time-varying head movement; needed for dynamic SSS. |
| **SSS origin (head center)** | Static geometric model fitted from Polhemus points. |
| **Destination pose** | Where to “align” the head before harmonic expansion. |
| **Movement compensation** | Corrects MEG data for within-run motion. |
| **Cropping/head-pos realignment** | Ensures pose and data use matching absolute time bases. |
| **Multiple-run alignment** | Ensures shared coordinate frame across experimental runs. |

**Key principle:**  
*Head movement and head center are mathematically independent.  
SSS requires both of them to be correct and in the right frame.*  

With correctly clipped head-pos arrays and consistent SSS origins, Maxwell filtering becomes stable, reproducible, and interpretable.