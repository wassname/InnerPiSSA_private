# Instructions for Creating the Pizza Diagram

## Overview
Create an SVG diagram illustrating the geometric interpretation of InnerPiSSA contrastive steering loss. The diagram should be both technically accurate and memorable through subtle pizza aesthetics.

## Technical Requirements

### Space and Geometry
- **Coordinate system**: 2D projection of high-dimensional U-space (output singular vectors from SVD(W))
- **Origin**: Center point at (0, 250) in diagram coordinates
- **Axes**: U₁ (horizontal) and U₂ (vertical), standard orientation
- **Coherence boundary**: Circle centered at origin with radius ~180px, representing backprojection of logp constraint into U-space

### Key Vectors (all originate from origin)
1. **`pref_dir`** (blue, vertical):
   - Direction: Straight up (0°, along U₂ axis)
   - Length: ~160px (stays well within coherence boundary)
   - Style: Solid blue line, width 3px
   - Meaning: Frozen target direction extracted from mean(hs_ref_cho - hs_ref_rej)
   
2. **`pref_dir_ref`** (green, 45° slice):
   - Direction: 45° wedge centered on vertical (±22.5° from U₂)
   - Length: ~160px to edge
   - Style: Dashed green line (5,3), width 2.5px, with filled triangular region (opacity 0.15)
   - Meaning: Reference model separation at t=0
   - Note: Chicago deep dish has 8 slices = 45° each
   
3. **`pref_dir_pi`** (orange, narrower slice):
   - Direction: Narrower wedge (~30° total) extending toward pref_dir
   - Length: ~160px
   - Style: Solid orange line, width 2.5px, with filled triangular region (opacity 0.25)
   - Meaning: Adapter separation at t>0, started at pref_dir_ref position
   - Initial position marker: Small circle at (60, 90) with "t=0" label

### Projections
- **proj_ref**: Horizontal dotted line from pref_dir_ref tip to U₂ axis
- **proj_pi**: Horizontal dotted line from pref_dir_pi tip to U₂ axis
- Both projections should be at y=90 (same height as vector tips)
- Labels on left side of U₂ axis

### Training Indicator
- Curved arrow from initial position (60, 90) toward current position (30, 90)
- Label: "training (rotate V, scale S)"
- Color: Red (#dc2626)

### Coherence Boundary (Pizza Crust)
- Circle: radius 180px, centered at origin
- Style: Brown (#8b6f47), width 4px, dashed (8,5)
- Fill: Radial gradient from light cream center to darker tan edge
- Labels: "pizza crust" and "(inside = InnerPiSSA space)"

## Visual Style (Pizza Theme)

### Color Palette
- Background: Warm dough color `#fef9f3`
- Browns: `#8b6f47` (crust), `#5a3825` (dark text)
- Gradients: Cream to tan (`#fff8e7` → `#ffedc4` → `#d4a574`)
- Vectors: Blue `#2563eb`, Green `#10b981`, Orange `#f59e0b`, Red `#dc2626`

### Typography
- Primary font: `'Palatino Linotype', 'Book Antiqua', Palatino, serif`
- Style: Italic for most text (Italian menu aesthetic)
- Title: 22px, bold italic
- Labels: 13-14px for vectors, 10-11px for annotations
- Monospace for equation: `'Courier New', monospace`

### Pizza Elements
1. **Crust**: Brown dotted circle boundary
2. **Dough**: Warm cream background and gradients
3. **Slices**: Filled triangular wedges for separation vectors
4. **Angle**: 45° for reference slice (authentic Chicago deep dish)

## Layout

### Main Diagram (left, 600x400px)
- Transform: `translate(150, 150)`
- Contains: axes, coherence circle, all vectors, projections, labels

### Loss Equation Box (bottom, 700x120px)
- Transform: `translate(50, 450)`
- Background: Linear gradient (cream to tan)
- Border: Brown, 2.5px, rounded corners
- Content: Loss formula and two-line explanation

## Mathematical Accuracy

### Vector Relationships
- At t=0: `pref_dir_pi` = `pref_dir_ref` (adapter has zero effect initially)
- Training: Rotates `pref_dir_pi` toward `pref_dir` by adjusting V (rotation) and S (scaling)
- Constraint: All vectors must stay within coherence boundary (the crust)

### Projections
- Both `pref_dir_ref` and `pref_dir_pi` project onto `pref_dir` (vertical axis)
- Projection magnitudes shown as horizontal distances to U₂ axis
- Loss maximizes ratio: `proj_pi / proj_ref`

### Loss Formula
```
L = -proj_pi / proj_ref + coherence_penalty(log p_π, log p_ref)
```
- First term: Maximize projection ratio (steering effect)
- Second term: Penalize degradation beyond threshold (maintain quality)

## Review Checklist

### Technical Correctness
- [ ] All vectors originate from origin
- [ ] `pref_dir` is vertical (along U₂)
- [ ] `pref_dir_ref` is 45° wedge (Chicago deep dish standard)
- [ ] `pref_dir_pi` starts at `pref_dir_ref` position (marked with t=0 circle)
- [ ] `pref_dir_pi` is narrower than `pref_dir_ref` (rotating toward target)
- [ ] All vectors stay within coherence circle
- [ ] Projections are horizontal lines to U₂ axis at same height
- [ ] Axes labeled U₁, U₂ (not S₁, S₂ - we're in U-space)

### Visual Quality
- [ ] Pizza crust is brown and dotted (8,5 dash pattern)
- [ ] Background is warm cream/dough color
- [ ] Fonts are Palatino (Italian menu style)
- [ ] Most text is italic
- [ ] Gradients create depth (radial for crust, linear for box)
- [ ] Slice regions are filled with transparency
- [ ] Colors are warm and cohesive

### Clarity
- [ ] Title clearly states "Pizza Space"
- [ ] Labels distinguish frozen vs trainable vs reference
- [ ] Time indicators (t=0, t>0) show evolution
- [ ] Training arrow shows mechanism (rotate V, scale S)
- [ ] Warning about staying within crust
- [ ] Note about 2D projection of high-dimensional space
- [ ] Loss equation is readable and explained

### Pizza Aesthetics
- [ ] Subtle enough to be professional
- [ ] Memorable enough to be called "the pizza diagram"
- [ ] Coherence boundary looks like a crust
- [ ] Slices look like actual pizza slices
- [ ] Warm, inviting color scheme
- [ ] Not overly gimmicky

## Common Mistakes to Avoid
1. Don't make vectors extend outside the coherence circle
2. Don't use S₁, S₂ for axes (S is the diagonal scaling matrix, not the space)
3. Don't show `pref_dir_pi` starting from arbitrary position (it starts at `pref_dir_ref`)
4. Don't make slices too wide or too narrow (45° is authentic)
5. Don't use harsh colors (keep warm pizza tones)
6. Don't forget the filled triangular regions (they make it look like slices)
7. Don't make projections diagonal (they should be perpendicular to pref_dir)

## File Locations
- SVG: `docs/img/loss_geometry.svg`
- Documentation: `docs/loss_geometry.md`
- Related code: `repeng/train/inner_contrastive_loss.py`
