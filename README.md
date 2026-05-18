# Depth Map Visualization Plugin for LichtFeld Studio

A plugin that creates a greyscale depth map for Gaussian Splats to allow render with "depth of field", Integrates with a standalone python utility
that allows adjustment of focal plane , depth of field and bokeh shape all in realtime.

<img width="2559" height="1536" alt="image" src="https://github.com/user-attachments/assets/4a365495-bed3-4ec9-99fd-c1fe78b592a2" />
3dGS thanks to dok11  - SuperSplat LINK (https://superspl.at/scene/67ba224d ) 

## Installation (LichtFeld Studio v0.5+)

In LichtFeld Studio:
1. Open the Plugins panel.
2. Enter: https://github.com/bgofish/LFS_DoF
3. Click Install.


## Usage
1. Requires an active 3dGSplat at the position you want to export an image
2. Open the "DoF" panel in the side panel.
3. Pick Cam-O preset - Grayscale colormap (Jet, Turbo, Viridis have been retained for better visual identification of features ) and camera axis radial (P or parallel is also available).
4. If needed set custom depth range using one of two methods:

### Method 1: Click to Pick (Default)
- Click "Pick Point 1" or "Pick Point 2", then click directly on the model to set min/max depth values.
- You can click multiple times to adjust the point - each click updates the depth value.
- Click the "Stop Picking" button or press ESC to exit picking mode.

### Method 2: Selection-Based (Old Method)
Useful when XYZ picking coordinates are unreliable:
1. Check "Use Selection Method (old)" in the Depth Range section.
2. Use the Splat Select tool to select gaussians at your desired min depth location.
3. Click "Set Point 1 from Selection".
4. Select gaussians at your desired max depth location.
5. Click "Set Point 2 from Selection".

4. Fine-tune the depth values with the +/- buttons or input fields.
5. Enable "Live Preview" to see changes in real-time.
6. Use "Restore Original" to revert to original colors.

---

### Render Depth Video (WIP)
Renders a depth-map video along an existing keyframe camera path:

- Set **Output** folder, **Frames**, and **FPS**
- Click **Render Depth Video** — each frame applies the depth map at the correct interpolated camera position before rendering
- Output saved as `frame_NNNN.png` + `depth_video.mp4` (requires `ffmpeg` on PATH) - To be fixed
- need to have manual load of the python /python/Depth_Blur7.py  (temp only need fixing)

###  Export Viewport PNG — DoF Compositor

Click **Export Viewport PNG** to:

1. Capture the viewport **with depth map OFF** → saves `VIEWPORT_DRGB.png` (colour image)
2. Capture the viewport **with depth map ON** → saves `VIEWPORT_DGSC.png` (greyscale depth)
3. Both files saved to the configured Output folder (`c:\temp` by default)
4. Launches **Still-DoF_Bokeh** with both images pre-loaded, ready to apply depth-of-field bokeh. Currently this can only be automtically opened once per LFS session.

Resolution options: **Viewport**, **1080p**, **1440p**, **4K**, **8K**  

## Usage Information

### 1. Select a Splat
The panel header shows the active target. Select a trained Gaussian Splat model in the viewport first.

### 2. Enable & Configure

| Control | Description |
|---|---|
| **Depth Map: ON/OFF** | Applies or removes the depth colourmap from the splat |
| **Live Preview: ON/OFF** | Auto-updates the depth map as you change settings or move the camera |

### 3. Colourmap
Choose from: **Jet**, **Gray**, **Turbo**, **Viridis**,  Toggle **Invert** to flip the gradient.

### 4. Depth Axis

| Button | Description |
|---|---|
| **Z / Y / X** | World-space axis depth |
| **Cam P** | Camera projective (flat parallel bands) — Greyscale |
| **Cam O** | Camera radial/spherical — Greyscale |

Selecting **Cam P** or **Cam O** automatically switches to Greyscale.

### 5. Depth Range
Set the min/max depth clipping range by clicking directly on the model:

1. Click **Pick Point 1** — click on the model at the nearest depth position
2. Click **Pick Point 2** — click on the model at the furthest depth position
3. Fine-tune with the **+/−** nudge buttons or type values directly

Press **ESC** or right-click to cancel picking.

### 6. Presets
One-click combinations of colourmap + axis:

| Preset | Colourmap | Axis |
|---|---|---|
| Jet Z | Jet | Z |
| Gray Z | Greyscale | Z |
| Turbo Z | Turbo | Z |
| Cam P | Greyscale | Camera Projective |
| Cam O | Greyscale | Camera Radial |

---

## Author
Brian Davis (bb6) — 2026  with code for picking and depth display from Jacob van  Beets
License: GPL-3.0-or-later
