# Depth Map Visualization Plugin for LichtFeld Studio

A plugin that creates a grayscale depth map for Gaussian Splats to allow render with "depth of field", Integrates with a standalone python utility
that allows adjustment of focal plane , depth of field and bokeh shape all in realtime.

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/c5e21e53-126e-4447-8467-e1a46d693098" />

<img width="2559" height="1536" alt="image" src="https://github.com/user-attachments/assets/4a365495-bed3-4ec9-99fd-c1fe78b592a2" />
3dGS thanks to dok11  - SuperSplat LINK (https://superspl.at/scene/67ba224d )   & ASZ20 for the idea in the first place.


IMPORTANT - KNOWN ISSUES:

1) Alpha transparency is not working at all  hiding the button for now
2) Image size must be a multiple of 4 (?) - otherwise they wont show in the compositor


## Installation (LichtFeld Studio v0.5.2+  This requires the LFS verison to have the Depth display)

In LichtFeld Studio:
1. Open the Plugins panel.
2. Enter: https://github.com/bgofish/LFS_DoF
3. Click Install.


## Usage
1. Requires an active 3dGSplat at the position you want to export an image
2. Open the "DoF" panel in the side panel.
3. Pick preset - Gray
4. If needed set a custom depth range:

Click to Pick 
- Click "Pick Point 1" or "Pick Point 2", then click directly on the model to set min/max depth values.
- You can click multiple times to adjust the point - each click updates the depth value.
- Click the "Stop Picking" button or press ESC or rightclick mouse to exit picking mode.
- Alternative Method using selection-based picking is also available

---

### Render Depth Video 
Renders a depth-map video along an existing keyframe camera path:

- Set **Output** folder, **Frames**, and **FPS**
- Click **Render Depth Video** — each frame applies the depth map at the correct interpolated camera position before rendering
- PNG frame outputs saved to two sub-folders `\DGS` (grayscales) & `\RGB` (Full colour) and `rgb_video.mp4` + `depth_video.mp4` are saved to the set output folder 
- [Open DoF Video] - Launches **Video-DoF_Bokeh** with both videos pre-loaded, ready to apply depth-of-field bokeh. Currently this can only be automtically opened once per LFS session.
- If further 'launches' are requred within one session then the python script can be opened, but this requires the 'required packages' to be installed.
  
<img width="2559" height="1534" alt="image" src="https://github.com/user-attachments/assets/f1ba1769-5bde-4b3a-be0d-1ab3bdc29984" />

###  ADDED BONUS - CUSTOM VIDEO EXPORT

For standard RBG Video (ie no depth information this tool can be used for highly customised output: MP4/MKV/MOV/AVI/WebM and any resolution,
  4 different compression levels Quality=L-M-H-VH  (approx only: 2,5,12,31 Mbps) & custom duration / FPS , Background HDRI can be loaded & rendered

<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/f89a3c2b-bf31-4c75-88dc-78965fc819b0" />

---

###  Export Viewport PNG — DoF Compositor

Click **Export**  to render Viewport PNG** :

1. Captures the viewport **with depth map OFF** → saves `VIEWPORT_DRGB.png` (colour image)
2. Captures the viewport **with depth map ON** → saves `VIEWPORT_DGSC.png` (grayscale depth)
3. Both files saved to the configured Output folder (`c:\temp` by default)

Click **[Open DoF Still]**  

Launches **Still-DoF_Bokeh** with both images pre-loaded, ready to apply depth-of-field bokeh. Currently this can only be automtically opened once per LFS session.

- If further 'launches' are requred within one session then the python script can be opened using [Run External Still Editor].  I use anaconda so this has been set as the default (but it will require the manual install of the required packages).

Resolution options: **Viewport**, **1080p**, **1440p**, **4K**, **8K**  

---


## Author
Brian Davis (bb6) — 2026  with code for picking and depth display from Jacob van  Beets

License: GPL-3.0-or-later
