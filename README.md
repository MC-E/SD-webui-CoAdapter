# SD-Webui Extension
This repo is a SD-Webui extension for Composable T2I-Adapter ([CoAdapter](https://github.com/TencentARC/T2I-Adapter)).

## 🔧 Install
- Install the [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Open "Extensions" tab.
- Open "Install from URL" tab in the tab.
- Enter URL of this repo to "URL for extension's git repository".
- Press "Install" button.
- Reload/Restart Web UI.

Once installed, the UI looks like:
<!-- <div align="center"> -->
<p align="center">
  <img src="assets/webui.PNG">
</p>

**Please check ``Enable'' box to activate the function of CoAdapter.**


## Demos

| Sketch                                                                                                                                    | Canny |                                                                   Depth                                                                   | Color (Spatial) | Style                                                                                                                                      | Results |
|:------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|--------------------------------------------------------------------------------------------------------------------------------------------|---------|
|  |      <img width="200" alt="image" src="assets/1_canny.png"> |                                                                                                                                           |                 | <img width="100" alt="image" src="assets/1_style.png"> |    <img width="200" alt="image" src="assets/1_res.png">     |
|  | <img width="200" alt="image" src="assets/1_canny.png">       |                                                                                                                                           |                 | <img width="150" alt="image" src="assets/2_style.png">  |    <img width="200" alt="image" src="assets/2_res.png">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661058-656d87d7-3c8d-4216-820e-a02a8a5f5a4a.png"> |       |  |                 | <img width="250" alt="image" src="https://user-images.githubusercontent.com/11482921/225661180-98f338ee-950e-45d0-bd5f-4e8b7e82cecb.png">  |    <img width="200" alt="image" src="assets/3_res.png">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661058-656d87d7-3c8d-4216-820e-a02a8a5f5a4a.png"> |       |  |            <img width="250" alt="image" src="assets/4_color.png">     | <img width="250" alt="image" src="https://user-images.githubusercontent.com/11482921/225661180-98f338ee-950e-45d0-bd5f-4e8b7e82cecb.png">  |    <img width="200" alt="image" src="assets/4_res.png">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661058-656d87d7-3c8d-4216-820e-a02a8a5f5a4a.png"> |       |  <img width="250" alt="image" src="assets/5_depth.png">|                | <img width="250" alt="image" src="https://user-images.githubusercontent.com/11482921/225661180-98f338ee-950e-45d0-bd5f-4e8b7e82cecb.png">  |    <img width="200" alt="image" src="assets/5_res.png">     |

## 🤗 Acknowledgements
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [SD WebUI extension for ControlNet and T2I-Adapter](https://github.com/Mikubill/sd-webui-controlnet)
