# About
Neural Style Transfer Algorithm using VGG-19.
Transfers style from Style Image to Content Image using white noise as input image. Additional input parameter in NeuralST.py can be modified to use the Content Image as input image.

<img align="middle" alt="Python" width="800px" src="NSTchart.svg" />

# Usage
1. Install dependencies `pip install -r requirements.txt`
2. Save desired Content Image to `content_images/` folder and Style Image to `style_images/` folder. 
3. Run `python3 main.py model img_dims content_img style_img`. Example: `python3 main.py VGG 512 bodybuilder.jpg the_scream.jpg`
4. Final result will be saved in `results/` folder.

Currently the only model available is VGG-19 (Version using ResNet coming soon) and `img_dims` refers to the output image dimensions (1:1 aspect ratio) in pixels. 
⚠️ Beware of memory usage ⚠️.
