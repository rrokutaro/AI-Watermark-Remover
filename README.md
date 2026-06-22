# AI Image Watermark Remover

**Easy tool to remove invisible AI watermarks from photos.**

![](./assets/main.jpg)
![](./assets/banner-1.png)
![](./assets/banner-2.png)
![](./assets/banner-3.jpg)

### What it does
Removes hidden AI watermarks while protecting important parts like faces, hair, clothes, hands, text, and shoes etc.

### How to Use (Super Simple)
1. Run the program — it opens in your web browser.
2. Upload one or more photos.
3. Use the sliders to adjust cleaning strength (Balanced preset is a good start).
4. Click **Process batch**.
5. Download your cleaned images.

**Tip**: Set **Text strength to 0** for perfectly clear text.

### Launch
```bash
# SETUP
pip install noai-watermark ultralytics gradio Pillow numpy fashn-human-parser
pip install torch torchvision open-clip-torch einops timm
pip install codeformer-inference easyocr gradio_client

# Launch
python3 remover.py
```

---

### Credits
- **Core engine**: Based on [noai-watermark](https://github.com/mertizci/noai-watermark) by mertizci
- **Clothing & body detection**: [FASHN Human Parser](https://github.com/fashn-AI/fashn-human-parser)
- **Detail enhancement**: [SUPIR](https://github.com/Fanghua-Yu/SUPIR) + CodeFormer
- **Other tools**: YOLO (Ultralytics), EasyOCR, Gradio

