# 🎭 Custom Emoji Assets Guide

This guide shows you how to use your own high-quality emoji images with the face detection system.

## 📁 Asset Directory Structure

Create an `emoji_assets` folder in your project directory:

```
yolo-object-detection/
├── emoji_assets/           # 👈 Create this folder
│   ├── happy.png          # Your custom emojis
│   ├── cool.png
│   ├── love.png
│   ├── party.png
│   └── ...
├── face_emoji_custom.py   # Script that uses custom assets
└── ...
```

## 🖼️ Image Requirements

### File Format
- **Format**: PNG with transparency (RGBA)
- **Transparency**: Use alpha channel for smooth blending
- **Size**: Any size (will be automatically resized)
- **Recommended**: 200x200 pixels or higher for best quality

### Naming Convention
- Use descriptive names: `happy.png`, `cool.png`, `love.png`
- Avoid spaces in filenames
- The filename (without `.png`) becomes the emoji name in the app

## 🎨 Creating/Finding High-Quality Emojis

### Option 1: Download from Free Sources
- **OpenMoji**: https://openmoji.org/ (Open source emojis)
- **Twemoji**: https://twemoji.twitter.com/ (Twitter's emoji set)
- **Noto Emoji**: https://fonts.google.com/noto/specimen/Noto+Emoji

### Option 2: Create Your Own
- Use design tools like:
  - **Figma** (free, web-based)
  - **GIMP** (free, desktop)
  - **Adobe Illustrator/Photoshop**
  - **Canva** (online design tool)

### Option 3: Convert Existing Emojis
Use online tools to convert emoji to PNG:
- **Emoji to PNG converters**
- **Icon extraction tools**

## 📋 Step-by-Step Setup

### 1. Create the Assets Directory
```bash
mkdir emoji_assets
```

### 2. Add Your Emoji Files
Place your PNG files in the `emoji_assets` folder:
```bash
emoji_assets/
├── happy.png      # 😀 Happy face
├── cool.png       # 😎 Cool with sunglasses  
├── laughing.png   # 🤣 Laughing
├── heart_eyes.png # 😍 Heart eyes
├── thinking.png   # 🤔 Thinking
├── sleeping.png   # 😴 Sleeping
├── mind_blown.png # 🤯 Mind blown
├── party.png      # 🥳 Party
└── devil.png      # 😈 Devil
```

### 3. Run with Custom Assets
```bash
# Activate virtual environment
source .venv/bin/activate

# Run with custom emoji assets
python3 face_emoji_custom.py

# Or specify a different assets directory
python3 face_emoji_custom.py --assets-dir my_custom_emojis
```

## 🎮 Enhanced Controls

The custom emoji detector has additional controls:

- **`q`** - Quit application
- **`1-9`** - Switch between loaded emojis
- **`s`** - Save current frame
- **`r`** - Reset performance stats
- **`p`** - Print performance info
- **`h`** - Show emoji list with names

## 🔧 Advanced Features

### Automatic Fallback
If no custom assets are found, the script will automatically:
1. Create the `emoji_assets` directory
2. Generate high-quality sample emojis
3. Save them as PNG files for you to replace

### Quality Optimizations
- **High-quality interpolation** (INTER_CUBIC) for resizing
- **Gamma correction** for smoother alpha blending
- **Anti-aliased rendering** for better visual quality
- **Gradient effects** on auto-generated emojis

### Performance
- Assets are loaded once at startup
- Real-time resizing with optimal interpolation
- Efficient alpha blending for 30+ FPS performance

## 💡 Tips for Best Results

### Image Quality
- Use **high-resolution source images** (200x200 or larger)
- Ensure **clean transparency** around edges
- Avoid **compression artifacts** in PNG files
- Use **consistent style** across all emojis

### Design Considerations
- Make emojis **slightly larger** than the face area
- Use **bold, clear features** that are visible when scaled
- Consider **contrast** against various backgrounds
- Test with different **lighting conditions**

### Performance Tips
- Keep file sizes reasonable (< 1MB per emoji)
- Use optimized PNG compression
- Limit number of emojis to 9 for keyboard shortcuts

## 🛠️ Troubleshooting

### "No PNG files found" Error
```bash
# Check if directory exists and contains PNG files
ls -la emoji_assets/
```

### "Failed to load" Error
- Check if PNG files are valid
- Ensure files aren't corrupted
- Try re-saving with different tool

### Poor Quality Overlays
- Use higher resolution source images
- Check alpha channel transparency
- Verify PNG format (not JPG with .png extension)

### Performance Issues
- Reduce emoji file sizes
- Use fewer emojis
- Lower camera resolution if needed

## 📝 Example Asset Creation

### Using GIMP (Free)
1. Open GIMP
2. Create new image (200x200, fill with transparent)
3. Design your emoji
4. Export as PNG with alpha channel

### Using Online Tools
1. Visit openmoji.org or similar
2. Download SVG or PNG with transparency
3. Resize if needed (maintain aspect ratio)
4. Save to `emoji_assets/` folder

### Converting Existing Emojis
1. Copy emoji from web/messages
2. Paste into design tool
3. Export as PNG with transparent background
4. Save with descriptive name

## 🎉 Ready to Go!

Once you have your custom emoji assets in place:

```bash
source .venv/bin/activate
python3 face_emoji_custom.py
```

Your high-quality custom emojis will be loaded automatically and you can switch between them using the number keys 1-9!

---

**Pro Tip**: Start with a few high-quality emojis and add more as you find ones you like. The system is designed to handle any number of emoji assets! 🚀 