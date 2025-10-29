# Grayscale -> Color Photo Restorer

This small utility colorizes grayscale photos using the pre-trained colorization
model from Richard Zhang et al. On first run it will attempt to download the
required model files into a `models/` directory. If downloading fails the script
falls back to a colormap-based visualization.

Requirements
- Python 3.8+
- See `requirements.txt` for pip dependencies.

Usage

- Run a quick self-test (creates `test_input.png` and `test_output.png`):

```bash
python main.py --test
```

- Colorize an image:

```bash
python main.py --input path/to/grayscale.jpg --output path/to/colorized.png
```

Notes
- The script tries to download three files (prototxt, caffemodel, pts_in_hull.npy).
  The caffemodel is ~80MB.
- If you prefer not to download, the script will still produce a colored visualization
  using a fallback colormap (not a true restoration but often useful).

Files
- `main.py` — primary colorization script
- `requirements.txt` — Python dependencies

If you want, I can also add a small unit test or a GUI wrapper.
