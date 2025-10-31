from backend.backend.ImageRestoration_Backend import colorizer

model_path = r"D:\2pj\2pj\humanImageRestorationProject\backend\backend\ImageRestoration_Backend\colorizer.ckpt"
model = colorizer.load_colorizer(model_path)
input_path=r"c:\Users\human\Pictures\bacd1a09-11ee-433c-b3f9-05dce14d69ee.jpg"
output_dir = r"c:\Users\human\Pictures\test"
colorizer.colorize_image(model, input_path, output_dir)