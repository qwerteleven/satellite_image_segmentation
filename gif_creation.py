import glob
from PIL import Image



def make_gif(frame_folder, output_name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.jpg")]
    frame_one = frames[0]
    frame_one.save(output_name, format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)
    


if __name__ == "__main__":

    folder = "gran_canaria"
    
    make_gif(folder, f"{folder}.gif")

    folder = "tenerife" 
    
    make_gif(folder, f"{folder}.gif")

    folder = "luna"
    
    make_gif(folder, f"{folder}.gif")


