#--------------------------------------------------------------#
#   对单张图片进行预测，运行结果保存在根目录
#   默认保存文件为results/predict_out/predict_srgan.png
#--------------------------------------------------------------#

from PIL import Image
from srgan import SRGAN


if __name__ == "__main__":
    srgan       = SRGAN()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'       表示单张图片预测，结果保存在save_path_1x1中
    
    #   'dir_predict'   表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    # mode            = "predict"
    mode            = "dir_predict"
    #-------------------------------------------------------------------------#
    #   save_path_1x1   单张图片的保存路径
    #-------------------------------------------------------------------------#
    save_path_1x1   = "results/predict_out/predict_srgan.png"
    #----------------------------------------------------------------------------------------------------------#
    
    
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = srgan.detect_image(image)
                r_image.save(save_path_1x1)
                r_image.show()
        
    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = srgan.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("模式输入错误！")
