from ultralytics import YOLO
import cv2
import csv
import os
import numpy as np
from PIL import Image, ExifTags
model = YOLO("best_v3_yolov8.pt")
def check_width_bound(x,w,ow,h,desired_ratio):
    w = ow
    excess_width = abs(x+w - w)
    x = int(x - excess_width*0.5)
    w = int(w - excess_width*0.5)
    current_ratio = h/w
    if(current_ratio > desired_ratio):  
        h = int(w * desired_ratio)
    return x,w,h
        
def check_height_bound(y,w,h,oh,desired_ratio):
    h = oh
    excess_height = abs(y+h - h)
    y = int(y - excess_height*0.5)
    h = int(h - excess_height*0.5)
    current_ratio = h/w
    if current_ratio > desired_ratio:
        h = int(w * desired_ratio)   
    else:
        h = oh
        excess_height = abs(y+h - h)
        y = int(y - excess_height*0.6)
        h = int(h - excess_height*0.4)
        w = int(h / desired_ratio) 
    return y,w,h

    
def fix_ratio(x,y,wi,hi,desired_ratio,oh,ow):
    current_ratio = round(hi/wi,2)
    if(current_ratio > desired_ratio):
        # print("Hereid the ratio",current_ratio)
        w = int(hi/desired_ratio)
        h = hi
        if x+w > ow:
            x,w,h = check_width_bound(x,w,ow,h,desired_ratio)
        if y+h > oh:
            y,w,h =check_height_bound(y,w,h,oh,desired_ratio)   
    else:
        h = int(wi*desired_ratio)
        w=wi
        if y+h > oh:
            y,w,h = check_height_bound(y,w,h,oh,desired_ratio)                       
        if x+w > ow:
            x,w,h= check_width_bound(x,w,ow,h,desired_ratio)        
    return(x,y,w,h)

# def best_crop(img):
#     results = model(img)
#     boxes = results[0].boxes
#     crops = []
#     for box in boxes:
#         class_0_prob = box.cls == 0
#         class_1_prob = box.cls == 1
#         if class_0_prob and box.conf[0].item() > 0.6:
#             xywh_tensor = box.xywh
#             xywh_tensor = xywh_tensor.cpu()
#             numpy_array = xywh_tensor.numpy()
#             python_list = numpy_array.tolist()
#             x = int(python_list[0][0] - ((python_list[0][2]) / 2))
#             y = int(python_list[0][1] - ((python_list[0][3]) / 2))
#             w = int(python_list[0][2])
#             h = int(python_list[0][3])
#             ow, oh = img.size
#             # oh,ow,_ = img.shape
#             ratio = 1.25
#             x,y,w,h = fix_ratio(x,y,w,h,ratio,oh,ow)
#             # crop_img = img[y:y + h, x:x + w]
#             crop_img = img.crop((x, y, x + w, y + h))
#             crops.append(crop_img)
            
#         elif class_1_prob and len(box.conf) > 0 and box.conf[0].item() > 0.6:
#             xywh_tensor = box.xywh
#             xywh_tensor = xywh_tensor.cpu()
#             numpy_array = xywh_tensor.numpy()
#             python_list = numpy_array.tolist()
#             x = int(python_list[0][0] - ((python_list[0][2]) / 2))
#             y = int(python_list[0][1] - ((python_list[0][3]) / 2))
#             w = int(python_list[0][2])
#             h = int(python_list[0][3])
#             ow, oh = img.size
#             # oh,ow,_ = img.shape
#             ratio = 0.8
#             x,y,w,h = fix_ratio(x,y,w,h,ratio,oh,ow)
#             # crop_img = img[y:y + h, x:x + w]
#             crop_img = img.crop((x, y, x + w, y + h))
#             crops.append(crop_img)
#     return crops



def best_crop(img):
    results = model(img)
    boxes = results[0].boxes
    best_crop_info = None
    highest_probability = 0.0

    for box in boxes:
        class_0_prob = box.cls == 0
        class_1_prob = box.cls == 1

        if class_0_prob and box.conf[0].item() > 0.6:
            xywh_tensor = box.xywh
            xywh_tensor = xywh_tensor.cpu()
            numpy_array = xywh_tensor.numpy()
            python_list = numpy_array.tolist()
            x = int(python_list[0][0] - ((python_list[0][2]) / 2))
            y = int(python_list[0][1] - ((python_list[0][3]) / 2))
            w = int(python_list[0][2])
            h = int(python_list[0][3])
            ow, oh = img.size
            ratio = 1.25
            x, y, w, h = fix_ratio(x, y, w, h, ratio, oh, ow)
            crop_img = img.crop((x, y, x + w, y + h))

            if box.conf[0].item() > highest_probability:
                highest_probability = box.conf[0].item()
                best_crop_info = {'crop_img': crop_img, 'probability': highest_probability}

        elif class_1_prob and len(box.conf) > 0 and box.conf[0].item() > 0.6:
            xywh_tensor = box.xywh
            xywh_tensor = xywh_tensor.cpu()
            numpy_array = xywh_tensor.numpy()
            python_list = numpy_array.tolist()
            x = int(python_list[0][0] - ((python_list[0][2]) / 2))
            y = int(python_list[0][1] - ((python_list[0][3]) / 2))
            w = int(python_list[0][2])
            h = int(python_list[0][3])
            ow, oh = img.size
            ratio = 0.8
            x, y, w, h = fix_ratio(x, y, w, h, ratio, oh, ow)
            crop_img = img.crop((x, y, x + w, y + h))

            if len(box.conf) > 0 and box.conf[0].item() > highest_probability:
                highest_probability = box.conf[0].item()
                best_crop_info = {'crop_img': crop_img, 'probability': highest_probability}

    if best_crop_info:
        return [best_crop_info['crop_img']]
    else:
        return []


def correct_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass

    return img



# src = "test/original"
# out = "test/result"
# os.makedirs(out, exist_ok=True)
# os.makedirs(src, exist_ok=True)

# images = os.listdir(src)
# for image in images:
#     image_path = os.path.join(src, image)
#     if os.path.isfile(image_path):
#         img = Image.open(image_path)
#         img = correct_orientation(img)
#         crops = best_crop(img)
#         crops[0].save(os.path.join(out, image))

# crops = best_crop(correct_orientation(Image.open("1F6A6527.jpg")))
# crops[0].save("crops.png")
# # cv2.imwrite("out_put_1.jpg", crops[1])
# print(len(crops))