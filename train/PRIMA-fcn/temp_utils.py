import os, cv2

import numpy as np
from xml.etree import ElementTree as ET
from PIL import Image

'''
color map
0=background, 1=GraphicRegion, 2=MathsRegion, 3=FrameRegion, 4=LineDrawingRegion, 
5=NoiseRegion 6=ChartRegion, 7=TableRegion, 8=TextRegion, 9=ImageRegion, 10=SeparatorRegion
'''
color_map = {0: 'background', 1: 'GraphicRegion', 2: 'MathsRegion', 3: 'FrameRegion', 4: 'LineDrawingRegion', 
             5: 'NoiseRegion', 6: 'ChartRegion', 7: 'TableRegion', 8: 'TextRegion', 9: 'ImageRegion', 
             10: 'SeparatorRegion'}

palette = [0,0,0, 64,128,64, 128,0,192, 192,128,0, 64,128,0,
            0,0,128, 128,0,64, 192,0,64, 64,128,192, 128,192,192,
            128,64,64]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

id2name = color_map
name2id = {v: k for k, v in color_map.items()}

def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    region_dic = {}
    for region in root[1]:
        # print(region.tag, region.attrib)
        for coords in region:
            # print(coords.tag, coords.attrib)
            point_array = None
            for ind, point in enumerate(coords):
                # print(point.tag, point.attrib)
                if point_array is None:
                    point_array = np.array([int(point.get('x')), int(point.get('y'))])
                else:
                    tmp_array = np.array([int(point.get('x')), int(point.get('y'))])
                    point_array = np.concatenate((point_array, tmp_array), axis=0)
            if point_array is not None:
                point_array = point_array.reshape(-1, 2)
                class_name = region.tag.split('}')[1]
                if class_name not in region_dic.keys():
                    region_dic[class_name] = []
                region_dic[class_name].append(point_array)
    return region_dic

def gen_mask(xml_path, save_path):
    if xml_path.split('/')[-1][0:2] == 'pc':
        img_path = xml_path.replace('XML', 'Images').replace('pc-', '').replace('xml', 'tif')
        save_path = save_path.replace('pc-', '')
    else:
        img_path = xml_path.replace('XML', 'Images').replace('xml', 'tif')
    img = cv2.imread(img_path)
    tmp_img = np.zeros((img.shape[0], img.shape[1], 3))
    regions = parse_xml(xml_path)

    for cls_name, points in regions.items():
        for point in points:
            id_ = name2id[cls_name]
            # cv2 write pixel in BGR mode
            cv2.fillConvexPoly(tmp_img, point, (palette[id_*3+2], palette[id_*3+1], palette[id_*3]))
    cv2.imwrite(save_path, tmp_img)
    return

def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image
    palette.load()
    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    # the 0 above means turn OFF dithering

    # Later versions of Pillow (4.x) rename _makeself to _new
    try:
        return silf._new(im)
    except AttributeError:
        return silf._makeself(im)

def RBG2pmode(path):
    palimage = Image.new('P', (16, 16))
    palimage.putpalette(palette)
    oldimage = Image.open(path)
    newimage = quantizetopalette(oldimage, palimage, dither=False)
    newimage.save(path.replace("Masks", "masks"))
    return

def pmode2RGB(path):
    
    return


if __name__ == '__main__':
    xml_path = '/home/intsig/dataset/layout/PRImA Layout Analysis Dataset/XML/00000273.xml'

    read_dir = '/home/intsig/dataset/layout/PRImA Layout Analysis Dataset/XML'
    save_dir = '/home/intsig/dataset/layout/PRImA Layout Analysis Dataset/Masks'
    # for file in os.listdir(read_dir):
    #     save_path = os.path.join(save_dir, file.split('/')[-1].replace('xml', 'png'))
    #     gen_mask(os.path.join(read_dir, file), save_path)

    for file in os.listdir(save_dir):
        Pmode(os.path.join(save_dir, file))