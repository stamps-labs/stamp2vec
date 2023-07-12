# generate images of documents with stamps on random coordinates, also write coresponding xlm files 
import xml.etree.ElementTree as ET
from PIL import Image
import random
import os
import glob

def generate_xml(file_params, objects):
    template = '''
    <annotation>
        <folder>{folder}</folder>
        <filename>{filename}</filename>
        <path>{path}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>0</depth>
        </size>
    </annotation>
    '''

    object_template = '''
    <object>
        <name>{stamp}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>\n
    '''
    root = ET.fromstring(template.format(**file_params))
    for o in objects:
        root.append(ET.fromstring(object_template.format(**o)))

    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    return tree


for i in range(0, 10):
    doc = Image.open(r"images\input_files\\" + random.choice(os.listdir(r"images\input_files")))
    width, height = doc.size
    file_params = {
        "folder": "images",
        "filename": f"img{i}.png",
        "path": f"images\img{i}.png",
        "width": width,
        "height": height
    }

    objects = []
    objects1 = []
    for _ in range(random.randint(1, 3)):
        stamp_path = random.choice(glob.glob(r"images\stamps\*.png"))
        stamp_name = stamp_path.split("\\")[2].split(".")[0]
        stamp = Image.open(stamp_path).convert("RGBA")
        
        # randomize stamp
        stamp = stamp.rotate(random.randint(0, 360), expand = True)
        stamp_width = int(width * 0.01 * random.randint(15, 25))
        wpercent = (stamp_width/float(stamp.width))
        hsize = int((float(stamp.height)*float(wpercent)))
        stamp = stamp.resize((stamp_width,hsize), Image.LANCZOS)

        mask = Image.open(f"images/masks/mask{random.randint(1, 6)}.png").convert("RGBA").rotate(random.randint(0, 360)).resize(stamp.size)
        for x in range(stamp.width):
            for y in range(stamp.height):
                r,g,b,a = stamp.getpixel((x, y))
                if(a > 0):
                    stamp.putpixel((x, y), (r,g,b, int(a * mask.getpixel((x, y))[3] / 255)))
        
        a, b = stamp.size
        if (width < a or height < b): continue
        coords = random.randrange(width-a), random.randrange(height-b)

        objects.append({
            "stamp": stamp_name,
            "xmin" : coords[0],
            "ymin" : coords[1],
            "xmax": coords[0]+a,
            "ymax": coords[1]+b})

        objects1.append({
            "stamp": stamp_name.split("_")[0],
            "xmin" : coords[0],
            "ymin" : coords[1],
            "xmax": coords[0]+a,
            "ymax": coords[1]+b})

        doc.paste(stamp, coords, stamp)
        cropped = doc.crop((coords[0], coords[1], coords[0] + a, coords[1] + b))
        cropped.save(f"images\cropped_stamps\{stamp_name}_{i}.png")

        stamp.close()

    doc.save(f"images\images\img{i}.png", "PNG")

    doc.close()

    tree = generate_xml(file_params, objects)
    file = open(f"images/annotations/img{i}.xml", "wb")
    tree.write(file)
    file.close()

    tree = generate_xml(file_params, objects1)
    file = open(f"images/annotations_shapes/img{i}.xml", "wb")
    tree.write(file)
    file.close()
