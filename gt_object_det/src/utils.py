import os
import json


def convert_gt_to_coco(data_path, channel, image_path, field_name, output='coco.json'):
    '''Convert AWS Ground Truth annotations to COCO annotation format'''
    D = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 0, 'name': 'person'}
        ],
    }
    anno_id = 0
    image_id = 0
    with open(os.path.join(data_path, f'{channel}.manifest')) as fp:
        for line in fp.readlines():
            info = json.loads(line.strip())
            labels = info[field_name]

            image_filepath = info['source-ref'][5:].partition('/')[2]
            image_id += 1

            image = {
                'coco_url': os.path.join(image_path, image_filepath),
                'height': labels['image_size'][0]['height'],
                'width': labels['image_size'][0]['width'],
                'id': image_id,
            }
            D['images'].append(image)

            for bbox in labels['annotations']:
                anno = {
                    'category_id': bbox['class_id'],
                    'bbox': [
                        int(bbox['left']),
                        int(bbox['top']),
                        int(bbox['left']) + int(bbox['width']),
                        int(bbox['top']) + int(bbox['height'])
                    ],
                    'image_id': image_id,
                    'id': anno_id,
                    'iscrowd': 0,
                    'area': int(bbox['width']) * int(bbox['height']),
                }
                anno_id += 1
                D['annotations'].append(anno)

    with open(os.path.join(data_path, output), 'w') as out_fp:
        out_fp.write(json.dumps(D))


if __name__ == '__main__':
    from pycocotools.coco import COCO
    data_path = './'
    image_path = './'
    channel = 'train'
    output = 'output.json'
    field_name = 'labels'
    convert_gt_to_coco(data_path, channel, image_path, field_name, output)
    coco = COCO(os.path.join(data_path, output))
