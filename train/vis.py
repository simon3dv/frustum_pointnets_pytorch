def results2kitti():
    '''
    Convert mmdetection's results to kitti label's format.
    needs config,pth,path of images,indexs,results.
    '''
    import mmcv
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    from mmdet.apis import inference_detector
    import os
    import cv2
    import numpy as np

    #can edit
    config='configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513.py'
    checkpoint = 'work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/latest.pth'
    file_dir = 'data/KITTI/object/training/image_2'
    idx_file = 'data/KITTI/object/image_sets/val.txt'
    result_dir = 'work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results'

    #load model and checkpoint
    cfg = mmcv.Config.fromfile(config)  # faster_rcnn_r50_fpn_1x.py
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint)

    #mkdir
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    print('load indexs.')
    idxs = []
    for line in open(idx_file,'r'):
        idxs.append(int(line))

    print('load imgs and inference and write.')
    typename = ['Car', 'Pedestrian', 'Cyclist']
    i = 0
    for idx in idxs:
        print('Writing %06d.txt' %(idx))
        img=mmcv.imread(os.path.join(file_dir, '%06d.png' % (idx)))
        result = inference_detector(model, img, cfg, device='cuda:0')
        # convert result[:3][n][5] to a txt file
        outputs=[]
        for type_id in range(3):#classes
            for j,r in enumerate(result[type_id]):#a class has j object
                box2d = r[:4]
                prob = r[4]
                output_str = typename[type_id] + " -1 -1 -10 "
                output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
                output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
                outputs.append(output_str)
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in outputs:
            fout.write(line + '\n')
        fout.close()

if __name__ == '__main__':
    results2kitti()
    #Then use ./kitti_eval/evaluate_object_3d_offline data/KITTI/object/training/label_2
    # work_dirs/carpedcyc/faster_rcnn_r50_fpn_1x_voc0712_carpedcyc190513/results
    # to evaluate