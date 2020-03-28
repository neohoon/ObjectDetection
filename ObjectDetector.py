#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import json
from enum import Enum
import time
import numpy as np
try:
    # noinspection PyUnresolvedReferences
    import UtilsCommon as utils
    # noinspection PyUnresolvedReferences
    import UtilsSocket as uSock
    # noinspection PyUnresolvedReferences
    import UtilsVideo as uVid
except ModuleNotFoundError:
    from Utils import UtilsCommon as utils
    from Utils import UtilsSocket as uSock
    from Utils import UtilsVideo as uVid
from Libs.YOLO  import Yolo
from Libs.DeepSortPytorch import deep_sort as DeepSort


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


class ObjectDetectorOpMode(Enum):
    img_path_proc = 1
    vid_file_proc = 2
    img_proc_server = 3


class ObjectDetector:
    def __init__(self, ini=None, logger=None, logging_=True, stdout_=True):
        self.ini = ini
        self.logger = logger
        self.logging_ = logging_
        self.stdout_ = stdout_

        self.object_detector_ini = None
        self.yolo_ini = None
        self.deepsort_ini = None

        self.detector_inst_ini = None
        self.detector_inst = None

        self.tracker_inst_ini = None
        self.tracker_inst = None

        self.detector_height = None
        self.detector_width = None
        self.tracker_ = False
        self.tracking_object_names = []
        self.tracking_object_indexes = []

        self.vid_fname = None

        self.out_vid = None
        self.out_vid_fname = None
        self.server = None

        self.out_folder = None
        self.save_obj_img_ = False
        self.save_obj_info_ = False

        if ini:
            self.init_ini()
            if self.logger is None:
                if self.logging_ is True:
                    self.logger = utils.setup_logger_with_ini(self.ini['LOGGER'],
                                                              logging_=self.logging_,
                                                              stdout_=self.stdout_)
                elif self.stdout_ is True:
                    self.logger = utils.get_stdout_logger()
                else:
                    pass
            self.server = self.init_server_mode(self.ini['SERVER_MODE'])

    def init_ini(self):
        self.ini = utils.remove_comments_in_ini(self.ini)
        self.object_detector_ini = dict(self.ini['OBJECT_DETECTOR'])
        self.yolo_ini = dict(self.ini['YOLO'])
        self.deepsort_ini = dict(self.ini['DEEPSORT'])

        self.object_detector_ini['fps'] = int(eval(self.object_detector_ini['fps']))
        self.object_detector_ini['roi'] = eval(self.object_detector_ini['roi'])
        self.object_detector_ini['detector_height'] = int(self.object_detector_ini["detector_height"])
        self.detector_height = self.object_detector_ini['detector_height']
        self.tracker_ = eval(self.object_detector_ini['tracker_'])
        self.tracking_object_names = self.object_detector_ini['tracking_object_names'].split(',')

    def init_server_mode(self, ini):
        self.server = uSock.Server(ini=ini)
        return self.server

    def init_detector_method(self):
        self.logger.info(" # Object Detector method initialization...")
        if self.object_detector_ini['detector_method'].lower() == 'yolo':
            self.detector_inst = self.YoloWrapper(ini=self.yolo_ini, logger=self.logger)
            self.tracking_object_indexes = self.detector_inst.get_indexes_from_names(self.tracking_object_names)
            self.detector_inst_ini = self.yolo_ini
        else:
            self.logger.error(" @ Incorrect detector method, {}".format(self.object_detector_ini['detect_method']))
            sys.exit()

    def init_tracker_method(self):
        self.logger.info(" # Object Tracker method initialization...")
        if self.object_detector_ini['tracker_method'].lower() == 'deepsort':
            self.tracker_inst = self.DeepSortWrapper(ini=self.deepsort_ini, logger=self.logger)
            self.tracker_inst_ini = self.deepsort_ini
        else:
            self.logger.error(" @ Incorrect tracker method, {}".format(self.object_detector_ini['detect_method']))
            sys.exit()

    class YoloWrapper:
        def __init__(self, ini=None, logger=utils.get_stdout_logger()):
            self.ini = ini
            self.logger = logger
            self.class_list = None
            self.net = None
            if ini:
                self.init_ini()

        def init_ini(self):
            self.init_net()

        def init_net(self):
            self.logger.info(" # YOLO : loading {} with {} target ...".
                             format(self.ini['target'], self.ini['model_path']))
            if self.ini['target'] == 'YoloCpuCv2':
                self.net = Yolo.YoloCpuCv2()
                self.net.initialize(weight_path=self.ini['model_path'],
                                    config_path=self.ini['config_path'],
                                    class_path=self.ini['class_path'])
            elif self.ini['target'] == 'YoloGpuPackage':
                self.net = Yolo.YoloGpuPackage()
                self.net.initialize(weight_path=self.ini['model_path'],
                                    config_path=self.ini['config_path'],
                                    data_path=self.ini['data_path'])
            else:
                self.logger.error(" @ Error: invalid YOLO target, {}.".format(self.ini['target']))
                sys.exit()

            self.class_list = []
            with open(self.ini['class_path'], 'r') as f:
                while True:
                    line = f.readline().rstrip()
                    if not line:
                        break
                    self.class_list.append(line)
            self.logger.info(" # YOLO : loaded.")

        def run(self, img):
            output_dict = self.net.run(img,
                                       conf_thresh=float(self.ini['conf_threshold']),
                                       nms_thresh=float(self.ini['nms_threshold']),
                                       scale=float(eval(self.ini['scale'])),
                                       blob_size=int(self.ini['blob_size']))
            obj_box_arr = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                           for box in output_dict['detection_boxes']]
            obj_score_arr = output_dict['detection_scores']
            obj_name_arr = output_dict['detection_names']

            return obj_box_arr, obj_name_arr, obj_score_arr

        def get_indexes_from_names(self, names):
            indexes = []
            for name in names:
                try:
                    indexes.append(self.class_list.index(name))
                except ValueError:
                    self.logger.error(" @ Warning: {} name DOES NOT exist in YOLO class list".format(name))
            return indexes

    class DeepSortWrapper:

        def __init__(self, ini=None, logger=utils.get_stdout_logger()):
            self.ini = ini
            self.logger = logger

            self.reid_ckpt = None
            self.max_dist = 0.2
            self.min_confidence = 0.3
            self.nms_max_overlap = 0.5
            self.max_iou_distance = 0.7
            self.max_age = 70
            self.n_init = 3
            self.nn_budget = 100
            self.use_cuda_ = True
            self.tracking_object_names = ['']

            self.net = None
            if ini:
                self.init_ini(ini=self.ini)

        def init_ini(self, ini=None):

            ini = self.ini if ini is None else ini

            self.reid_ckpt = ini['reid_ckpt']
            self.max_dist = eval(ini['max_dist']) if 'max_dist' in ini else 0.2
            self.min_confidence = eval(ini['min_confidence']) if 'min_confidence' in ini else 0.3
            self.nms_max_overlap = eval(ini['nms_max_overlap']) if 'nms_max_overlap' in ini else 0.5
            self.max_iou_distance = eval(ini['max_iou_distance']) if 'max_iou_distance' in ini else 0.7
            self.max_age = eval(ini['max_age']) if 'max_age' in ini else 70
            self.n_init = eval(ini['n_init']) if 'n_init' in ini else 3
            self.nn_budget = eval(ini['nn_budget']) if 'nn_budget' in ini else 100
            self.use_cuda_ = eval(ini['use_cuda_']) if 'use_cuda_' in ini else True

            self.init_net()

        def init_net(self):
            self.logger.info(" # DeepSort : loading {} to {} device ...".
                             format(self.reid_ckpt, 'cuda' if self.use_cuda_ else 'cpu'))
            self.net = DeepSort.DeepSort(self.reid_ckpt,
                                         max_dist=self.max_dist,
                                         min_confidence=self.min_confidence,
                                         nms_max_overlap=self.nms_max_overlap,
                                         max_iou_distance=self.max_iou_distance,
                                         max_age=self.max_age,
                                         n_init=self.n_init,
                                         nn_budget=self.nn_budget,
                                         use_cuda=self.use_cuda_)
            self.logger.info(" # DeepSort : loaded.")

        def run(self, bbox_xyxy, cls_conf, img):

            bbox = []
            identities = []
            if bbox_xyxy:
                bbox_xywh = []
                for box in bbox_xyxy:
                    w, h = box[2] - box[0], box[3] - box[1]
                    bbox_xywh.append([box[0]+w/2, box[1]+h/2, w, h])
                bbox_xywh = np.array(bbox_xywh)
                bbox_xywh[:,3:] *= 1.2    # bbox dilation just in case bbox too small.
                bbox_xywh.astype(np.int32)
                outputs = self.net.update(bbox_xywh, cls_conf, img)     # do tracking
                if len(outputs) > 0:
                    bbox = outputs[:, :4]
                    identities = outputs[:, -1]

            return bbox, identities

    def make_object_boxed_image(self,
                                img,
                                obj_box_arr,
                                obj_name_arr=None,
                                obj_score_arr=None,
                                obj_index_arr=None,
                                roi=None,
                                color=utils.RED,
                                thickness=2,
                                alpha=0.):

        self.logger.info(" # Making object boxes image...")
        for idx in range(len(obj_box_arr)):
            pos = obj_box_arr[idx]
            if all([True if x <= 2  else False for x in pos]):
                sz = img.shape[1::-1]
                pos = [int(pos[0]*sz[0]), int(pos[1]*sz[1]), int(pos[2]*sz[0]), int(pos[3]*sz[1])]
            text  = obj_name_arr[idx] if obj_name_arr is not None else ''
            text += "({:d})".format(obj_index_arr[idx]) if obj_index_arr is not None else ''
            text += " : " + str(int(obj_score_arr[idx] * 1000)) if obj_score_arr is not None else ''
            if isinstance(color, int):
                if color == 0:
                    box_color = utils.WHITE if obj_name_arr[idx] == 'person' else utils.get_random_color(3)
                else:
                    box_color = utils.get_random_color(3)
            else:
                box_color = color
            img = utils.draw_box_on_img(img, pos, color=box_color, thickness=thickness, alpha=alpha)
            if text is not '':
                img = cv2.putText(img, text, (pos[0] + 4, pos[3] - 4), CV2_FONT, 0.5, utils.WHITE, 6)
                img = cv2.putText(img, text, (pos[0] + 4, pos[3] - 4), CV2_FONT, 0.5, utils.BLACK, 2)

        if roi is not None:
            if len(roi) == 4:
                h, w, _ = img.shape
                pt1 = [int(w * roi[0]), int(h * roi[1])]
                pt2 = [int(w * roi[2]), int(h * roi[3])]
                img = cv2.rectangle(img, tuple(pt1), tuple(pt2), utils.BLACK, thickness=4)

        return img

    def make_tracking_object_boxed_image(self,
                                         img,
                                         obj_boxes,
                                         obj_indexes,
                                         roi=None,
                                         thickness=2,
                                         alpha=0.):

        self.logger.info(" # Making tracking object boxes image...")
        for idx in range(len(obj_boxes)):

            pos = obj_boxes[idx]
            if all([True if x <= 2  else False for x in pos]):
                sz = img.shape[1::-1]
                pos = [int(pos[0]*sz[0]), int(pos[1]*sz[1]), int(pos[2]*sz[0]), int(pos[3]*sz[1])]

            box_text  = str(obj_indexes[idx])
            box_color = utils.COLOR_ARRAY_RGBCMY[int(obj_indexes[idx]) % 6]
            img = utils.draw_box_on_img(img, pos, color=box_color, thickness=thickness, alpha=alpha)
            img = cv2.putText(img, box_text, (pos[0] + 4, pos[1] - 4), CV2_FONT, 1, utils.BLACK, 10)
            img = cv2.putText(img, box_text, (pos[0] + 4, pos[1] - 4), CV2_FONT, 1, utils.WHITE, 2)

        if roi is not None:
            if len(roi) == 4:
                h, w, _ = img.shape
                pt1 = [int(w * roi[0]), int(h * roi[1])]
                pt2 = [int(w * roi[2]), int(h * roi[3])]
                img = cv2.rectangle(img, tuple(pt1), tuple(pt2), utils.RED, thickness=4)

        return img

    def detect_objects(self, img):
        stt_time = time.time()

        width, height = img.shape[1], img.shape[0]
        if self.detector_height > 0:
            detector_width = int(self.detector_height / float(height) * width)
            detector_height = self.detector_height
            resize_img = utils.imresize(img,
                                        width=detector_width,
                                        height=detector_height,
                                        interpolation=cv2.INTER_CUBIC)
            self.logger.info(" # Resize image from {:d} x {:d} to {:d} x {:d}.".
                             format(width, height, detector_width, detector_height))
        else:
            detector_height = height
            detector_width = width
            resize_img = img
        scale = detector_height / float(height)

        x0 = int(detector_width  * self.object_detector_ini['roi'][0])
        y0 = int(detector_height * self.object_detector_ini['roi'][1])
        x1 = int(detector_width  * self.object_detector_ini['roi'][2])
        y1 = int(detector_height * self.object_detector_ini['roi'][3])

        crop_img = resize_img[y0:y1, x0:x1]

        try:
            obj_box_arr, obj_name_arr, obj_score_arr = self.detector_inst.run(crop_img)
        except Exception as e:
            obj_box_arr, obj_name_arr, obj_score_arr = [], [], []
            self.logger.error(e)

        scale_obj_box_arr = []
        for obj_box in obj_box_arr:
            box = [obj_box[0] + x0, obj_box[1] + y0, obj_box[2] + x0, obj_box[3] + y0]
            box = [p / scale for p in box]
            box = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
            scale_obj_box_arr.append(box)

        elapsed_time = time.time() - stt_time

        return scale_obj_box_arr, obj_name_arr, obj_score_arr, elapsed_time

    def track_objects(self, img, obj_box_arr, obj_name_arr, obj_score_arr):

        boxes = []
        scores = []
        h, w, _ = img.shape
        hh, ww = float(h), float(w)
        for idx in range(len(obj_box_arr)):
            if obj_name_arr[idx] in self.tracking_object_names:
                if sum(obj_box_arr[idx]) < 4:
                    boxes.append([int(obj_box_arr[idx][0]*w), int(obj_box_arr[idx][1]*h),
                                  int(obj_box_arr[idx][2]*w), int(obj_box_arr[idx][3]*h)])
                else:
                    boxes.append(obj_box_arr[idx])
                scores.append(obj_score_arr[idx])

        tracker_boxes, tracker_indexes = self.tracker_inst.run(boxes, scores, img)
        if bool(list(tracker_boxes)):
            tracker_boxes = [[box[0]/ww, box[1]/hh, box[2]/ww, box[3]/hh] for box in tracker_boxes]
            tracker_indexes = [int(p) for p in tracker_indexes]

        return tracker_boxes, tracker_indexes

    def detect_objects_in_image(self,
                                img,
                                img_fname=None,
                                imshow_sec=-1,
                                show_video_=False,
                                out_folder=".",
                                save_obj_img_=False,
                                save_obj_info_=False,
                                logger=utils.get_stdout_logger()):

        logger.info(" # Read image, {}".format(img_fname))
        obj_box_arr, obj_name_arr, obj_score_arr, elapsed_time = self.detect_objects(img)
        logger.info(" # Detected {:d} objects in {:.3f} sec in image, {}.".
                         format(len(obj_box_arr), elapsed_time, img_fname))

        tracker_boxes, tracker_indexes = [], []
        if self.tracker_:
            tracker_boxes, tracker_indexes = self.track_objects(img, obj_box_arr, obj_name_arr, obj_score_arr)

        obj_img = None
        prefix = '__obj_track' if self.tracker_ else '__obj'
        if save_obj_img_ or show_video_:
            obj_img = self.make_object_boxed_image(img.copy(),
                                                   obj_box_arr,
                                                   obj_name_arr=obj_name_arr,
                                                   obj_score_arr=obj_score_arr,
                                                   roi=self.object_detector_ini['roi'],
                                                   color=0,
                                                   thickness=2,
                                                   alpha=0.5)
            if self.tracker_:
                obj_img = self.make_tracking_object_boxed_image(obj_img,
                                                                tracker_boxes,
                                                                tracker_indexes,
                                                                roi=self.object_detector_ini['roi'],
                                                                thickness=4,
                                                                alpha=0)

            out_img_fname = os.path.join(out_folder,
                                         os.path.splitext(os.path.basename(img_fname))[0] + prefix + ".jpg")
            utils.imshow(obj_img, pause_sec=imshow_sec)
            if save_obj_img_:
                utils.imwrite(obj_img, out_img_fname)
            if show_video_:
                imshow_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame', imshow_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False, None, None

        obj_info = {'img_fname': img_fname,
                    'obj_box_arr': obj_box_arr,
                    'obj_name_arr': obj_name_arr,
                    'obj_score_arr': obj_score_arr,
                    'tracker_box_arr': tracker_boxes,
                    'tracker_index_arr': tracker_indexes,
                    'elapsed_time': float("{:.3f}".format(elapsed_time))}
        if save_obj_info_:
            out_info_fname = os.path.join(out_folder,
                                          os.path.splitext(os.path.basename(img_fname))[0] + prefix + ".json")
            with open(out_info_fname, 'w') as f:
                json.dump(obj_info, f, indent=4)

        return True, obj_info, obj_img

    def process_img(self,
                    img_path,
                    out_folder='.',
                    imshow_sec=-1,
                    save_obj_img_=False,
                    save_obj_info_=True,
                    logger=utils.get_stdout_logger()):

        img_fnames = utils.get_filenames(img_path, extensions=utils.IMG_EXTENSIONS)
        logger.info(" # {:d} image files detected...".format(len(img_fnames)))
        logger.info("")

        for img_fname in img_fnames:
            img = utils.imread(img_fname)
            self.detect_objects_in_image(img,
                                         img_fname=img_fname,
                                         imshow_sec=imshow_sec,
                                         out_folder=out_folder,
                                         save_obj_img_=save_obj_img_,
                                         save_obj_info_=save_obj_info_,
                                         logger=logger)

    def process_vid(self,
                    vid_file,
                    out_folder='.',
                    save_obj_vid_=False,
                    show_video_=False,
                    save_obj_info_=False,
                    logger=utils.get_stdout_logger()):

        utils.check_file_existence(vid_file, print_=False, exit_=True)
        utils.check_file_extensions(vid_file, utils.VIDEO_EXTENSIONS, print_=False, exit_=True)

        vid_cap = cv2.VideoCapture(vid_file)
        # vid_info = uVid.get_vid_info(vid_file, print_=True)
        fps = vid_cap.get(cv2.CAP_PROP_FPS) if self.object_detector_ini['fps'] == 0 else self.object_detector_ini['fps']

        out_vid_fname = None
        out_vid = None
        prefix = '__obj_track' if self.tracker_ else '__obj'
        if save_obj_vid_:
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_vid_fname = os.path.join(out_folder,
                                         os.path.splitext(os.path.basename(vid_file))[0] + prefix + ".avi")
            out_vid = cv2.VideoWriter(out_vid_fname,
                                      fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                      fps=float(fps),
                                      frameSize=(width, height))
        count = 0
        obj_info_arr = []
        while vid_cap.isOpened():
            time_stamp = count / fps
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp * 1000))
            # print(vid_cap.get(cv2.CAP_PROP_POS_MSEC))
            success, frame = vid_cap.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ret_, obj_info, obj_img = \
                self.detect_objects_in_image(frame,
                                             img_fname="{:d} at {:.3f} sec".format(count + 1, time_stamp),
                                             show_video_=show_video_,
                                             out_folder=out_folder,
                                             save_obj_img_=False,
                                             save_obj_info_=False,
                                             logger=logger)
            if not ret_:
                return

            obj_info_arr.append(obj_info)
            out_vid.write(obj_img)
            count += 1

        if save_obj_info_:
            out_info_fname = os.path.join(out_folder,
                                          os.path.splitext(os.path.basename(vid_file))[0] + prefix + ".json")
            with open(out_info_fname, 'w') as f:
                json.dump(obj_info_arr, f, indent=4)

        if save_obj_vid_:
            out_vid.release()
            uVid.convert_avi_to_mp4(os.path.splitext(out_vid_fname)[0] + ".avi", logger=self.logger)

    def process_request(self, request_dict, logger=utils.get_stdout_logger()):
        try:
            self.object_detector_ini['roi'] = request_dict['roi']
            img = np.memmap(request_dict['mmap_fname'], dtype='uint8', mode='r',
                            shape=tuple(request_dict['mmap_shape']))
        except all:
            return {'result': 'fail',
                    'description': 'something wrong in request_dict, {}'.format(str(request_dict))}

        ret_, obj_info, obj_img = \
            self.detect_objects_in_image(img,
                                         img_fname="{}.jpg".format(utils.get_datetime()),
                                         imshow_sec=-1,
                                         out_folder=self.out_folder,
                                         save_obj_img_=self.save_obj_img_,
                                         save_obj_info_=self.save_obj_info_,
                                         logger=logger)

        return {'result': 'success',
                'obj_info': obj_info}


def main(args):

    this = ObjectDetector(ini=utils.get_ini_parameters(args.ini_fname),
                          logging_=args.logging_,
                          stdout_=args.stdout_)
    op_mode = ObjectDetectorOpMode[args.op_mode]
    this.logger.info(" # ObjectDetector starts with \"{}\" operation mode...".format(op_mode.name))
    this.init_detector_method()
    if args.tracking_:
        this.init_tracker_method()
    utils.check_directory_existence(args.out_folder, create_=True)

    if op_mode is ObjectDetectorOpMode.img_path_proc:
        this.process_img(args.in_path,
                         out_folder=args.out_folder,
                         imshow_sec=-1,
                         save_obj_img_=args.save_obj_img_,
                         save_obj_info_=args.save_obj_info_,
                         logger=this.logger)

    elif op_mode is ObjectDetectorOpMode.vid_file_proc:
        this.process_vid(args.in_path,
                         out_folder=args.out_folder,
                         save_obj_vid_=True,
                         show_video_=True,
                         save_obj_info_=args.save_obj_info_,
                         logger=this.logger)

    elif op_mode is ObjectDetectorOpMode.img_proc_server:
        this.out_folder = args.out_folder
        this.save_obj_img_ = args.save_obj_img_
        this.save_obj_info_ = args.save_obj_info_
        uSock.run_command_server(this.server,
                                 func=this.process_request,
                                 logger=this.logger)
    else:
        pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", help="operation mode", required=True,
                        choices=[x.name for x in ObjectDetectorOpMode])
    parser.add_argument("--ini_fname", required=True, help="ini filename")
    parser.add_argument("--in_path", required=True, help="Input image or video file path")
    parser.add_argument("--out_folder", default="./Output", help="Output folder")

    parser.add_argument("--tracking_", default=False, action='store_true', help="Flag for tracking objects")
    parser.add_argument("--save_obj_img_", default=False, action='store_true', help="Flag for saving object_image")
    parser.add_argument("--save_obj_info_", default=False, action='store_true', help="Flag for saving_object info")

    parser.add_argument("--logging_", default=False, action='store_true', help="Logging flag")
    parser.add_argument("--stdout_", default=False, action='store_true', help="Standard output flag")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
INI_FNAME = "ObjectDetector.ini"
# OP_MODE = "img_path_proc"
OP_MODE = "vid_file_proc"
# OP_MODE = 'img_proc_server'

if OP_MODE == 'img_path_proc':
    # IN_PATH = "./Input/baggage_claim.jpg"
    IN_PATH = "./Input/"
    OUT_FOLDER = "./Output/"
elif OP_MODE == 'vid_file_proc':
    # IN_PATH = "./Input/overpass.mp4"
    # IN_PATH = "./Input/airport.mp4"
    # IN_PATH = "./Input/drone_person_1.mp4"
    IN_PATH = "./Input/drone_wedding_1.mp4"
    OUT_FOLDER = "./Output/"
elif OP_MODE == 'img_proc_server':
    IN_PATH = ''
    OUT_FOLDER = "./Output/Server/"
else:
    IN_PATH = "./Input/"
    OUT_FOLDER = "./Output/"


if __name__ == "__main__":

    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--in_path", IN_PATH])
            sys.argv.extend(["--out_folder", OUT_FOLDER])

            sys.argv.extend(["--tracking_"])
            sys.argv.extend(["--save_obj_img_"])
            sys.argv.extend(["--save_obj_info_"])
            # sys.argv.extend(["--logging_"])
            sys.argv.extend(["--stdout_"])

        else:
            sys.argv.extend(["--help"])
    main(parse_arguments(sys.argv[1:]))
