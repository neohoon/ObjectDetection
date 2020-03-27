#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import numpy as np
import ObjectDetection
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


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]
MMAP_FNAME = "ObjectDetection.mmap"


def main(args):

    this = ObjectDetection.ObjectDetection(ini=utils.get_ini_parameters(args.ini_fname),
                                           logging_=False,
                                           stdout_=True)
    this.logger.info(" # ObjectDetection_client starts ...")

    utils.check_directory_existence(args.out_folder, exit_=False, create_=True)
    img_fnames = utils.get_filenames(args.img_path, extensions=utils.IMG_EXTENSIONS)
    this.logger.info(" # {:d} image files detected...".format(len(img_fnames)))
    this.logger.info("")

    for img_fname in img_fnames:
        this.logger.info("")
        img = utils.imread(img_fname)
        mmap = np.memmap(MMAP_FNAME, dtype='uint8', mode='w+', shape=tuple(img.shape))
        mmap[:] = img[:]

        request_dict = {"mmap_fname": os.path.abspath(MMAP_FNAME),
                        "mmap_shape": img.shape,
                        "roi": this.object_detector_ini['roi']}
        recv_dict, proc_time \
            = uSock.send_run_request_and_recv_response(this.server.ip,
                                                       this.server.port,
                                                       request_dict,
                                                       show_send_dat_=True,
                                                       show_recv_dat_=True,
                                                       prefix = " Client #",
                                                       logger = this.logger)

        if recv_dict:
            if recv_dict['result'] == "success":
                obj_img = this.make_object_boxed_image(
                    img,
                    recv_dict['obj_info']['obj_box_arr'],
                    obj_name_arr=recv_dict['obj_info']['obj_name_arr'],
                    obj_score_arr=recv_dict['obj_info']['obj_score_arr'],
                    roi=this.object_detector_ini['roi'],
                    color=0,
                    thickness=2,
                    alpha=0)
                if True:
                    out_img_fname = os.path.join(
                        args.out_folder,
                        os.path.splitext(os.path.basename(img_fname))[0] + "__obj.jpg")
                    utils.imwrite(obj_img, out_img_fname)
                    utils.imshow(obj_img, pause_sec=args.imshow_sec)

                if True:
                    obj_info = {'img_fname': img_fname,
                                'obj_box_arr': recv_dict['obj_info']['obj_box_arr'],
                                'obj_name_arr': recv_dict['obj_info']['obj_name_arr'],
                                'obj_score_arr': recv_dict['obj_info']['obj_score_arr'],
                                'elapsed_time': recv_dict['obj_info']
                                }

                    out_info_fname = os.path.join(args.out_folder,
                                                  os.path.splitext(os.path.basename(img_fname))[0] + "__obj.json")
                    with open(out_info_fname, 'w') as f:
                        json.dump(obj_info, f, indent=4)

            else:
                this.logger.error(" @ Error: result is not success, {}.".format(recv_dict['result']))
                this.logger.error(" % recv_dict is {},".format(str(recv_dict)))

        else:
            this.logger.error(" @ Error: response is \"{}\".".format(recv_dict))

        del mmap


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--ini_fname", required=True, help="ini filename")
    parser.add_argument("--img_path", required=True, help="Input image path")
    parser.add_argument("--out_folder", default="./Output", help="Output folder")
    parser.add_argument("--imshow_sec", default="0", type=int, help="imshow second")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
INI_FNAME = "ObjectDetection.ini"
# IMG_PATH = "./Input/baggage_claim.jpg"
IMG_PATH = "./Input/"
OUT_FOLDER = "./Output/Client/"


if __name__ == "__main__":

    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--img_path", IMG_PATH])
            sys.argv.extend(["--out_folder", OUT_FOLDER])
            sys.argv.extend(["--imshow_sec", '0'])
        else:
            sys.argv.extend(["--help"])
    main(parse_arguments(sys.argv[1:]))
