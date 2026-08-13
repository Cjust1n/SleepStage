/*
 * cvapp.h
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#ifndef SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_
#define SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_

/* Forward declaration — the full struct definition lives in the YOLOv8
 * scenario app headers which are not compiled as part of this TFLM
 * sleepstage build. The parameter is not dereferenced in this scenario. */
typedef struct { int dummy; } struct_yolov8_ob_algoResult;

#ifdef __cplusplus
extern "C" {
#endif

int cv_mb_cls_init(bool security_enable, bool privilege_enable, uint32_t model_addr);

int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob);

int cv_mb_cls_deinit();
#ifdef __cplusplus
}
#endif

#endif /* SCENARIO_TFLM_2IN1_FD_FL_PL_CVAPP_PL_ */
