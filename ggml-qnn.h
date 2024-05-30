#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif


#define GGML_QNN_NAME           "QNN"
#define GGML_QNN_MAX_DEVICES    3

//QNN cDSP and HTA backend would not be used currently, just focus on QNN CPU/GPU/NPU(aka HTP/DSP) backend currently
enum QNNBackend {
    QNN_BACKEND_CPU,
    QNN_BACKEND_GPU,
    QNN_BACKEND_NPU,
    QNN_BACKEND_GGML, //"fake" QNN backend just for compare performance between QNN and original GGML
};

GGML_API int            ggml_backend_qnn_reg_devices(void);

/**
 *
 * @param device            0: QNN_BACKEND_CPU 1: QNN_BACKEND_GPU 2: QNN_BACKEND_NPU(aka HTP/DSP)
 * @param qnn_lib_path      qnn library path, such as "/data/local/tmp/" on Android or APK's internal data path on Android
 * @return
 */
GGML_API ggml_backend_t ggml_backend_qnn_init(size_t dev_num, const char * qnn_lib_path);

GGML_API bool           ggml_backend_is_qnn(ggml_backend_t backend);

GGML_API void           ggml_backend_qnn_set_n_threads(ggml_backend_t backend, int n_threads);

GGML_API int            ggml_backend_qnn_get_device_count(void);

GGML_API void           ggml_backend_qnn_get_device_description(int device, char * description, size_t description_size);


GGML_API ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(size_t dev_num);

#ifdef __cplusplus
}
#endif
