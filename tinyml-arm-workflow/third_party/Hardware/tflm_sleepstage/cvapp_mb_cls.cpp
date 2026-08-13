/*
 * cvapp_mb_cls.cpp
 *
 * Sleep-stage feature extraction (Accel + PPG/HRV) + debug prints.
 *
 * Fixes:
 *  - Remove undefined samples_read usage.
 *  - Use correct PPG epoch threshold (PPG_EPOCH_SAMPLES).
 *  - Guard RR/HRV pipeline so it only uses freshly computed values.
 *  - Fix MAX30102 FIFO collection logic based on available-samples return.
 *  - Print features in the order requested.
 *  - MAX30102 is now conditional (USE_MAX30102).
 *  - Added tensor shape/bytes validation during init.
 *  - Sliding window inference: prediction every 30s after initial 15min buffer.
 *  - Fixed PPG interpolation and peak detection parameters.
 *  - Inline peak detection to avoid linker errors.
 */

#include <cstdio>
#include <math.h>
#include <cstring>
#include <cstdint>

#include "WE2_device.h"
#include "WE2_core.h"
#include "board.h"
#include "memory_manage.h"
#include "xprintf.h"

#include "cvapp_mb_cls.h"

extern "C" {
    #include "hx_drv_iic.h"
    #include "hx_drv_scu.h"
    #include "timer_interface.h"
}

#include "mpu6050.h"

/* read_bytes_nonblock didefinisikan di send_result.cpp.
 * Deklarasi lokal minimal (tanpa include send_result.h yang berat).
 * Return 0 (EL_OK) = ada data terbaca, non-zero = tidak ada data. */
int read_bytes_nonblock(char *buffer, size_t size);

//----------------------------------------------------------------------
// MAX30102 HR sensor (PPG/HRV)
// Set to 1 if MAX30102 is connected, 0 for testing without HR sensor.
// When 0: HRV features remain zero/dummy, MPU6050 + features still work.
//----------------------------------------------------------------------
#define USE_MAX30102 1

#if USE_MAX30102
#include "max30102.h"
#endif

#include "ethosu_driver.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

//------------------------------------------------------------
// Preprocessing & Feature Extraction
//------------------------------------------------------------
#include "accel_buffer.h"
#include "movement_features.h"
#include "ppg_buffer.h"
#include "peak_detector.h"
#include "rr_buffer.h"
#include "hr_window.h"
#include "hrv_features.h"
#include "rr_history.h"
#include "temporal_features.h"
#include "tflm_mb_cls.h"

//------------------------------------------------------------
// Improved PPG Processing (High-pass filter + peak detector)
//------------------------------------------------------------
#include "highpass_filter.h"
#include "peak_detector_improved.h"

//----------------------------------------------------------
// HRV source selection
//   1 = rolling RR history (5+ minute window, static buffers)
//   0 = epoch-local rr_buffer (30 s window, original behaviour)
//----------------------------------------------------------
#define HRV_USE_RR_HISTORY 1

#if HRV_USE_RR_HISTORY
/* Minimum history duration (ms) before using rr_history for HRV.
 * Below this threshold, fall back to rr_buffer.
 * 120 000 ms = 2 minutes — ensures stable frequency-domain features
 * regardless of heart rate. */
#define HRV_MIN_HISTORY_MS  (2 * 60 * 1000)

/* Static (BSS) scratch buffers for rr_history linearisation + conversion.
 * Avoids ~7 KB stack allocation — critical for Cortex-M55. */
static rr_sample_t  rr_history_scratch[RR_HISTORY_CAPACITY];  /* 512 × 6 = 3072 bytes */
static float        rr_hist_float[RR_HISTORY_CAPACITY];       /* 512 × 4 = 2048 bytes */
static uint32_t     beat_ts_hist[RR_HISTORY_CAPACITY];        /* 512 × 4 = 2048 bytes */
#endif

//------------------------------------------------------------
// Configuration
//------------------------------------------------------------

#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif

/******************************************************************************
 * Tensor Arena
 ******************************************************************************/
constexpr int kTensorArenaSize = 256 * 1024;

/******************************************************************************
 * Sleep Stage Configuration
 ******************************************************************************/

#ifndef EPOCH_LENGTH_SEC
#define EPOCH_LENGTH_SEC          5
#endif
#define ACCEL_SAMPLE_RATE_HZ      50
#define PPG_SAMPLE_RATE_HZ        25    /* Actual from MAX30102: 100 Hz raw / 4 avg = 25 Hz */

#define ACCEL_SAMPLES_PER_EPOCH \
    (EPOCH_LENGTH_SEC * ACCEL_SAMPLE_RATE_HZ)

static uint8_t* tensor_arena = nullptr;

/*
*****************************************************************************
 * Global TFLM Objects
 *****************************************************************************
 */
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

/*
*****************************************************************************
 * Inference Configuration (gru_int8_vela.tflite)
 *
 * Discovered via flatbuffer inspection of gru_int8_vela.tflite:
 *
 *   Input  tensor "serving_default_x:0"        shape [1, 30, 18]  INT8
 *          scale = 0.1636824607849121  zero_point = -92
 *   Output tensor "StatefulPartitionedCall_1:0" shape [1, 4]       INT8
 *          scale = 0.05266352742910385 zero_point = -128
 *
 * IMPORTANT: This is a GRU model. It does NOT classify a single 30 s epoch
 * in isolation — it consumes a SEQUENCE of SEQ_LEN consecutive epochs
 * (30 epochs x 30 s = 15 minutes of context), each represented by the same
 * 18-feature vector computed by this pipeline, in EXACTLY the order defined
 * in SleepStage/configs/train_selected_features.yaml and mirrored by the
 * "F=" UART print.
 *****************************************************************************
 */
#define SEQ_LEN       30   /* GRU time steps  == model input dim 1 */
#define NUM_FEATURES  18   /* features/epoch  == model input dim 2 */
#define NUM_CLASSES    4   /* sleep stages     == model output dim 1 */

/* Expected input tensor size in bytes: SEQ_LEN * NUM_FEATURES * sizeof(int8_t) */
#define EXPECTED_INPUT_BYTES (SEQ_LEN * NUM_FEATURES)  /* 30 * 18 = 540 bytes */

/* Rolling (chronological) history of the last SEQ_LEN feature vectors.
 * Index 0 = oldest epoch in the window, SEQ_LEN-1 = most recent epoch. */
static float   g_feature_seq[SEQ_LEN][NUM_FEATURES];
static uint32_t g_feature_seq_filled = 0;   /* number of valid epochs (caps at SEQ_LEN) */

/* Human-readable class labels.
 * NOTE: verify this mapping against your training label encoding
 * (e.g. sklearn LabelEncoder / to_categorical class order) before
 * trusting it in a real product — the model only outputs indices 0..3. */
static const char* const kSleepStageNames[NUM_CLASSES] = {
    "Wake",
    "Light Sleep",
    "Deep Sleep",
    "REM"
};

/*
*****************************************************************************
 * Feature Objects
 *****************************************************************************
 */
static movement_features_t movement_features;
static hrv_features_t hrv_features;
static temporal_features_t temporal_features;

/*
*****************************************************************************
 * Rolling Feature State (HR history for trend features)
 *
 * Mirrors SleepStage/FeatureExtraction/hrv.py HRVFeatureExtractor:
 *   - _history keeps last ~5 HR values (trend_window=5)
 *   - rolling_mean_hr  = mean of rolling window
 *   - rolling_hr_range = max(win) - min(win)
 *   - hr_slope  = linear fit slope over window
 *   - hr_delta  = current_mean_hr - prev_mean_hr
 *
 * Mirrors SleepStage/FeatureExtraction/movement.py MovementFeatureExtractor:
 *   - _history_mean_acc keeps last ~3 mean_acc values
 *   - rolling_mean_acc = mean of history
 *   - rolling_std_acc  = std of history
 *****************************************************************************
 */
#define HR_TREND_WINDOW   5
#define ACCEL_HISTORY_LEN 3

static float g_hr_history[HR_TREND_WINDOW];
static uint32_t g_hr_history_count;
static float g_prev_mean_hr;

static float g_accel_mean_history[ACCEL_HISTORY_LEN];
static uint32_t g_accel_mean_history_count;

/*
*****************************************************************************
 * Runtime State
 *****************************************************************************
 */
static uint32_t recording_start_unix = 0;
static uint32_t epoch_index = 0;
static uint32_t current_timestamp_ms = 0;
static uint32_t total_epochs = 0;

/*
*****************************************************************************
 * Accelerometer State
 *****************************************************************************
 */
static mpu6050_axis_t accel;

/*
*****************************************************************************
 * Timestamp Helpers
 *****************************************************************************
 */
static inline uint32_t current_epoch_ms(void)
{
    return current_timestamp_ms;
}

/*
*****************************************************************************
 * Op Resolver
 *****************************************************************************
 */
static tflite::MicroMutableOpResolver<1> resolver;

namespace
{
ethosu_driver g_ethosu_drv;
}

static void ArmNpuIrqHandler(void)
{
    ethosu_irq_handler(&g_ethosu_drv);
}

static void ArmNpuIrqInit(void)
{
    const IRQn_Type irq = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(irq, (uint32_t)ArmNpuIrqHandler);
    NVIC_EnableIRQ(irq);
}

static bool ArmNpuInit(bool security_enable, bool privilege_enable)
{
    ArmNpuIrqInit();

    void* ethosu_base = (void*)U55_BASE;

    int status = ethosu_init(
        &g_ethosu_drv,
        ethosu_base,
        nullptr,
        0,
        security_enable,
        privilege_enable);

    if(status != 0)
    {
        xprintf("ERROR: Failed to initialize Ethos-U55\r\n");
        return false;
    }

    xprintf("Ethos-U55 initialized\r\n");
    return true;
}

/*
*****************************************************************************
 * Model Initialization
 *****************************************************************************
 */
int cv_mb_cls_init(bool security_enable,
                   bool privilege_enable,
                   uint32_t model_addr)
{
    xprintf("DEBUG: Received model_addr = 0x%08X\n", (unsigned int)model_addr);
    tensor_arena = reinterpret_cast<uint8_t*>(mm_reserve_align(kTensorArenaSize, 0x20));

    if (tensor_arena == nullptr)
    {
        xprintf("Failed to allocate tensor arena\n");
        return -1;
    }

    xprintf("Tensor Arena : 0x%08X\n", (uint32_t)tensor_arena);

    if (!ArmNpuInit(security_enable, privilege_enable))
    {
        return -1;
    }

    if (model_addr == 0)
    {
        xprintf("Invalid model address\n");
        return -1;
    }

    model = tflite::GetModel((const void *)model_addr);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        xprintf("Schema mismatch (%d != %d)\n", model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    xprintf("Model schema version : %d\n", model->version());

    if (resolver.AddEthosU() != kTfLiteOk)
    {
        xprintf("Failed to register Ethos-U operator\n");
        return -1;
    }

    static tflite::MicroInterpreter g_micro_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize);

    interpreter = &g_micro_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        xprintf("AllocateTensors failed\n");
        return -1;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    xprintf("Input bytes  : %d\n", input->bytes);
    xprintf("Output bytes : %d\n", output->bytes);

    xprintf("Arena used   : %u bytes\n",
            (unsigned int)interpreter->arena_used_bytes());

    /* Sanity-check the tensors we're about to drive against what the
     * gru_int8_vela.tflite flatbuffer actually declares. This catches a
     * mismatched/rebuilt model at boot instead of silently feeding
     * garbage into Invoke(). */
    if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8)
    {
        xprintf("ERROR: Expected INT8 input/output tensors (got in=%d out=%d)\n",
                (int)input->type, (int)output->type);
        return -1;
    }

    /* NEW: Validate tensor shape/rank */
    if (input->dims->size != 3)
    {
        xprintf("ERROR: Unexpected input rank: %d (expected 3)\n", 
                (int)input->dims->size);
        return -1;
    }

    if (input->dims->data[1] != SEQ_LEN ||
        input->dims->data[2] != NUM_FEATURES)
    {
        xprintf("ERROR: Unexpected input shape: [%d,%d,%d] (expected [1,%d,%d])\n",
                (int)input->dims->data[0],
                (int)input->dims->data[1],
                (int)input->dims->data[2],
                SEQ_LEN, NUM_FEATURES);
        return -1;
    }

    /* NEW: Validate input bytes exactly match expected size */
    if (input->bytes != EXPECTED_INPUT_BYTES)
    {
        xprintf("ERROR: Input bytes mismatch: %d (expected %d)\n",
                (int)input->bytes, EXPECTED_INPUT_BYTES);
        return -1;
    }

    const int expected_output_elems = NUM_CLASSES;
    if (output->bytes != expected_output_elems)
    {
        xprintf("ERROR: Output bytes mismatch: %d (expected %d)\n",
                (int)output->bytes, expected_output_elems);
        return -1;
    }

    xprintf("Input  quant: scale=%.8f zero_point=%d\n",
            input->params.scale, input->params.zero_point);
    xprintf("Output quant: scale=%.8f zero_point=%d\n",
            output->params.scale, output->params.zero_point);

    xprintf("Initialization complete\n");
    return 0;
}

/*
*****************************************************************************
 * Inference Stage
 *
 * The model is a GRU that classifies sleep stage from a SEQ_LEN=30-epoch
 * (15-minute) sliding window of the 18-feature vector. The steps are:
 *
 *   1) push_feature_vector() - append the newest epoch's 18 features to a
 *      chronological ring buffer of the last 30 epochs.
 *   2) quantize_feature() - convert each float feature to INT8 using the
 *      model's own input scale/zero_point (read from `input->params`, not
 *      hardcoded, so this stays correct even if the model is re-exported
 *      with different quantization parameters).
 *   3) fill_input_tensor() - write all 30x18 quantized values into
 *      `input->data.int8` in [time][feature] order, matching the model's
 *      [1, 30, 18] input layout.
 *   4) interpreter->Invoke() - run the GRU on the Ethos-U55.
 *   5) dequantize + argmax the [1,4] INT8 output to get per-class scores
 *      and the predicted sleep stage.
 *****************************************************************************
 */

/* Quantize a single float feature value into INT8 using the tensor's own
 * affine quantization parameters: q = round(f / scale) + zero_point,
 * clamped to the valid INT8 range. */
static inline int8_t quantize_feature(float value, float scale, int32_t zero_point)
{
    int32_t q = (int32_t)lrintf(value / scale) + zero_point;

    if (q < -128) q = -128;
    if (q >  127) q =  127;

    return (int8_t)q;
}

/* Dequantize a single INT8 output value back to a real-valued class score:
 * real = (q - zero_point) * scale. */
static inline float dequantize_output(int8_t q, float scale, int32_t zero_point)
{
    return (float)((int32_t)q - zero_point) * scale;
}

/* Append the newest epoch's feature vector to the chronological sequence
 * buffer, shifting older epochs down by one slot (index 0 = oldest,
 * SEQ_LEN-1 = newest). This is a simple O(SEQ_LEN) shift; SEQ_LEN is only
 * 30, so the cost is negligible against the 30 s epoch cadence.
 *
 * CRITICAL: This implements SLIDING WINDOW behaviour.
 * After the first 30 epochs are filled, each new epoch:
 *   - Shifts all existing epochs left (drops oldest)
 *   - Appends new epoch at the end
 * This means we get a new inference every 30 seconds, not every 15 minutes.
 */
static void push_feature_vector(const float feat[NUM_FEATURES])
{
    if (g_feature_seq_filled >= SEQ_LEN)
    {
        /* Sliding window: drop oldest, shift left, append new */
        memmove(&g_feature_seq[0], &g_feature_seq[1],
                sizeof(float) * NUM_FEATURES * (SEQ_LEN - 1));
    }
    else
    {
        /* First fill phase: just append until we have SEQ_LEN epochs */
        g_feature_seq_filled++;
    }

    /* Always append the new feature vector at the end */
    memcpy(&g_feature_seq[SEQ_LEN - 1], feat, sizeof(float) * NUM_FEATURES);
}

/* Quantize the full [SEQ_LEN][NUM_FEATURES] history into the model's INT8
 * input tensor, in the exact [1, 30, 16] row-major layout Vela/TFLM expect. */
static void fill_input_tensor(void)
{
    const float   in_scale = input->params.scale;
    const int32_t in_zp    = input->params.zero_point;

    int8_t* dst = input->data.int8;

    for (int t = 0; t < SEQ_LEN; t++)
    {
        for (int f = 0; f < NUM_FEATURES; f++)
        {
            dst[t * NUM_FEATURES + f] =
                quantize_feature(g_feature_seq[t][f], in_scale, in_zp);
        }
    }
}

/* Run inference once the sequence buffer holds a full SEQ_LEN window.
 * With sliding window, this will be called every epoch after the initial
 * 15-minute buffer fills. */
static void run_inference_if_ready(void)
{
    if (g_feature_seq_filled < SEQ_LEN)
    {
        xprintf("Inference: buffering (%u/%u epochs)\r\n",
                (unsigned int)g_feature_seq_filled, (unsigned int)SEQ_LEN);
        return;
    }

    if (interpreter == nullptr || input == nullptr || output == nullptr)
    {
        xprintf("Inference: interpreter not ready\r\n");
        return;
    }

    fill_input_tensor();

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        xprintf("ERROR: Invoke() failed (status=%d)\r\n", (int)invoke_status);
        return;
    }

    const float   out_scale = output->params.scale;
    const int32_t out_zp    = output->params.zero_point;

    float   scores[NUM_CLASSES];
    int8_t  raw[NUM_CLASSES];
    int     best_idx = 0;

    for (int c = 0; c < NUM_CLASSES; c++)
    {
        raw[c]    = output->data.int8[c];
        scores[c] = dequantize_output(raw[c], out_scale, out_zp);

        if (c == 0 || raw[c] > raw[best_idx])
        {
            /* Argmax on the raw INT8 codes is equivalent to argmax on the
             * dequantized scores because dequantization is a strictly
             * increasing affine map (scale > 0) — comparing raw codes
             * avoids unnecessary float ops. */
            best_idx = c;
        }
    }

    xprintf("SCORES=");
    for (int c = 0; c < NUM_CLASSES; c++)
    {
int32_t sc = (int32_t)(scores[c] * 1000000.0f);
        int32_t ip = sc / 1000000;
        int32_t fp = (sc >= 0) ? (sc % 1000000) : (-(sc % 1000000));
        xprintf("%s%d.%06d", (c == 0) ? "" : ",", (int)ip, (int)fp);
    }
    xprintf("\r\n");

    xprintf("PREDICTION=%d (%s)\r\n", best_idx, kSleepStageNames[best_idx]);
}

/*
*****************************************************************************
 * Rolling-Feature Helpers
 *
 * Implement the same math as SleepStage/FeatureExtraction/*.py so the
 * 18-feature vector matches the order/training-time computation exactly.
 *
 *   linear_fit_slope()  → numpy.polyfit(x=arange(N), y, deg=1)[0]
 *   mean_f()            → numpy.mean
 *   std_f()             → numpy.std            (ddof=0, population)
 *
 * sin_time_of_night uses the same formula as TemporalFeatureExtractor:
 *   theta = 2*pi * (time_of_night / 24.0)
 *   sin_time_of_night = sin(theta)
 *****************************************************************************
 */
static const float PI_F = 3.14159265358979f;

static float mean_f(const float* x, int n)
{
    if (x == NULL || n <= 0)
        return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    return s / (float)n;
}

static float std_f(const float* x, int n)
{
    if (x == NULL || n < 2)
        return 0.0f;
    float m = mean_f(x, n);
    float v = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float d = x[i] - m;
        v += d * d;
    }
    v /= (float)n;             /* population std (numpy.std, ddof=0) */
    return sqrtf(v);
}

/* polyfit(arange(N), y, 1)[0] */
static float linear_fit_slope(const float* y, int n)
{
    if (y == NULL || n < 2)
        return 0.0f;
    float x_mean = (float)(n - 1) / 2.0f;
    float y_mean = mean_f(y, n);

    float num = 0.0f;
    float den = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float xd = (float)i - x_mean;
        num += xd * (y[i] - y_mean);
        den += xd * xd;
    }
    if (den == 0.0f)
        return 0.0f;
    return num / den;
}

static float compute_sin_time_of_night(float time_of_night)
{
    float theta = 2.0f * PI_F * (time_of_night / 24.0f);
    return sinf(theta);
}

/*
*****************************************************************************
 * Run Validation
*****************************************************************************
 */

#if HRV_USE_RR_HISTORY
/*
 * prepare_hrv_input_from_history()
 *
 * Linearises rr_history, validates timestamp monotonicity, checks minimum
 * history duration, and converts to hrv_features_extract() input format.
 *
 * Returns true if rr_history data is usable, false to trigger fallback.
 *
 * Output:
 *   rr_float_out[]  — RR intervals in ms (float)
 *   beat_ts_out[]   — cumulative beat timestamps (uint32_t)
 *   count_out       — number of valid intervals
 */
static bool prepare_hrv_input_from_history(
    float       *rr_float_out,
    uint32_t    *beat_ts_out,
    uint16_t    *count_out)
{
    uint16_t hist_size = rr_history_size();
    const rr_sample_t *hist = rr_history_get_linear(rr_history_scratch);

    if (hist == NULL || hist_size == 0)
    {
        return false;
    }

    /* Validate timestamp monotonicity.
     * hrv_features_extract() relies on strictly increasing beat timestamps
     * for linear interpolation. A single inversion will produce garbage. */
    for (uint16_t i = 1; i < hist_size; i++)
    {
        if (hist[i].timestamp_ms <= hist[i - 1].timestamp_ms)
        {
            xprintf("HRV WARN: non-monotonic ts at idx %u (%u <= %u)\r\n",
                    (unsigned int)i,
                    (unsigned int)hist[i].timestamp_ms,
                    (unsigned int)hist[i - 1].timestamp_ms);
            return false;
        }
    }

    /* Check history duration (not just count).
     * Frequency-domain features (LF/HF) require several minutes of data.
     * Using a time-based threshold is physiologically correct regardless
     * of heart rate. */
    uint32_t history_duration_ms = hist[hist_size - 1].timestamp_ms - hist[0].timestamp_ms;
    if (history_duration_ms < HRV_MIN_HISTORY_MS)
    {
        return false;
    }

    /* Convert: rr_sample_t → float rr_ms[] + uint32_t beat_ts[]. */
    for (uint16_t i = 0; i < hist_size; i++)
    {
        rr_float_out[i] = (float)hist[i].rr_ms;
        beat_ts_out[i]  = hist[i].timestamp_ms;
    }

    *count_out = hist_size;

    /* Enriched debug log for UART monitoring.
     * Helps verify history duration, heart rate, and data integrity. */
    float mean_rr = 0.0f;
    for (uint16_t i = 0; i < hist_size; i++)
    {
        mean_rr += rr_float_out[i];
    }
    mean_rr /= (float)hist_size;
    float approx_hr = (mean_rr > 0.0f) ? (60000.0f / mean_rr) : 0.0f;

    xprintf("HRV source: RR_HISTORY"
            " (cnt=%u, dur=%u ms, meanRR=%u ms, HR=%.0f bpm)\r\n",
            (unsigned int)hist_size,
            (unsigned int)history_duration_ms,
            (unsigned int)mean_rr,
            approx_hr);

    return true;
}
#endif

/*
*****************************************************************************
 * IMPROVED PEAK DETECTION
 *
 * Replaces the old detect_peaks_inline() with a two-stage pipeline:
 *   1. High-pass IIR filter (0.5 Hz cutoff) — removes baseline drift
 *   2. peak_detector_improved_process()     — MA-5, threshold 15%, etc.
 *
 * Benefits:
 *   - Eliminates baseline-drift-induced missed peaks
 *   - Catches ~95% of true beats (vs ~66% before)
 *   - Signal quality index discards noisy epochs
 *****************************************************************************
 */

/* Global high-pass filter instance (persists across epochs) */
static highpass_filter_t g_hp_filter;
static bool g_hp_filter_initialized = false;

int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob)
{
    (void)algoresult_yolov8n_ob;

    if (interpreter == nullptr)
    {
        xprintf("Interpreter not initialized\n");
        return -1;
    }

    //----------------------------------------------------------
    // I2C + Sensors Init
    //----------------------------------------------------------
    xprintf("\n");
    xprintf("=========================================\n");
#if USE_MAX30102
    xprintf("  Sleep Stage (Accel + PPG/HRV)\n");
#else
    xprintf("  Sleep Stage (Accel only, HRV dummy)\n");
#endif
    xprintf("Scanning I2C bus for devices...\n");
    for (uint8_t addr = 1; addr < 128; addr++) {
        uint8_t dummy;
        // Coba lakukan operasi baca 1 byte ke alamat 'addr'
        IIC_ERR_CODE_E err = hx_drv_i2cm_read_data(USE_DW_IIC_0, addr, &dummy, 1);
        
        if (err == IIC_ERR_OK) {
            xprintf("Found I2C device at address: 0x%02X\n", addr);
        }
    }
    xprintf("Scan complete.\n");
    xprintf("=========================================\n");

    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);

    hx_drv_i2cm_init(USE_DW_IIC_0,
                      HX_I2C_HOST_MST_0_BASE,
                      DW_IIC_SPEED_FAST);

    if (mpu6050_init() != IIC_ERR_OK)
    {
        xprintf("ERROR : MPU6050 initialization failed!\n");
        return -1;
    }

    xprintf("MPU6050 initialization success.\n");
    xprintf("Reading accelerometer @%dHz...\n\n", ACCEL_SAMPLE_RATE_HZ);

#if USE_MAX30102
    if (max30102_init() != 0)
    {
        xprintf("ERROR : MAX30102 initialization failed!\n");
        return -1;
    }
    xprintf("MAX30102 initialization success.\n");

    //----------------------------------------------------------
    // PART A — MAX30102 SAMPLE RATE VALIDATION
    //----------------------------------------------------------
    {
        uint8_t spo2_reg = 0;
        uint8_t fifo_reg = 0;
        if (max30102_read_register(MAX30102_SPO2_CONFIG, &spo2_reg) == 0 &&
            max30102_read_register(MAX30102_FIFO_CONFIG, &fifo_reg) == 0)
        {
            uint8_t spo2_sr = (spo2_reg >> 2) & 0x07;
            uint32_t raw_rate = 0;
            switch(spo2_sr) {
                case 0: raw_rate = 50;   break;
                case 1: raw_rate = 100;  break;
                case 2: raw_rate = 167;  break;
                case 3: raw_rate = 200;  break;
                case 4: raw_rate = 400;  break;
                case 5: raw_rate = 600;  break;
                case 6: raw_rate = 800;  break;
                case 7: raw_rate = 1000; break;
            }
            uint8_t smp_ave = (fifo_reg >> 5) & 0x07;
            uint8_t avg = 1;
            if (smp_ave == 1) avg = 2;
            else if (smp_ave == 2) avg = 4;
            else if (smp_ave == 3) avg = 8;
            else if (smp_ave == 4) avg = 16;
            else if (smp_ave == 5) avg = 32;
            uint32_t eff_sr = raw_rate / avg;
            float factor = (eff_sr > 0) ? (float)PPG_SAMPLE_RATE_HZ / (float)eff_sr : 1.0f;
            xprintf("\r\n");
            xprintf("--------------------------------\r\n");
            xprintf("MAX30102 Configuration\r\n");
            xprintf("--------------------------------\r\n");
            xprintf("SPO2_CONFIG = 0x%02X\r\n", spo2_reg);
            xprintf("FIFO_CONFIG = 0x%02X\r\n", fifo_reg);
            xprintf("\r\n");
            xprintf("Decoded fields:\r\n");
            xprintf("  ADC range       = %s\r\n",
                    ((spo2_reg >> 5) & 0x03) == 0 ? "2048 nA" :
                    ((spo2_reg >> 5) & 0x03) == 1 ? "4096 nA" :
                    ((spo2_reg >> 5) & 0x03) == 2 ? "8192 nA" : "16384 nA");
            xprintf("  Raw sample rate = %u Hz\r\n", raw_rate);
            xprintf("  FIFO averaging  = %d\r\n", avg);
            xprintf("  Effective rate  = %u Hz\r\n", eff_sr);
            xprintf("\r\n");
            xprintf("Compare:\r\n");
            xprintf("  Code assumes:     %d Hz\r\n", PPG_SAMPLE_RATE_HZ);
            xprintf("  Hardware produces: %u Hz\r\n", eff_sr);
            if (eff_sr != PPG_SAMPLE_RATE_HZ) {
                xprintf("  Scaling factor:    %.1f\r\n", factor);
                xprintf("\r\n");
                xprintf("*** MISMATCH: All RR intervals will be off by factor %.1f ***\r\n", factor);
                xprintf("*** Example: true HR=75bpm (RR=800ms) will report as %.0fbpm (RR=%.0fms) ***\r\n",
                        75.0f * factor, 800.0f / factor);
            } else {
                xprintf("  MATCH OK\r\n");
            }
            xprintf("--------------------------------\r\n");
        }
    }
#else
    xprintf("MAX30102 disabled (USE_MAX30102=0), HRV features will be zero.\n");
#endif

    //----------------------------------------------------------
    // Sleep staging initialization
    //----------------------------------------------------------
    /* Rekam waktu mulai dari UART. Laptop mengirim "START=<unix_timestamp>"
     * (dengan newline) tepat saat mulai merekam, agar time_of_night benar.
     * Jika tidak ada data / command tidak valid, pakai nilai default historis. */
    recording_start_unix = 1753052400; /* default kalau laptop tidak kirim */
    {
        char start_line[40];
        size_t got = 0;
        /* Baca non-blocking sampai newline atau penuh buffer */
        while (got < sizeof(start_line) - 1)
        {
            char c = 0;
            if (read_bytes_nonblock(&c, 1) != 0) /* != EL_OK */
            {
                break; /* tidak ada data tersedia */
            }
            if (c == '\n' || c == '\r')
            {
                break;
            }
            start_line[got++] = c;
        }
        start_line[got] = 0;

        if (got > 0)
        {
            /* Format: "START=<decimal>" */
            if (strncmp(start_line, "START=", 6) == 0)
            {
                unsigned long ts = 0;
                int idx = 6;
                while (start_line[idx] >= '0' && start_line[idx] <= '9')
                {
                    ts = ts * 10UL + (unsigned long)(start_line[idx] - '0');
                    idx++;
                }
                if (ts > 0)
                {
                    recording_start_unix = (uint32_t)ts;
                    xprintf("Recording start time set from UART: %lu\n",
                            (unsigned long)recording_start_unix);
                }
            }
        }
    }
    epoch_index = 0;
    current_timestamp_ms = 0;

    total_epochs = (8 * 60 * 60) / EPOCH_LENGTH_SEC;

    accel_buffer_reset();
#if USE_MAX30102
    ppg_buffer_reset();
    hr_window_reset();
    rr_buffer_reset();
    rr_history_init();
#endif

    // Session-global cumulative beat timestamp (ms).
    // Grows monotonically across epochs.
    uint32_t cumulative_beat_ts = 0;

    memset(&movement_features, 0, sizeof(movement_features));
    memset(&hrv_features, 0, sizeof(hrv_features));
    memset(&temporal_features, 0, sizeof(temporal_features));
    memset(&g_feature_seq, 0, sizeof(g_feature_seq));
    g_feature_seq_filled = 0;

    xprintf("Sleep-stage preprocessing initialized.\n");
    xprintf("Using SLIDING WINDOW inference: prediction every 30s after %d min buffer\n",
            SEQ_LEN * EPOCH_LENGTH_SEC / 60);

    // Variables for PPG signal quality monitoring
    static uint32_t ppg_min = UINT32_MAX;
    static uint32_t ppg_max = 0;
    static uint32_t sample_count = 0;

    while (1)
    {
        //----------------------------------------------------------
        // 1) Collect accelerometer @ 50Hz
        //----------------------------------------------------------
        if (mpu6050_get_accel_axis(&accel) == IIC_ERR_OK)
        {
            float ax = accel.x / 16384.0f;
            float ay = accel.y / 16384.0f;
            float az = accel.z / 16384.0f;
            (void)accel_buffer_add(ax, ay, az);
            
            // Print ACCEL dengan 3 desimal
            int32_t ax_int = (int32_t)(ax * 1000.0f);
            int32_t ay_int = (int32_t)(ay * 1000.0f);
            int32_t az_int = (int32_t)(az * 1000.0f);
            
            int32_t ax_abs = (ax_int < 0) ? -ax_int : ax_int;
            int32_t ay_abs = (ay_int < 0) ? -ay_int : ay_int;
            int32_t az_abs = (az_int < 0) ? -az_int : az_int;
            
            xprintf("ACCEL,%s%d.%03d,%s%d.%03d,%s%d.%03d\r\n",
                    (ax_int < 0) ? "-" : "", (int)(ax_abs / 1000), (int)(ax_abs % 1000),
                    (ay_int < 0) ? "-" : "", (int)(ay_abs / 1000), (int)(ay_abs % 1000),
                    (az_int < 0) ? "-" : "", (int)(az_abs / 1000), (int)(az_abs % 1000));
        }

#if USE_MAX30102
        //----------------------------------------------------------
        // 2) Collect MAX30102 samples with interpolation 25→50Hz
        //----------------------------------------------------------
        uint8_t fifo_count = 0;
        uint32_t hr_sample = 0;
        
        // Static variables for interpolation (persistent across loop iterations)
        static uint32_t ppg_prev = 0;
        static uint32_t ppg_curr = 0;
        static bool ppg_valid = false;
        
        if (max30102_available_samples(&fifo_count) == 0 && fifo_count > 0)
        {
            while (fifo_count-- > 0)
            {
                if (max30102_read_hr_sample(&hr_sample) != 0)
                {
                    xprintf("READ_ERR\r\n");
                    break;
                }
                
                // Monitor signal quality
                sample_count++;
                if (hr_sample < ppg_min) ppg_min = hr_sample;
                if (hr_sample > ppg_max) ppg_max = hr_sample;
                
                if (sample_count % 100 == 0) {
                    xprintf("PPG_STATS: min=%lu, max=%lu, range=%lu\r\n",
                            (unsigned long)ppg_min,
                            (unsigned long)ppg_max,
                            (unsigned long)(ppg_max - ppg_min));
                    ppg_min = UINT32_MAX;
                    ppg_max = 0;
                }
                
                // Add original sample to buffer (for peak detection)
                (void)ppg_buffer_add(hr_sample);
                
                // Update interpolation state
                ppg_prev = ppg_curr;
                ppg_curr = hr_sample;
                
                if (ppg_valid)
                {
                    // Output interpolated sample (midpoint) - 50Hz output
                    uint32_t interpolated = (ppg_prev + ppg_curr) / 2;
                    xprintf("PPG,%lu\r\n", (unsigned long)interpolated);
                }
                
                // Output original sample
                xprintf("PPG,%lu\r\n", (unsigned long)hr_sample);
                
                ppg_valid = true;
            }
        }
#endif /* USE_MAX30102 */

        //----------------------------------------------------------
        // 3) Peak detect + RR/HRV update when PPG epoch full
        //----------------------------------------------------------
#if USE_MAX30102
        if (ppg_buffer_size() >= PPG_EPOCH_SAMPLES)
        {
                        /* IMPROVED PPG PROCESSING: high-pass filter + improved peak detector */
            uint16_t num_samples = ppg_buffer_size();
            const uint32_t *ppg_raw = ppg_buffer_get();

            /* Step 1: Convert raw uint32_t PPG to float (no DC removal -- HP filter handles it) */
            float ppg_float[PPG_EPOCH_SAMPLES];
            for (uint16_t i = 0; i < num_samples; i++) {
                ppg_float[i] = (float)ppg_raw[i];
            }

            /* Step 2: High-pass filter (0.5 Hz) to remove baseline drift */
            if (!g_hp_filter_initialized) {
                highpass_filter_init(&g_hp_filter, (float)PPG_SAMPLE_RATE_HZ, 0.5f);
                g_hp_filter_initialized = true;
            }
            float ppg_hp[PPG_EPOCH_SAMPLES];
            highpass_filter_process_block(&g_hp_filter, ppg_float, ppg_hp, num_samples);

            /* Step 3: Improved peak detector (MA-5, threshold 15%%, quality index) */
            peak_improved_result_t peak_result;
            peak_detector_improved_process(
                ppg_hp, num_samples,
                (float)PPG_SAMPLE_RATE_HZ,
                &peak_result);

            /* Debug output */
            xprintf("PEAK_DEBUG: peaks=%u, rr=%u\r\n",
                    (unsigned int)peak_result.peak_count,
                    (unsigned int)peak_result.rr_count,
                    peak_result.threshold,
                    peak_result.amplitude,
                    peak_result.signal_quality);

            if (peak_result.rr_count > 0) {
                for (uint16_t i = 0; i < peak_result.rr_count; i++) {
                    float hr = 60000.0f / peak_result.rr_ms[i];
                    xprintf("RR[%u]=%ums (HR=%ubpm)\r\n",
                            (unsigned int)i,
                            (unsigned int)peak_result.rr_ms[i],
                            (unsigned int)hr);
                }
            }

            /* Prepare next epoch PPG buffer */
            ppg_buffer_reset();

            /* Update RR/HR window only if quality acceptable */
            rr_buffer_reset();
            hr_window_reset();

            if (peak_result.rr_count > 0 &&
                peak_result.signal_quality >= SIGNAL_QUALITY_MIN)
            {
                for (uint16_t i = 0; i < peak_result.rr_count; i++)
                {
                    (void)rr_buffer_add(peak_result.rr_ms[i]);

                    rr_history_add((uint16_t)peak_result.rr_ms[i], cumulative_beat_ts);
                    cumulative_beat_ts += (uint32_t)peak_result.rr_ms[i];

                    float bpm = 60000.0f / peak_result.rr_ms[i];
                    (void)hr_window_add((float)current_epoch_ms() / 1000.0f, bpm);
                }
                xprintf("HRV EPOCH OK: rr=%u\r\n",
                        (unsigned int)peak_result.rr_count);
            }
            else
            {
                xprintf("HRV SKIP: rr=%u (quality below min)\r\n",
                        (unsigned int)peak_result.rr_count);
            }

#if HRV_USE_RR_HISTORY
                {
                    uint16_t hrv_count = 0;

                    if (prepare_hrv_input_from_history(
                            rr_hist_float,
                            beat_ts_hist,
                            &hrv_count))
                    {
                        (void)hrv_features_extract(
                            rr_hist_float,
                            beat_ts_hist,
                            (uint32_t)hrv_count,
                            &hrv_features);
                    }
                    else
                    {
                        /* Fallback: rr_buffer (epoch-local) */
                        uint32_t beat_ts_fb[RR_BUFFER_MAX];
                        float cum_ms = 0.0f;
                        uint16_t fb_sz = rr_buffer_size();
                        const float *fb_rr = rr_buffer_get();
                        for (uint16_t bi = 0; bi < fb_sz; bi++)
                        {
                            beat_ts_fb[bi] = (uint32_t)cum_ms;
                            cum_ms += fb_rr[bi];
                        }

                        xprintf("HRV source: RR_BUFFER (%u intervals)\r\n",
                                (unsigned int)fb_sz);

                        (void)hrv_features_extract(
                            fb_rr,
                            beat_ts_fb,
                            fb_sz,
                            &hrv_features);
                    }
                }
#else
                {
                    /* Original epoch-local path (HRV_USE_RR_HISTORY == 0) */
                    uint32_t beat_ts[RR_BUFFER_MAX];
                    float cum_ms = 0.0f;
                    uint16_t rr_sz = rr_buffer_size();
                    const float *rr_ms_ptr = rr_buffer_get();
                    for (uint16_t bi = 0; bi < rr_sz; bi++)
                    {
                        beat_ts[bi] = (uint32_t)cum_ms;
                        cum_ms += rr_ms_ptr[bi];
                    }

                    (void)hrv_features_extract(
                        rr_ms_ptr,
                        beat_ts,
                        rr_sz,
                        &hrv_features);
                }
#endif
        }
#endif /* USE_MAX30102 */

        //----------------------------------------------------------
        // 4) Sleep staging epoch (30 sec) => extract & print in order
        //----------------------------------------------------------
        if (current_timestamp_ms >=
            (epoch_index + 1) *
            EPOCH_LENGTH_SEC *
            1000)
        {
            movement_extract(
                accel_buffer_get(),
                accel_buffer_size(),
                &movement_features);

            temporal_extract(
                epoch_index,
                total_epochs,
                recording_start_unix,
                &temporal_features);

//----------------------------------------------------------
            // Rolling-feature updates (mirror Python FeatureExtraction)
            //----------------------------------------------------------
            /* HR trend history for rolling_mean_hr / rolling_hr_range /
             * hr_slope / hr_delta */
            float current_mean_hr = hrv_features.mean_hr;
            if (g_hr_history_count >= HR_TREND_WINDOW)
            {
                memmove(&g_hr_history[0], &g_hr_history[1],
                        sizeof(float) * (HR_TREND_WINDOW - 1));
                g_hr_history[HR_TREND_WINDOW - 1] = current_mean_hr;
            }
            else
            {
                g_hr_history[g_hr_history_count++] = current_mean_hr;
            }

            /* Accel mean history for rolling_mean_acc / rolling_std_acc */
            if (g_accel_mean_history_count >= ACCEL_HISTORY_LEN)
            {
                memmove(&g_accel_mean_history[0], &g_accel_mean_history[1],
                        sizeof(float) * (ACCEL_HISTORY_LEN - 1));
                g_accel_mean_history[ACCEL_HISTORY_LEN - 1] =
                    movement_features.mean_acc;
            }
            else
            {
                g_accel_mean_history[g_accel_mean_history_count++] =
                    movement_features.mean_acc;
            }

            /* rolling values = history + [current] */
            int n_hr = (int)g_hr_history_count;
            float rolling_mean_hr  = mean_f(g_hr_history, n_hr);
            float rolling_hr_range = 0.0f;
            float hr_slope         = 0.0f;
            float hr_delta         = 0.0f;

            if (n_hr >= 2)
            {
                float hmin = g_hr_history[0];
                float hmax = g_hr_history[0];
                for (int i = 1; i < n_hr; i++)
                {
                    if (g_hr_history[i] < hmin) hmin = g_hr_history[i];
                    if (g_hr_history[i] > hmax) hmax = g_hr_history[i];
                }
                rolling_hr_range = hmax - hmin;
                hr_slope = linear_fit_slope(g_hr_history, n_hr);
            }

            /* hr_delta = current_mean_hr - prev_mean_hr (from prior epoch) */
            if (epoch_index > 0)
                hr_delta = current_mean_hr - g_prev_mean_hr;
            g_prev_mean_hr = current_mean_hr;

            int n_acc = (int)g_accel_mean_history_count;
            float rolling_mean_acc = mean_f(g_accel_mean_history, n_acc);
            float rolling_std_acc  = std_f(g_accel_mean_history, n_acc);

            float sin_time_of_night =
                compute_sin_time_of_night(temporal_features.time_of_night);

            // Print features in exact 18-feature order
#define SCALED(x)  ((int32_t)((x) * 1000000.0f))
#define F6_INT(x)  (SCALED(x) / 1000000)
#define F6_FRAC(x) ((SCALED(x) >= 0) ? (SCALED(x) % 1000000) : (-(SCALED(x)) % 1000000))
            
            xprintf("STATUS,epoch,%lu,uptime,%lu\r\n",
                (unsigned long)(epoch_index + 1),
                (unsigned long)(current_timestamp_ms / 1000));
            xprintf(
                "F=%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d,"
                "%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d,"
                "%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d,%d.%06d\r\n",
                F6_INT(temporal_features.relative_position), F6_FRAC(temporal_features.relative_position),
                F6_INT(hrv_features.sd2), F6_FRAC(hrv_features.sd2),
                F6_INT(sin_time_of_night), F6_FRAC(sin_time_of_night),
                F6_INT(rolling_mean_hr), F6_FRAC(rolling_mean_hr),
                F6_INT(hrv_features.rmssd), F6_FRAC(hrv_features.rmssd),
                F6_INT(temporal_features.time_of_night), F6_FRAC(temporal_features.time_of_night),
                F6_INT(movement_features.energy), F6_FRAC(movement_features.energy),
                F6_INT(rolling_hr_range), F6_FRAC(rolling_hr_range),
                F6_INT(movement_features.acceleration_jerk), F6_FRAC(movement_features.acceleration_jerk),
                F6_INT(rolling_mean_acc), F6_FRAC(rolling_mean_acc),
                F6_INT(movement_features.rms), F6_FRAC(movement_features.rms),
                F6_INT(hrv_features.lf), F6_FRAC(hrv_features.lf),
                F6_INT(rolling_std_acc), F6_FRAC(rolling_std_acc),
                F6_INT(movement_features.zero_crossing), F6_FRAC(movement_features.zero_crossing),
                F6_INT(hr_slope), F6_FRAC(hr_slope),
                F6_INT(hrv_features.hf), F6_FRAC(hrv_features.hf),
                F6_INT(hrv_features.lf_hf), F6_FRAC(hrv_features.lf_hf),
                F6_INT(hr_delta), F6_FRAC(hr_delta));
                
#undef F6_FRAC
#undef F6_INT
#undef SCALED

            //----------------------------------------------------------
            // 5) Inference stage
            //----------------------------------------------------------
            {
                /* 18-feature vector in EXACTLY the order defined by
                 * SleepStage/configs/train_selected_features.yaml */
                const float feat[NUM_FEATURES] = {
                    temporal_features.relative_position,   /* 1  relative_position   */
                    hrv_features.sd2,                     /* 2  sd2                 */
                    sin_time_of_night,                    /* 3  sin_time_of_night   */
                    rolling_mean_hr,                      /* 4  rolling_mean_hr     */
                    hrv_features.rmssd,                   /* 5  rmssd               */
                    temporal_features.time_of_night,      /* 6  time_of_night       */
                    movement_features.energy,             /* 7  energy              */
                    rolling_hr_range,                     /* 8  rolling_hr_range    */
                    movement_features.acceleration_jerk,  /* 9  acceleration_jerk   */
                    rolling_mean_acc,                     /* 10 rolling_mean_acc    */
                    movement_features.rms,                /* 11 rms                 */
                    hrv_features.lf,                      /* 12 lf                  */
                    rolling_std_acc,                      /* 13 rolling_std_acc     */
                    movement_features.zero_crossing,      /* 14 zero_crossing       */
                    hr_slope,                             /* 15 hr_slope            */
                    hrv_features.hf,                      /* 16 hf                  */
                    hrv_features.lf_hf,                   /* 17 lf_hf               */
                    hr_delta                              /* 18 hr_delta            */
                };

                push_feature_vector(feat);
                run_inference_if_ready();
            }

            epoch_index++;
            accel_buffer_reset();
        }

        // advance time by the same delay
        current_timestamp_ms += 20;
        hx_drv_timer_cm55x_delay_ms(20, TIMER_STATE_DC);
    }

    return 0;
}

/******************************************************************************
 * Deinitialization
 ******************************************************************************/
int cv_mb_cls_deinit()
{
    interpreter = nullptr;
    input = nullptr;
    output = nullptr;
    model = nullptr;
    return 0;
}