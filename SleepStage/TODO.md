# TODO - Fix SimpleRNN TFLite INT8 Quantization

- [x] `src/models/simple_rnn.py`: Add `unroll=True` to SimpleRNN layer to eliminate TensorList/while_loop ops
- [x] `src/training/quantize_tflite.py`: Set `experimental_enable_resource_variables = True` to fix variable constant folding
- [x] `src/training/quantize_tflite.py`: Add pre-conversion SavedModel op diagnostic
- [ ] Test conversion to verify fix
