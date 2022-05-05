#include <math.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tflite_model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"



#include <stdio.h>

#include "pico/stdlib.h"
#include "hardware/pwm.h"

extern "C" {
#include "pico/pdm_microphone.h"
}

#include "tflite_model.h"

#include "dsp_pipeline.h"
#include "ml_model.h"



int main(){

    stdio_init_all();

    float x = 0.0f;
    float y_true = sin(x);

    // Set up logging
    tflite::MicroErrorReporter micro_error_reporter;

    const tflite::Model* model = ::tflite::GetModel(tflite_model);
    tflite::AllOpsResolver resolver;

    constexpr int kTensorArenaSize = 2000;
    uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
    // Allocate memory from the tensor_arena for the model's tensors
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);


    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;


    int8_t x_quantized = x / input_scale + input_zero_point;
    // Place the quantized input in the model's input tensor
    input->data.int8[0] = x_quantized;

    // Run the model and check that it succeeds
    TfLiteStatus invoke_status = interpreter.Invoke();
    printf("invoke status: ",invoke_status);

    TfLiteTensor* output = interpreter.output(0);
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;

    int8_t y_pred_quantized = output->data.int8[0];
    // Dequantize the output from integer to floating-point
    float y_pred = (y_pred_quantized - output_zero_point) * output_scale;
    x = 1.f;

    while (1){

        
        y_true = sin(x);
        input->data.int8[0] = x / input_scale + input_zero_point;

        printf("input is: %f", x);
        interpreter.Invoke();
        y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
        printf("output is: %f", y_pred);
        printf("\n");
        x = x+0.1f;

        if (x > 6.3f){
            x = 0.f;
        }

    }
}