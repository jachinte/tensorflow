/*
 * demo.c
 * To compile: gcc nanojpeg.c demo.c -ltensorflow
 * To run: ./a.out data/inception_v3_2016_08_28_frozen.pb data/imagenet_slim_labels.txt data/grace_hopper.jpg
 *
 * author: Miguel Jiménez
 * date: May 9, 2017
 */
#define _NJ_INCLUDE_HEADER_ONLY

#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include "nanojpeg.c"

/*
 * check_status_ok
 * description:
 * Verifies status OK after each TensorFlow operation.
 * parameters:
 *     input status - the TensorFlow status
 *     input step   - a description of the last operation performed
 */
void check_status_ok(TF_Status* status, char* step) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(status));
        fprintf(stderr, "Error message: %s\n", TF_Message(status));
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

/*
 * check_result_ok
 * description:
 * Verifies result OK after decoding an image.
 * parameters:
 *     input result - the nanojpeg result
 *     input step   - a description of operation performed
 */
void check_result_ok(enum _nj_result result, char* step) {
    if (result != NJ_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, result);
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

/*
 * file_length
 * description:
 * Returns the length of the given file.
 * parameters:
 *     input file - any file
 */
unsigned long file_length(FILE* file) {
    fseek(file, 0, SEEK_END);
    unsigned long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    return length;
}

/*
 * load_file
 * description:
 * Loads a binary buffer from the given file
 * parameters:
 *     input file  - a binary file
 *     inut length - the length of the file
 */
char* load_file(FILE* file, unsigned long length) {
    char* buffer;
    buffer = (char *) malloc(length + 1);
    if (!buffer) {
        fprintf(stderr, "Memory error while reading buffer");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fread(buffer, length, 1, file);
    return buffer;
}

/*
 * 1. Initialize TensorFlow session
 * 2. Read in the previosly exported graph
 * 3. Read tensor from image
 * 4. Run the image through the model
 * 5. Print top labels
 * 6. Close session to release resources
 *
 * Note: the input image is not automatically resized. A jpg image is expected,
 * with the same dimensions of the images in the trained model.
 *
 * arguments:
 *     graph  - a file containing the TensorFlow graph
 *     labels - a file containing the list of labels
 *     image  - an image to test the model
 */
int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "3 arguments expected, %d received\n", argc - 1);
        exit(EXIT_FAILURE);
    }

    char* input_graph = argv[1];
    char* input_labels = argv[2];
    char* input_image = argv[3];

    // int input_width = 299;
    // int input_height = 299;
    // int input_mean = 0;
    // int input_std = 255;

    // 1. Initialize TensorFlow session
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Status* status = TF_NewStatus();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    check_status_ok(status, "Initialization of TensorFlow session");

    // 2. Read in the previosly exported graph
    FILE* pb_file = fopen(input_graph, "rb");
    if (!pb_file) {
        fprintf(stderr, "Could not read graph file \"%s\"\n", input_graph);
        exit(EXIT_FAILURE);
    }
    unsigned long pb_file_length = file_length(pb_file);
    char* pb_file_buffer = load_file(pb_file, pb_file_length);
    TF_Buffer* graph_def = TF_NewBufferFromString(pb_file_buffer, pb_file_length);
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    check_status_ok(status, "Loading of .pb graph");

    // 3. Read tensor from image
    njInit();
    FILE* image_file = fopen(input_image, "rb");
    if (!image_file) {
        fprintf(stderr, "Could not read image file \"%s\"\n", input_image);
        exit(EXIT_FAILURE);
    }
    unsigned long image_file_length = file_length(image_file);
    char* image_file_buffer = load_file(image_file, image_file_length);
    nj_result_t result = njDecode(image_file_buffer, image_file_length);
    check_result_ok(result, "Loading of test image");
    unsigned char* image_data = njGetImage();
    int image_data_length = njGetImageSize();
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel].
    int ndims = 4, channels_in_image = 3;
    int64_t dims[] = {1, (long) njGetHeight(), (long) njGetWidth(), channels_in_image};
    TF_Tensor* tensor = TF_NewTensor(TF_FLOAT, dims, ndims, image_data, image_data_length, NULL, NULL);

    // 4. Run the image through the model
    TF_Output output1;
    output1.oper = TF_GraphOperationByName(graph, "input");
    output1.index = 0;
    TF_Output* inputs = {&output1};
    TF_Tensor* const* input_values = {&tensor};

    const TF_Operation* target_op = TF_GraphOperationByName(graph, "InceptionV3/Predictions/Reshape_1");
    TF_Output output2;
    output2.oper = (void *) target_op;
    output2.index = 0;
    TF_Output* outputs = {&output2};
    TF_Tensor* output_values;

    const TF_Operation* const* target_opers = {&target_op};
    TF_SessionRun(
        session,
        NULL,
        inputs, input_values, 1,
        outputs, &output_values, 1,
        target_opers, 1,
        NULL,
        status
    );
    check_status_ok(status, "Running of the image through the model");

    // 6. Close session to release resources
    fclose(pb_file);
    free(pb_file_buffer);
    njDone(); // resets NanoJPEG's internal state and frees memory
    return EXIT_SUCCESS;
}
