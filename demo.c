/*
 * demo.c
 * To compile: gcc demo.c -ltensorflow
 * To run: ./a.out ./data/inception_v3_2016_08_28_frozen.pb ./data/imagenet_slim_labels.txt ./data/img.jpg
 *
 * author: Miguel Jiménez
 * date: May 9, 2017
 */
#include <stdlib.h>
#include <stdio.h>
#include <tensorflow/c/c_api.h>

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
    // From http://stackoverflow.com/a/238609/738968
    fseek(file, 0, SEEK_END);
    unsigned long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    return length;
}

/*
 * load_graph_def
 * description:
 * Loads a binary buffer from the given file
 * parameters:
 *     input file  - a binary file containing a TensorFlow graph
 *     inut length - the length of the file
 */
char* load_graph_def(FILE* file, unsigned long length) {
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
 * 5. Run the image through the model
 * 6. Print top labels
 * 7. Close session to release resources
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
        fprintf(stderr, "3 arguments expected, %d received\n", argc);
        exit(EXIT_FAILURE);
    }

    char* input_graph = argv[1];
    char* input_labels = argv[2];
    char* input_image = argv[3];

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
    unsigned long length = file_length(pb_file);
    char* buffer = load_graph_def(pb_file, length);
    TF_Buffer* graph_def = TF_NewBufferFromString(buffer, length);
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
    check_status_ok(status, "Loading .pb graph");

    // 3. Read tensor from image
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L87
    // https://medium.com/jim-fleming/loading-tensorflow-graphs-via-host-languages-be10fd81876f
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h#L226

    // ...

    // 7. Close session to release resources
    fclose(pb_file);
    free(buffer);
    return EXIT_SUCCESS;
}
