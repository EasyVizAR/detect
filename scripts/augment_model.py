import os
import sys

import onnx


#bboxes_layer = "/model.22/Mul_output_0"
bboxes_layer = "/model.22/Mul_2_output_0"
scores_layer = "/model.22/Sigmoid_output_0"


def add_reduce_max_layer(model):
    max_node = onnx.helper.make_node(
        "ReduceMax",
        inputs=[scores_layer],
        outputs=["max_score"],
        axes=[2],
        keepdims=0
    )

    max_shape = [1, 80]
    max_value_info = onnx.helper.make_tensor_value_info("max_score", onnx.TensorProto.FLOAT, shape=max_shape)

    model.graph.node.append(max_node)
    model.graph.output.append(max_value_info)


def add_nms_layer(model):
    # make constant tensors
    score_threshold = onnx.helper.make_tensor("score_threshold", onnx.TensorProto.FLOAT, [1], [0.65])
    iou_threshold = onnx.helper.make_tensor("iou_threshold", onnx.TensorProto.FLOAT, [1], [0.3])
    max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", onnx.TensorProto.INT64, [1], [200])

    transpose_boxes = onnx.helper.make_node(
        "Transpose",
        inputs=[bboxes_layer],
        outputs=["transposed_boxes"],
        perm=[0,2,1]
    )

    nms_node = onnx.helper.make_node(
        "NonMaxSuppression",
        inputs=["transposed_boxes", scores_layer, "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        outputs=["selected_indices"],
        center_point_box=1
    )

    model.graph.node.append(transpose_boxes)
    model.graph.node.append(nms_node)

    nms_shape = [None, 3]
    nms_value_info = onnx.helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, shape=nms_shape)
    model.graph.output.append(nms_value_info)

    model.graph.initializer.append(score_threshold)
    model.graph.initializer.append(iou_threshold)
    model.graph.initializer.append(max_output_boxes_per_class)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <onnx file>".format(sys.argv[0]))
        sys.exit(1)

    input_model_file = sys.argv[1]

    model_name = os.path.splitext(input_model_file)[0]

    #
    # Create one model with a ReduceMax layer
    #

    model = onnx.load_model(input_model_file)

    add_reduce_max_layer(model)

    output_file = "{}-max.onnx".format(model_name)
    onnx.save(model, output_file)

    #
    # Make another model with an NMS layer
    #

    model = onnx.load_model(input_model_file)
    #model = onnx.version_converter.convert_version(model, 11)

    add_nms_layer(model)

    output_file = "{}-nms.onnx".format(model_name)
    onnx.save(model, output_file)

