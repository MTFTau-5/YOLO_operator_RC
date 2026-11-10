#!/usr/bin/env python3
import argparse

import onnx


def convert_nonuniform_splits(model):
    graph = model.graph
    converted = []
    for node in list(graph.node):
        if node.op_type != "Split":
            continue
        split_attr = next((a for a in node.attribute if a.name == "split"), None)
        if split_attr is None:
            continue
        widths = list(onnx.helper.get_attribute_value(split_attr))
        if len(widths) != len(node.output) or len(set(widths)) == 1:
            continue
        axis_attr = next((a for a in node.attribute if a.name == "axis"), None)
        axis = onnx.helper.get_attribute_value(axis_attr) if axis_attr else 0
        if axis < 0:
            raise ValueError(f"{node.name}: 负 axis 需要先按输入秩转换为正数")

        prefix = node.name or "Split"
        slices = []
        start = 0
        for index, (output, width) in enumerate(zip(node.output, widths)):
            names = [f"{prefix}_slice{index}_{suffix}" for suffix in ("starts", "ends", "axes")]
            tensors = [
                onnx.helper.make_tensor(names[0], onnx.TensorProto.INT64, [1], [start]),
                onnx.helper.make_tensor(names[1], onnx.TensorProto.INT64, [1], [start + width]),
                onnx.helper.make_tensor(names[2], onnx.TensorProto.INT64, [1], [axis]),
            ]
            graph.initializer.extend(tensors)
            slices.append(onnx.helper.make_node(
                "Slice", [node.input[0], *names], [output],
                name=f"{prefix}_slice{index}"))
            start += width

        index = list(graph.node).index(node)
        for offset, slice_node in enumerate(slices):
            graph.node.insert(index + offset, slice_node)
        graph.node.remove(node)
        converted.append((node.name, widths))
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="把非等分 Split 改为 Slice，规避 SpaceMIT EP 的错误形状推导")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    converted = convert_nonuniform_splits(model)
    if not converted:
        raise SystemExit("没有发现需要转换的非等分 Split")
    for name, widths in converted:
        print(f"转换 {name}: split={widths} -> Slice")
    onnx.checker.check_model(model)
    onnx.save_model(model, args.output)
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
