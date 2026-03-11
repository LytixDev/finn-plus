from onnxscript import ir
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.util.logging import log


def dtype_fits_in(narrow_dt, wide_dt):
    """Check if all values representable by narrow_dt fit in wide_dt."""
    if narrow_dt.min() < wide_dt.min():
        return False
    if narrow_dt.max() > wide_dt.max():
        return False
    return True


def _next_byte_aligned_dtype(dt):
    """Return the smallest byte-aligned DataType that can represent all values of dt."""
    bw = dt.bitwidth()
    if bw % 8 == 0:
        return dt
    new_bw = ((bw + 7) // 8) * 8
    if dt.signed():
        return DataType[f"INT{new_bw}"]
    else:
        return DataType[f"UINT{new_bw}"]


# TODO: Surely this could be done in a better way.
def _set_node_output_dtype(node_inst, dt):
    """Set the output datatype on a HWCustomOp node, trying known attribute names."""
    try:
        node_inst.set_nodeattr("out_dtype", dt.name)
    except Exception:
        try:
            node_inst.set_nodeattr("outputDataType", dt.name)
        except Exception:
            log.warning(
                f"Could not set output dtype on {node_inst.onnx_node.name} "
                f"({node_inst.onnx_node.op_type})"
            )


# TODO: Again, surely this could be done in a better way.
def _set_node_input_dtype(node_inst, dt):
    """Set the input datatype on a HWCustomOp node, trying known attribute names."""
    try:
        node_inst.set_nodeattr("inputDataType", dt.name)
    except Exception:
        log.warning(
            f"Could not set input dtype on {node_inst.onnx_node.name} "
            f"({node_inst.onnx_node.op_type})"
        )


def _derive_minimal_odt_from_ir(last_node):
    # TODO: THIS IS STUPID AND VERY TEMPORARY
    #       Need to handle this generally and properly.
    op_type = last_node.op_type

    # Only handle known op types where we can compute the minimum
    if "ElementwiseAdd" not in op_type:
        return None

    # ElementwiseAdd has two data inputs (lhs, rhs)
    input_dtypes = []
    for inp in last_node.inputs:
        meta = inp.meta.get("quant_parameter_tensor_names", {})
        dt_str = meta.get("finn_datatype")
        if dt_str is not None:
            input_dtypes.append(DataType[dt_str])

    if len(input_dtypes) < 2:
        return None

    lhs_dt, rhs_dt = input_dtypes[0], input_dtypes[1]
    max_width = max(lhs_dt.bitwidth(), rhs_dt.bitwidth())
    signed = any([lhs_dt.signed(), rhs_dt.signed()])
    out_width = max_width + 1

    if signed:
        return DataType[f"INT{out_width}"]
    else:
        return DataType[f"UINT{out_width}"]


def match_loop_body_template_dtypes(loop_body_template):
    # Relax dtype requirements for input and output streams by upcasting
    g = loop_body_template._ir_graph

    first_node = g._nodes[0]
    last_node = g._nodes[-1]
    idt_str = first_node.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]
    odt_str = last_node.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]

    log.info(f"match_loop_body_template_dtypes: template idt={idt_str}, odt={odt_str}")

    if idt_str == odt_str:
        log.info("match_loop_body_template_dtypes: dtypes already match, nothing to do")
        return

    idt = DataType[idt_str]
    last_node = g._nodes[-1]

    # TODO: this is stupid
    minimal_odt = _derive_minimal_odt_from_ir(last_node)
    if minimal_odt is not None:
        log.info(f"match_loop_body_template_dtypes: minimal output dtype = {minimal_odt}")
        if not dtype_fits_in(minimal_odt, idt):
            raise ValueError(
                f"match_loop_body_template_dtypes: Cannot match loop body dtypes. "
                f"Minimal output dtype {minimal_odt} does not fit in input dtype {idt}. "
                f"Output range [{minimal_odt.min()}, {minimal_odt.max()}] exceeds "
                f"input range [{idt.min()}, {idt.max()}]."
            )
    else:
        log.warning(
            f"match_loop_body_template_dtypes: Cannot derive minimal output dtype for "
            f"{last_node.op_type}."
        )
        return

    # Here we know the output dtype can safely be narrowed to the input dtype

    # Update the last node's output tensor finn_datatype metadata
    last_node.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str

    # Also update the graph-level input/output metadata, which build_loop_replace_pattern reads
    if "quant_parameter_tensor_names" not in g.inputs[0].meta:
        g.inputs[0].meta["quant_parameter_tensor_names"] = {}
    g.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str
    if "quant_parameter_tensor_names" not in g.outputs[0].meta:
        g.outputs[0].meta["quant_parameter_tensor_names"] = {}
    g.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str

    # Update the last node's out_dtype attribute
    if "out_dtype" in last_node.attributes:
        last_node.attributes["out_dtype"] = ir.Attr("out_dtype", ir.AttributeType.STRING, idt_str)
        log.info(
            f"match_loop_body_template_dtypes: Updated last node {last_node.op_type} "
            f"out_dtype from {odt_str} to {idt_str}"
        )
    elif "outputDataType" in last_node.attributes:
        last_node.attributes["outputDataType"] = ir.Attr(
            "outputDataType", ir.AttributeType.STRING, idt_str
        )
        log.info(
            f"match_loop_body_template_dtypes: Updated last node {last_node.op_type} "
            f"outputDataType from {odt_str} to {idt_str}"
        )
    else:
        log.warning(
            f"match_loop_body_template_dtypes: Last node {last_node.op_type} has no "
            f"out_dtype or outputDataType attribute to update"
        )


class EnforceLoopBodyDtypeConstraint(Transformation):
    """Transformation to be ran after minimize bit widths to ensure FINNLoop body input/output
    dtypes are valid for the MLO infrastructure.

    Two constraints are enforced:
      1. Input and output dtypes of the loop body must match.
         If minimize_bit_width narrowed the output, we widen it back to match the input.
         TODO: If it narrowed the input, we widen it back.
      2. The loop body output (and the input as well from constraint 1) must be byte-aligned 
         because MLO uses AXI-MM DMA for intermediate feature maps between iterations.
         If not byte-aligned, we upcast to the next byte-aligned dtype and propagate the
         change to the FINNLoop node, the loop body's first/last nodes, and the adjacent
         top-level nodes.

    Is automatically ran after step_minimize_bit_width when MLO is active.
    """

    def apply(self, model):
        for node in model.graph.node:
            if node.op_type != "FINNLoop":
                continue

            inst = getCustomOp(node)
            loop_body = inst.get_nodeattr("body")

            first_node = loop_body.graph.node[0]
            last_node = loop_body.graph.node[-1]
            first_inst = getCustomOp(first_node)
            last_inst = getCustomOp(last_node)

            idt = first_inst.get_input_datatype(0)
            odt = last_inst.get_output_datatype(0)

            # Constraint 1: match input/output dtypes
            if idt != odt:
                # Loop rolling only succeeds if idt == odt, meaning some later transformation,
                # most likely minimize bit width, has altered that.
                log.warning(
                    f"EnforceLoopBodyDtypeConstraint: FINNLoop {node.name} has "
                    f"mismatched body dtypes: input={idt}, output={odt}. "
                    f"Widening output back to {idt}."
                )

                # This is probably not possible to hit?
                if not dtype_fits_in(odt, idt):
                    raise ValueError(
                        f"EnforceLoopBodyDtypeConstraint: Cannot fix dtype mismatch "
                        f"for FINNLoop {node.name}. Output dtype {odt} does not fit "
                        f"in input dtype {idt}."
                    )

                _set_node_output_dtype(last_inst, idt)
                output_tensor = last_node.output[0]
                loop_body.set_tensor_datatype(output_tensor, idt)
                # get_nodeattr("body") returns a new ModelWrapper copy each time,
                # so we must write the modified graph back to the FINNLoop node
                inst.set_nodeattr("body", loop_body.graph)
                odt = idt

            # Constraint 2: byte-aligned outputs (and inputs)
            # TODO: Something to think about, is upcasting (essentially zero-padding) always OK?
            aligned_dt = _next_byte_aligned_dtype(idt)
            if aligned_dt != idt:
                log.warning(
                    f"EnforceLoopBodyDtypeConstraint: FINNLoop {node.name} has "
                    f"non-byte-aligned I/O dtype {idt} ({idt.bitwidth()} bits). "
                    f"Upcasting to {aligned_dt} ({aligned_dt.bitwidth()} bits)."
                )

                # Update the first node in the subgraph's input dtype
                _set_node_input_dtype(first_inst, aligned_dt)
                input_tensor = first_node.input[0]
                loop_body.set_tensor_datatype(input_tensor, aligned_dt)

                # Update the last node in the subgraph's output dtype
                _set_node_output_dtype(last_inst, aligned_dt)
                output_tensor = last_node.output[0]
                loop_body.set_tensor_datatype(output_tensor, aligned_dt)

                # Update FINNLoop's own attributes
                inst.set_nodeattr("inputDataType", aligned_dt.name)
                inst.set_nodeattr("outputDataType", aligned_dt.name)

                # Update the top-level tensor annotations on the FINNLoop's I/O
                finnloop_input = node.input[0]
                finnloop_output = node.output[0]
                model.set_tensor_datatype(finnloop_input, aligned_dt)
                model.set_tensor_datatype(finnloop_output, aligned_dt)

                # Update the node above the FINNLoop
                producer = model.find_producer(finnloop_input)
                if producer is not None:
                    producer_inst = getCustomOp(producer)
                    _set_node_output_dtype(producer_inst, aligned_dt)
                    log.info(
                        f"  Updated producer {producer.name} ({producer.op_type}) "
                        f"output dtype to {aligned_dt}"
                    )

                # Update the node below the FINNLoop
                consumers = model.find_consumers(finnloop_output)
                if consumers:
                    for consumer in consumers:
                        consumer_inst = getCustomOp(consumer)
                        _set_node_input_dtype(consumer_inst, aligned_dt)
                        log.info(
                            f"  Updated consumer {consumer.name} ({consumer.op_type}) "
                            f"input dtype to {aligned_dt}"
                        )

                inst.set_nodeattr("body", loop_body.graph)
            else:
                # dtypes match and are byte-aligned
                # Update the node attrs on the loop node just in case the previous transformation(s)
                # didn't do this properly
                inst.set_nodeattr("inputDataType", idt.name)
                inst.set_nodeattr("outputDataType", idt.name)

        return model, False
