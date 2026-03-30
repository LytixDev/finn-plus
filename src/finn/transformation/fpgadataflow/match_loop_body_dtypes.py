from onnxscript import ir
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.util.logging import log



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


# NICCHANGE:
def match_loop_body_template_dtypes(loop_body_template):
    """
    Constraints:
        1. input and output dtype must match exactly
        2. this dtype must be byte aligned

    If constraint 1. and 2. is not met, find the most narrow valid byte-aligned dtype.

    TODO: If any dtypes are updated, this must also be progated in the model.
    TODO: If the signedness differs we will error. Something to look into later. 
    """
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

    log.warning("match_loop_body_template_dtypes: TODO: loop body idt != odt")
    return
    # TODO: This branch of the function is stupid and not properly tested. Fix later.
    # idt = DataType[idt_str]
    # last_node = g._nodes[-1]

    # minimal_odt = _derive_minimal_odt_from_ir(last_node)
    # if minimal_odt is not None:
    #     log.info(f"match_loop_body_template_dtypes: minimal output dtype = {minimal_odt}")
    #     if not dtype_fits_in(minimal_odt, idt):
    #         raise ValueError(
    #             f"match_loop_body_template_dtypes: Cannot match loop body dtypes. "
    #             f"Minimal output dtype {minimal_odt} does not fit in input dtype {idt}. "
    #             f"Output range [{minimal_odt.min()}, {minimal_odt.max()}] exceeds "
    #             f"input range [{idt.min()}, {idt.max()}]."
    #         )
    # else:
    #     log.warning(
    #         f"match_loop_body_template_dtypes: Cannot derive minimal output dtype for "
    #         f"{last_node.op_type}."
    #     )
    #     return

    # # Here we know the output dtype can safely be narrowed to the input dtype

    # # Update the last node's output tensor finn_datatype metadata
    # last_node.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str

    # # Also update the graph-level input/output metadata, which build_loop_replace_pattern reads
    # if "quant_parameter_tensor_names" not in g.inputs[0].meta:
    #     g.inputs[0].meta["quant_parameter_tensor_names"] = {}
    # g.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str
    # if "quant_parameter_tensor_names" not in g.outputs[0].meta:
    #     g.outputs[0].meta["quant_parameter_tensor_names"] = {}
    # g.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str

    # # Update the last node's out_dtype attribute
    # if "out_dtype" in last_node.attributes:
    #     last_node.attributes["out_dtype"] = ir.Attr("out_dtype", ir.AttributeType.STRING, idt_str)
    #     log.info(
    #         f"match_loop_body_template_dtypes: Updated last node {last_node.op_type} "
    #         f"out_dtype from {odt_str} to {idt_str}"
    #     )
    # elif "outputDataType" in last_node.attributes:
    #     last_node.attributes["outputDataType"] = ir.Attr(
    #         "outputDataType", ir.AttributeType.STRING, idt_str
    #     )
    #     log.info(
    #         f"match_loop_body_template_dtypes: Updated last node {last_node.op_type} "
    #         f"outputDataType from {odt_str} to {idt_str}"
    #     )
    # else:
    #     log.warning(
    #         f"match_loop_body_template_dtypes: Last node {last_node.op_type} has no "
    #         f"out_dtype or outputDataType attribute to update"
    #     )


class EnforceLoopBodyDtypeConstraints(Transformation):
    """
    Two constraints are enforced:
      1. Input and output dtypes of the loop body must match.
      2. The loop body output (and the input as well from constraint 1) must be byte-aligned 
         because MLO uses AXI-MM DMA for intermediate feature maps between iterations.
         If not byte-aligned, we upcast to the next byte-aligned dtype and propagate the
         change to the FINNLoop node, the FINNLoop producer and consumer nodes, and the loop body's 
         first and last nodes.

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
                # most likely minimize bit width, has altered that. Here we just set both dtypes
                # to the widest of the two.
                wider = idt if idt.bitwidth() >= odt.bitwidth() else odt
                log.warning(
                    f"EnforceLoopBodyDtypeConstraints: FINNLoop {node.name} has "
                    f"mismatched body dtypes: input={idt}, output={odt}. "
                    f"Widening both to {wider}."
                )
                idt = wider
                odt = wider

            # Constraint 2: byte-align
            # NOTE: Something to think about, is upcasting (essentially zero-padding) always OK?
            target_dt = _next_byte_aligned_dtype(odt)
            if target_dt != odt:
                log.warning(
                    f"EnforceLoopBodyDtypeConstraints: FINNLoop {node.name} has "
                    f"non-byte-aligned I/O dtype {odt} ({odt.bitwidth()} bits). "
                    f"Widening to {target_dt} ({target_dt.bitwidth()} bits)."
                )

            # Propagate target_dt to all relevant places.
            # 1. Loop body: set the first node's input tensor annotation, then
            #    run InferDataTypes to propagate through all intermediate nodes
            loop_body.set_tensor_datatype(first_node.input[0], target_dt)
            loop_body = loop_body.transform(InferDataTypes())
            # Re-fetch node references since transform returns a new model
            first_node = loop_body.graph.node[0]
            last_node = loop_body.graph.node[-1]
            last_inst = getCustomOp(last_node)

            # 2. Loop body last node: override output dtype to target_dt
            #    (InferDataTypes may have inferred something different)
            _set_node_output_dtype(last_inst, target_dt)
            loop_body.set_tensor_datatype(last_node.output[0], target_dt)

            # 3. Producer node: directly set output dtype
            producer = model.find_producer(node.input[0])
            if producer is not None:
                producer_inst = getCustomOp(producer)
                _set_node_output_dtype(producer_inst, target_dt)

            # 4. FINNLoop node attributes and tensor annotations
            inst.set_nodeattr("inputDataType", target_dt.name)
            inst.set_nodeattr("outputDataType", target_dt.name)
            model.set_tensor_datatype(node.input[0], target_dt)
            model.set_tensor_datatype(node.output[0], target_dt)

            #  Write modified loop body back
            inst.set_nodeattr("body", loop_body.graph)

            # 5. Consumer nodes: tensor annotation is set above, call
            #    infer_node_datatype so they pick up target_dt
            consumers = model.find_consumers(node.output[0])
            if consumers:
                for consumer in consumers:
                    consumer_inst = getCustomOp(consumer)
                    consumer_inst.infer_node_datatype(model)

        return model, False
