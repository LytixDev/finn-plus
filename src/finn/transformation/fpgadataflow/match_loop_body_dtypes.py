from onnxscript import ir
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
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
def enforce_loop_body_template_dtype_constraints(loop_body_template):
    """Enforce the same constraints as EnforceLoopBodyDtypeConstraints, but on a
    LoopBodyTemplate (onnxscript IR) before LoopRolling runs.

    Constraints:
        1. input and output dtype must match exactly
        2. this dtype must be byte aligned

    If not met, find the smallest valid byte-aligned dtype (wider of the two),
    propagate it through the template's intermediate nodes via InferDataTypes,
    and rebuild the template's pattern/function.
    """
    g = loop_body_template._ir_graph

    first_node = g._nodes[0]
    last_node = g._nodes[-1]
    idt_str = first_node.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]
    odt_str = last_node.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]

    log.info(f"enforce_loop_body_template_dtype_constraints: template idt={idt_str}, odt={odt_str}")

    idt = DataType[idt_str]
    odt = DataType[odt_str]

    # Constraint 1: pick the wider of the two
    if idt != odt:
        wider = idt if idt.bitwidth() >= odt.bitwidth() else odt
        log.warning(
            f"enforce_loop_body_template_dtype_constraints: "
            f"mismatched dtypes: input={idt}, output={odt}. "
            f"Widening both to {wider}."
        )
        idt = wider
        odt = wider

    # Constraint 2: byte-align
    target_dt = _next_byte_aligned_dtype(odt)
    if target_dt != odt:
        log.warning(
            f"enforce_loop_body_template_dtype_constraints: "
            f"non-byte-aligned I/O dtype {odt} ({odt.bitwidth()} bits). "
            f"Upcasting to {target_dt} ({target_dt.bitwidth()} bits)."
        )

    if target_dt == DataType[idt_str] and target_dt == DataType[odt_str]:
        log.info("enforce_loop_body_template_dtype_constraints: constraints already met")
        return

    target_dt_str = target_dt.name

    # Serialize the template to protobuf, wrap in ModelWrapper, propagate dtypes,
    # then deserialize back to IR.
    loop_body_template.update()
    model = ModelWrapper(loop_body_template._model_proto)

    # Set the input tensor dtype and run InferDataTypes to propagate
    model.set_tensor_datatype(model.graph.input[0].name, target_dt)
    model = model.transform(InferDataTypes())

    # Override the last node's output dtype to target_dt
    last_node_proto = model.graph.node[-1]
    last_inst = getCustomOp(last_node_proto)
    _set_node_output_dtype(last_inst, target_dt)
    model.set_tensor_datatype(last_node_proto.output[0], target_dt)

    # Deserialize back to IR and update the template
    import onnxscript
    loop_body_template._model_proto = model.model
    loop_body_template._ir_model = onnxscript.ir.serde.deserialize_model(model.model)
    loop_body_template._ir_graph = loop_body_template._ir_model.graph

    # Update IR-level finn_datatype metadata on graph inputs/outputs
    # (build_loop_replace_pattern reads these)
    for inp in loop_body_template._ir_graph.inputs:
        if "quant_parameter_tensor_names" not in inp.meta:
            inp.meta["quant_parameter_tensor_names"] = {}
        inp.meta["quant_parameter_tensor_names"]["finn_datatype"] = target_dt_str
    for out in loop_body_template._ir_graph.outputs:
        if "quant_parameter_tensor_names" not in out.meta:
            out.meta["quant_parameter_tensor_names"] = {}
        out.meta["quant_parameter_tensor_names"]["finn_datatype"] = target_dt_str

    # Also update the first/last node's tensor metadata
    first_node = loop_body_template._ir_graph._nodes[0]
    last_node = loop_body_template._ir_graph._nodes[-1]
    first_node.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = target_dt_str
    last_node.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = target_dt_str

    # Rebuild pattern and function from the updated IR graph
    loop_body_template._ir_graph.sort()
    from finn.util import onnxscript_helpers as osh
    loop_body_template.pattern = osh.direct_convert_ir_graph_to_pattern(
        loop_body_template._ir_graph
    )
    loop_body_template.function = loop_body_template._build_ir_function()
    loop_body_template.function_replace = loop_body_template._build_function_replace_pattern()

    log.info(
        f"enforce_loop_body_template_dtype_constraints: "
        f"updated template dtypes to {target_dt_str}"
    )


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
