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
    # Relax dtype requirements for input and output streams by casting
    g = loop_body_template._ir_graph

    # Read finn_datatype strings from the IR graph's input/output metadata
    idt_str = g.inputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]
    odt_str = g.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"]

    log.info(f"match_loop_body_template_dtypes: template idt={idt_str}, odt={odt_str}")

    if idt_str == odt_str:
        log.info("match_loop_body_template_dtypes: dtypes already match, nothing to do")
        return

    idt = DataType[idt_str]
    last_node = g._nodes[-1]

    # Compute the mathematical minimum output dtype from the last node's inputs
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

    # Update the output tensor's finn_datatype metadata
    g.outputs[0].meta["quant_parameter_tensor_names"]["finn_datatype"] = idt_str

    # Update the last node's out_dtype attribute
    last_node = g._nodes[-1]
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
    dtypes match.

    If minimize_bit_width narrowed the loop body output below the input dtype this transformation 
    widens it back. Is automatically ran after step_minimize_bit_width when MLO is active.
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
            if idt == odt:
                continue

            # Loop rolling only succeeds if idt == odt, meaning some later transformation, most
            # likely the minimize bit width transformation, has altered that.
            log.warning(
                f"EnforceLoopBodyDtypeConstraint: FINNLoop {node.name} has "
                f"mismatched body dtypes: input={idt}, output={odt}. "
                f"Widening output back to {idt}."
            )

            # NOTE: I don't think this should be reachable? We know idt == odt was true at some earlier point.
            if not dtype_fits_in(odt, idt):
                raise ValueError(
                    f"EnforceLoopBodyDtypeConstraint: Cannot fix dtype mismatch "
                    f"for FINNLoop {node.name}. Output dtype {odt} does not fit "
                    f"in input dtype {idt}."
                )

            # TODO/NOTE: Ensure the rtl/hdl gen actually uses this type?
            # Widen the last node's output dtype
            if hasattr(last_inst, "set_nodeattr"):
                try:
                    last_inst.set_nodeattr("out_dtype", idt.name)
                except Exception:
                    last_inst.set_nodeattr("outputDataType", idt.name)

            # Update tensor annotation on loop body output
            output_tensor = last_node.output[0]
            loop_body.set_tensor_datatype(output_tensor, idt)

            # Also update the FINNLoop's own outputDataType attribute
            inst.set_nodeattr("outputDataType", idt.name)

        return model, False
