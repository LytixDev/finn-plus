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


class MatchLoopBodyBoundaryDtypes(Transformation):
    """TODO comment

    In MLO, the loop body input and output dtypes must match because the RTL uses a single FM_SIZE 
    (computed from INPUT_BYTES) for both the first-sample and backedge paths. This transformation 
    ensures they match by widening the output dtype of the last node in each repeating block to 
    match the input dtype of the first node, provided it fits.

    Must be run AFTER SetLoopBoundary and BEFORE loop rolling. Automatically run as a part of the 
    loop_rolling step.
    """

    def __init__(self, loop_body_range):
        super().__init__()
        self.start_name = loop_body_range[0].name
        self.end_name = loop_body_range[1].name

    def _get_loop_body_indices(self, graph):
        """Find the first and last indices of loop body nodes by name."""
        start_idx = None
        end_idx = None
        for i, node in enumerate(graph.node):
            if node.name == self.start_name:
                start_idx = i
            if node.name == self.end_name:
                end_idx = i

        if start_idx is None or end_idx is None:
            raise ValueError(
                f"MatchLoopBodyBoundaryDtypes: Could not find loop body range "
                f"nodes in graph: start={self.start_name} "
                f"({'found' if start_idx is not None else 'NOT found'}), "
                f"end={self.end_name} "
                f"({'found' if end_idx is not None else 'NOT found'}). "
                f"Ensure loop_body_range node names match the current model."
            )

        return start_idx, end_idx

    def apply(self, model):
        graph = model.graph
        start_idx, end_idx = self._get_loop_body_indices(graph)

        first_node = graph.node[start_idx]
        first_inst = getCustomOp(first_node)
        idt = first_inst.get_input_datatype(0)

        log.info(f"MatchLoopBodyBoundaryDtypes: loop body input dtype = {idt}")

        last_node = graph.node[end_idx]
        last_inst = getCustomOp(last_node)
        current_odt = last_inst.get_output_datatype(0)

        log.info(
            f"MatchLoopBodyBoundaryDtypes: loop body output dtype = {current_odt} "
            f"(last node: {last_node.op_type} {last_node.name})"
        )

        if current_odt == idt:
            log.info("MatchLoopBodyBoundaryDtypes: dtypes already match, nothing to do")
            return model, False

        # Compute the true minimal output dtype if the node supports it
        if hasattr(last_inst, "_derive_out_dtype"):
            minimal_odt = last_inst._derive_out_dtype(model)
        else:
            minimal_odt = current_odt

        log.info(f"MatchLoopBodyBoundaryDtypes: true minimal output dtype = {minimal_odt}")

        if not dtype_fits_in(minimal_odt, idt):
            raise ValueError(
                f"MatchLoopBodyBoundaryDtypes: Cannot match loop body dtypes. "
                f"Minimal output dtype {minimal_odt} does not fit in input dtype "
                f"{idt}. The loop body output range [{minimal_odt.min()}, "
                f"{minimal_odt.max()}] exceeds input range [{idt.min()}, "
                f"{idt.max()}]."
            )

        # Widen output dtype of the last node in EVERY repeating block.
        # The loop body range covers all blocks. We identify block boundaries
        # by finding nodes of the same op_type as the last node within the range.
        # Each such node is the last node of a block.
        # TODO: VERIFY
        changed = False
        for i in range(start_idx, end_idx + 1):
            node = graph.node[i]
            if node.op_type == last_node.op_type:
                inst = getCustomOp(node)
                node_odt = inst.get_output_datatype(0)
                if node_odt != idt:
                    # Check this node's minimal output also fits
                    if hasattr(inst, "_derive_out_dtype"):
                        node_minimal = inst._derive_out_dtype(model)
                    else:
                        node_minimal = node_odt

                    if not dtype_fits_in(node_minimal, idt):
                        log.warning(
                            f"MatchLoopBodyBoundaryDtypes: Skipping {node.name}, "
                            f"minimal dtype {node_minimal} does not fit in {idt}"
                        )
                        continue

                    # Set the output dtype to match the input
                    if hasattr(inst, "set_nodeattr"):
                        try:
                            inst.set_nodeattr("out_dtype", idt.name)
                        except Exception:
                            inst.set_nodeattr("outputDataType", idt.name)
                    # Update tensor annotation
                    output_tensor = node.output[0]
                    model.set_tensor_datatype(output_tensor, idt)
                    log.info(
                        f"MatchLoopBodyBoundaryDtypes: Widened {node.name} output "
                        f"from {node_odt} to {idt}"
                    )
                    changed = True

        return model, changed


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

            # NOTE: I don't think this should be unreachable? We know idt == odt was true at some earlier point.
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
