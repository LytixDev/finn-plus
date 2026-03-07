"""Insert DWCs at FINNLoop body boundaries to ensure byte-aligned stream widths.

The MLO infrastructure uses AXI memory-mapped DMA transfers for intermediate
feature maps between loop iterations. DMA operates in bytes, so the stream
widths at the loop body boundary (where data enters/exits the DMA path) must
be multiples of 8 bits. This transformation inserts StreamingDataWidthConverter
nodes at the loop body input and output when the stream width is not byte-aligned.
"""

import logging

from onnx import TensorProto, helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple

log = logging.getLogger(__name__)


class InsertLoopBodyDWC(Transformation):
    """Insert DWCs at FINNLoop body boundaries for byte-aligned stream widths.

    The MLO loop control infrastructure requires byte-aligned stream widths at 
    the loop body boundary because intermediate results are transferred via AXI-MM DMA.
    This transformation inserts DWC nodes (if needed) at the output and input of the loop body 
    subgraph to ensure the stream is byte-aligned.
    """

    def apply(self, model: ModelWrapper):
        graph_modified = False

        for node in model.graph.node:
            if node.op_type != "FINNLoop":
                continue

            inst = getCustomOp(node)
            loop_body = inst.get_nodeattr("body")
            body_model = ModelWrapper(loop_body)

            body_modified = False

            # Output boundary
            last_node = loop_body.graph.node[-1]
            last_inst = getCustomOp(last_node)
            out_width = last_inst.get_outstream_width(0)

            if out_width % 8 != 0:
                padded_width = roundup_to_integer_multiple(out_width, 8)
                log.info(
                    f"FINNLoop output stream width {out_width} is not byte-aligned, "
                    f"inserting DWC to widen to {padded_width}"
                )

                out_dtype = last_inst.get_output_datatype(0)
                out_shape = list(last_inst.get_normal_output_shape(0))

                # The current graph output tensor name
                old_output_name = loop_body.graph.output[0].name
                # New intermediate tensor between last node and DWC
                dwc_input_name = old_output_name + "_pre_dwc"

                # Rename the last node's output to the intermediate name so we can splice the dwc in between
                for idx, out in enumerate(last_node.output):
                    if out == old_output_name:
                        last_node.output[idx] = dwc_input_name

                # Create value_info for the intermediate tensor
                dwc_input_vi = oh.make_tensor_value_info(
                    dwc_input_name, TensorProto.FLOAT, out_shape
                )
                loop_body.graph.value_info.append(dwc_input_vi)
                body_model.set_tensor_datatype(dwc_input_name, out_dtype)

                dwc_node = oh.make_node(
                    "StreamingDataWidthConverter",
                    [dwc_input_name],
                    [old_output_name],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name=f"{node.name}_output_dwc",
                    inShape=out_shape,
                    outShape=out_shape,
                    inWidth=out_width,
                    outWidth=padded_width,
                    dataType=str(out_dtype.name),
                )
                loop_body.graph.node.append(dwc_node)
                body_modified = True

            # Input boundary
            first_node = loop_body.graph.node[0]
            first_inst = getCustomOp(first_node)
            in_width = first_inst.get_instream_width(0)

            if in_width % 8 != 0:
                padded_width = roundup_to_integer_multiple(in_width, 8)
                log.info(
                    f"FINNLoop input stream width {in_width} is not byte-aligned, "
                    f"inserting DWC to narrow from {padded_width} to {in_width}"
                )

                in_dtype = first_inst.get_input_datatype(0)
                in_shape = list(first_inst.get_normal_input_shape(0))

                # The current graph input tensor name (activation input, index 0)
                old_input_name = loop_body.graph.input[0].name
                # New intermediate tensor between DWC and first node
                dwc_output_name = old_input_name + "_post_dwc"

                # Rename the first node's output to the intermediate name so we can splice the dwc in between
                for idx, inp in enumerate(first_node.input):
                    if inp == old_input_name:
                        first_node.input[idx] = dwc_output_name

                # Create value_info for the intermediate tensor
                dwc_output_vi = oh.make_tensor_value_info(
                    dwc_output_name, TensorProto.FLOAT, in_shape
                )
                loop_body.graph.value_info.append(dwc_output_vi)
                body_model.set_tensor_datatype(dwc_output_name, in_dtype)

                dwc_node = oh.make_node(
                    "StreamingDataWidthConverter",
                    [old_input_name],
                    [dwc_output_name],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name=f"{node.name}_input_dwc",
                    inShape=in_shape,
                    outShape=in_shape,
                    inWidth=padded_width,
                    outWidth=in_width,
                    dataType=str(in_dtype.name),
                )
                # Insert at the beginning of the graph
                loop_body.graph.node.insert(0, dwc_node)
                body_modified = True

            if body_modified:
                # Write modified loop body back to the FINNLoop node
                # TODO: is this needed
                inst.set_nodeattr("body", body_model.model.graph)
                graph_modified = True

        return (model, graph_modified)
