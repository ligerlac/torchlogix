import torch
from torchlogix.layers import LogicDense, LogicConv2d, GroupSum, OrPooling2d
from torchlogix import Circuit
from torchlogix.utils import set_export_mode

model = torch.nn.Sequential(
    LogicConv2d(in_dim=28, channels=1, num_kernels=16, tree_depth=2, receptive_field_size=3),  # 
    OrPooling2d(kernel_size=2, stride=2, padding=0),  # Reduce dimensionality with pooling operation
    torch.nn.Flatten(),
    LogicDense(16*13*13, 4_000),
    LogicDense(4_000, 4_000),
    GroupSum(k=10, tau=8)  # classify into ten classes
)
print(model)



set_export_mode(model)
circuit = Circuit.from_model(model, input_shape=(1, 28, 28))
print(f"Original circuit: {circuit}")
circuit.simplify()
print(f"Simplified circuit: {circuit}")



example_input = torch.randint(0, 2, (1, 1, 28, 28), dtype=torch.bool)
# explanation of shape: (batch_size, channels, height, width)
# meaning: a single image, a single channel (grayscale), 28x28 pixels
print(f"Example input: {example_input}")




preds_model = model(example_input)
preds_circuit = circuit(example_input)

assert torch.equal(preds_model, preds_circuit), "Predictions from model and circuit do not match!"




and_inverter_graph = circuit.to_and_inverter_graph()
print(and_inverter_graph)
and_inverter_graph.write_to_aiger_file("circuit.aig")

## or: write directly?
circuit.write_to_aiger_file("circuit.aig")

# now we should be able to read it w/ another library (e.g. ABC)
# in the terminal you should now be able to run:
# read_aiger circuit.aig



