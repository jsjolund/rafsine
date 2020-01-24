require "NodeDescriptor"

-- Defines the D3Q6 node used for the temperature
D3Q6Descriptor =
  NodeDescriptor(
  3,
  6,
  {
    --main axis
    {1, 0, 0},
    {-1, 0, 0},
    {0, 1, 0},
    {0, -1, 0},
    {0, 0, 1},
    {0, 0, -1}
  },
  {
    1.0 / 6,
    1.0 / 6,
    1.0 / 6,
    1.0 / 6,
    1.0 / 6,
    1.0 / 6
  }
)
