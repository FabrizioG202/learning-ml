Bonito's training data contains three items:
- Chunk data: Normalized ionic current across a fixed 5000-timesteps window. The selected dataset contains 1 million chunks. Thus, the array has a shape of `(1000000, 5000)`
- Reference seqeuence: The corresponding known sequence for each chunk, encoded into integer using the standard encoding of (1=A, 2=C, 3=G, 4=T, 0=padding) padded to a fixed length of 500.
- Reference length: The actual length of each sequence, excluding ending padding.

#figure(
  image(".images/current.svg", width: 100%),
  caption: [
    First 500 timesteps of a sampled chunk
  ],
)
