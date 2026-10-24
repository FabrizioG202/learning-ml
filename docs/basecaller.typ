= Data Format
We downloaded the training dataset for bonito 10.4.1, which contains data for a million read chunks.
Bonito's training data contains three items:
- Chunk data: Normalized ionic current across a fixed 5000-timesteps window. The selected dataset contains 1 million chunks. Thus, the array has a shape of `(1000000, 5000)`
- Reference seqeuence: The corresponding known sequence for each chunk, encoded into integer using the standard encoding of (1=A, 2=C, 3=G, 4=T, 0=padding) padded to a fixed length of 500.
- Reference length: The actual length of each sequence, excluding ending padding.

The documentation does not contain information about the origin of the reads, and we found that data leakage might occurr if chunks originating from the same run ended up in both the test and train datasets, leading to an over-optimistic test performance. However, having no additonal information on the origin of reads prevents us from accounting for this possibility.

= Model Structure


== CTC Loss
As the loss function, we used `pytorch`'s `CTCLoss`. This loss function wllows training a sequence-to-sequence model without knowing the exact alignment between the input signal and the target sequence. This is a necessary feature to us since we only have access to the raw sequence data for each chunk, without information about the correspondance between signal sample and individual nucleotides is unknown.

#figure(
  image(".images/current.svg", width: 100%),
  caption: [
    First 500 timesteps of a sampled chunk
  ],
)

= Future Directions
The model, as its currently organized, is flexible and can be retrained on different datasets, including modified bases, such as 5mc, 6ma or 4mc. The code could be adapted easily to fit this new scenario, only requiring a slightly bigger model, as we currently produce 5 logits (4 unmofied bases and a blank state) for each timestep.

Moreover, following Bonito's Architecture (#link("https://github.com/nanoporetech/bonito/blob/master/bonito/crf/model.py")[Implementation]) our CTC formulation can be replaced with a CTC-CRF model to better model dependencies between neighbouring bases.

= LLM Disclosure
LLMs were used during the research phase, but every line of code in the `src` folder was written manually by me.
