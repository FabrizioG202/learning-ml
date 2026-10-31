
= Introduction
Third generation sequencing (TGS) has been the cornerstone of a revolution in genomics. When compared with Next Generation Sequencing (NGS), the set of techonlogies which predates it, it has many advantages. Among these, of particular note, read lengths a order of magnitude longer and the possibility of directly detecting epigenetic modifiations on natrive DNA @third_gen_seq. TGS technologies can be divided into three major categories @third_gen_seq2:
- Microscopy-based, where individual DNA molecules are imaged directly.
- Nanopore-based sequecning, where a single molecule of DNA is threaded through a nanopore to produce variation in electrical currents across a membrane.
- Polymerase-based, where the enzyme is observed as it synthetizes a single molecule of DNA

In this work, we focus on the second category, and, within it, Oxford Nanopore Sequencing.

A Basecaller is a piece of software which can translate the raw electrical current signals into nucleotide sequences. One of the most prevalent basecallers is Bonito @bonito, which is based on a deeplearning model implemented in PyTorch. A generally faster architecture and slightly more accurate version of Bonito was also recently presented @fast_bonito.

= Data and Methods

== Raw Data Format
We downloaded the training dataset for bonito 10.4.1, which contains data for a million read chunks.
Bonito's training data contains three items:
- _Chunk data_: Normalized ionic current across a fixed 5000-timesteps window. The selected dataset contains 1 million chunks. Thus, the array has a shape of `(1000000, 5000)`. An example chunk is shown in figure @chunk
- _Reference seqeuence_: The corresponding known sequence for each chunk, encoded into integer using the standard encoding of (1=A, 2=C, 3=G, 4=T, 0=padding) padded to a fixed length of 500.
- _Reference length_: The actual length of each sequence, excluding ending padding.


#figure(
  image(".images/current.svg", width: 100%),
  caption: [
    First 500 timesteps of an example chunk of data.
  ],
)<chunk>

The data does not contain information about the origin of the reads, and we know data leakage might occurr if chunks originating from the same run ended up in both the test and train datasets, leading to an over-optimistic test performance. However, having no additonal information on the origin of reads prevents us from accounting for this possibility.

== Preliminary data analysis
The distribution of sequence lengths in the whole dataset is shown in figure @ref_lengths. We can see how the lengths of the reference sequences are capped at 500 nucleotides, with a mean length of 420 and a minimum of around 300 nucleotides.

#figure(
  image(".images/reference_lengths.svg", width: 100%),
  caption: [
    Distribution of reference sequence lengths over the whole dataset.
  ],
)<ref_lengths>

Nucleotide composition of the reference sequences is shown in figure @composition.
#figure(
  image(".images/nucleotide_composition.svg", width: 100%),
  caption: [
    Nucleotide composition for the whole dataset.
  ],
)<composition>

= Model Structure


== CTC Loss
As the loss function, we used `pytorch`'s `CTCLoss`. This loss function wllows training a sequence-to-sequence model without knowing the exact alignment between the input signal and the target sequence. This is a necessary feature to us since we only have access to the raw sequence data for each chunk, without information about the correspondance between signal sample and individual nucleotides is unknown.

#figure(
  image(".images/current.svg", width: 100%),
  caption: [
    First 500 timesteps of a sampled chunk
  ],
)

= Hyperparameter Tuning
We ran 2 sets of tests to select:
- A model size. Different models structures are reported in figure x and run results are reported in @model_size_selection.
- The value of Learning Rate (lr). Results are reported in @lr_selection.

The metric used for comparison was the average Character Error Rate (CER) computed on the sequences from the validation set.
Due to complexity constraints and long training times, all tests were ran for a maximum of 100 epochs (with early stopping and a patience of $10$) on a dataset size of $10000$, with a train, validation and test split of 80%, 5% and 15%. Entries for this dataset were randomly sampled from the whole $1$ million entry dataset.

#figure(table(
  columns: (1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  inset: 8pt,
  table.header([*Model*], [*Validation CER*], [*Training time*], [*Early stopping epoch*]),
  [Small], [0.2083], [00:31:20], [—],
  [*Medium*], [0.1634], [01:51:39], [60],
  [Large], [0.1719], [01:42:08], [55],
),
caption: [
  Results of Model selection runs. Medium achieved the smallest CER
])<model_size_selection>

#figure(table(
  columns: (1fr, 1fr, 1fr, 1fr),
  align: (center, center, center, center),
  inset: 8pt,
  table.header(
    [*Learning rate*],
    [*Validation CER*],
    [*Training time*],
    [*Early stopping epoch*],
  ),
  [*0.001*], [0.1600], [01:15:33], [41],
  [0.0005], [0.1603], [01:48:30], [59],
  [0.0001], [0.2121], [03:04:52], [—],
),
caption: [
  Results of Learning Rate selection runs. 0.001 achieved the smallest CER
]
)<lr_selection>


= Future Directions
The model, as its currently organized, is flexible and can be retrained on different datasets, including modified bases, such as 5mc, 6ma or 4mc. The code could be adapted easily to fit this new scenario, only requiring a slightly bigger model, as we currently produce 5 logits (4 unmofied bases and a blank state) for each timestep.

Moreover, following Bonito's Architecture (#link("https://github.com/nanoporetech/bonito/blob/master/bonito/crf/model.py")[Implementation]) our CTC formulation can be replaced with a CTC-CRF model to better model dependencies between neighbouring bases.

= LLM Disclosure
LLMs were used during the research phase, but every line of code in the `src` folder (excluding the `vendor` subfolder) and the report was written entirely by me.

#bibliography("basecaller.bib")
